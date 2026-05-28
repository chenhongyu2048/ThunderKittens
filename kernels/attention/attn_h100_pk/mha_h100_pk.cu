// # Define TORCH_COMPILE macro

#include "kittens.cuh"
#include "prototype.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;
template<int D, int NUM_WORKERS> struct attn_fwd_layout {
    using qo_tile   = st_bf<64, D>;
    using kv_tile   = st_bf<D==64?192:128, D>;
    using qo_global = kittens::gl<bf16, -1, -1, -1, D, qo_tile>;
    using kv_global = kittens::gl<bf16, -1, -1, -1, D, kv_tile>;
    struct globals { qo_global O, Q; kv_global K, V; };
    struct input_block    { kv_tile k, v; };
    struct scratch_block  { qo_tile q[NUM_WORKERS]; };
    struct common_state   { int batch, q_head, kv_head, seq, kv_iters; };
    struct consumer_state {
        rt_fl<16, qo_tile::cols> o_reg;
        col_vec<rt_fl<16, kv_tile::rows>> max_vec, norm_vec;
        col_vec<rt_fl<16, kv_tile::rows>> max_vec_last_scaled, max_vec_scaled;
        rt_fl<16, kv_tile::rows> att_block;
        rt_bf<16, kv_tile::rows> att_block_mma;
    };
};
template<int D, bool is_causal = false> struct attn_fwd_template {
    static constexpr int NUM_CONSUMER_WARPS = 8, NUM_WORKERS = NUM_CONSUMER_WARPS/4, INPUT_PIPE_STAGES = 2;
    using layout = attn_fwd_layout<D, NUM_WORKERS>;
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        int task_id = gridDim.x*args.task_iter + blockIdx.x;
        const int q_heads  = args.globals.Q.depth();
        const int kv_heads = args.globals.K.depth();
        const int group_size = q_heads / kv_heads;
        int seq_q = (args.globals.Q.rows() + NUM_WORKERS*layout::qo_tile::rows - 1)/(NUM_WORKERS*layout::qo_tile::rows);
        args.common.batch = task_id / (seq_q * q_heads);
        task_id -= args.common.batch * seq_q * q_heads;
        args.common.q_head = task_id / seq_q;
        task_id -= args.common.q_head * seq_q;
        args.common.seq = task_id;
        args.common.kv_head = args.common.q_head / group_size;
        // for causal=false:
        // args.num_iters = args.common.batch < args.globals.Q.batch() ? (args.globals.K.rows() + layout::kv_tile::rows - 1)/(layout::kv_tile::rows) : -1;
        // for causal=true:
        if (args.common.batch < args.globals.Q.batch()) {
            if constexpr (is_causal) {
                const int q_start = args.common.seq * NUM_WORKERS * layout::qo_tile::rows;
                const int q_end_exclusive = q_start + NUM_WORKERS * layout::qo_tile::rows;
                int max_k_needed = q_end_exclusive - 1;
                const int k_rows = args.globals.K.rows();

                if (max_k_needed >= k_rows) {
                    max_k_needed = k_rows - 1;
                }

                args.common.kv_iters = (max_k_needed + layout::kv_tile::rows) / layout::kv_tile::rows;
            }
            else {
                args.common.kv_iters = (args.globals.K.rows() + layout::kv_tile::rows - 1) / layout::kv_tile::rows;
            }
            args.num_iters = args.common.kv_iters;
        }
        else {
            args.common.kv_iters = -1;
            args.num_iters = -1;
        }
    }
    struct producer {
        __device__ static inline void setup(producer_setup_args<layout> args) {
            // warpgroup::producer_registers();
        }
        __device__ static inline void load(producer_load_args<layout> args) {
            if(warpgroup::warpid() == 0) {
                warp::tma::expect(args.inputs_arrived_k, args.input.k);
                warp::tma::load_async(args.input.k, args.globals.K, {args.common.batch, args.common.kv_head, args.iter, 0}, args.inputs_arrived_k);
                warp::tma::expect(args.inputs_arrived_v, args.input.v);
                warp::tma::load_async(args.input.v, args.globals.V, {args.common.batch, args.common.kv_head, args.iter, 0}, args.inputs_arrived_v);
            }
            else if(laneid() == 0) {
                arrive(args.inputs_arrived_k);
                arrive(args.inputs_arrived_v);
            }
        }
    };
    struct consumer {
        __device__ static inline void setup(consumer_setup_args<layout> args) {
            // warpgroup::consumer_registers<NUM_WORKERS>();
            if((args.common.seq*NUM_WORKERS + warpgroup::groupid())*layout::qo_tile::rows < args.globals.Q.rows()) // out of bounds?
                warpgroup::load(args.scratch.q[warpgroup::groupid()], args.globals.Q,
                                {args.common.batch, args.common.q_head, args.common.seq*NUM_WORKERS+warpgroup::groupid(), 0});
            args.state.o_reg = 0.f;
            args.state.norm_vec = 0.f;
            args.state.max_vec = base_types::constants<float>::neg_infty();
            warpgroup::sync(warpgroup::groupid());
        }
        __device__ static inline void compute(consumer_compute_args<layout> args, uint32_t& semaphore_bitfield, int& input_ring) {
            wait(args.inputs_arrived_k, get_phasebit<0>(semaphore_bitfield, input_ring)); // wait for memory to arrive, phase changes at half the rate of the ring
            constexpr float TEMPERATURE_SCALE = (D == 128) ? 0.08838834764f*1.44269504089f : 0.125f*1.44269504089f;
            // A = Q @ K.T
            warpgroup::mm<transpose::N, transpose::T>(args.state.att_block, args.scratch.q[warpgroup::groupid()], args.input.k);
            args.state.max_vec_last_scaled = args.state.max_vec * TEMPERATURE_SCALE;
            warpgroup::mma_async_wait();
            // causal mask
            if constexpr (is_causal) {
                constexpr int SUBTILE = kittens::TILE_ROW_DIM<bf16>;
                const int q_tile_idx = args.common.seq * NUM_WORKERS + warpgroup::groupid();
                const int q_warp_start = q_tile_idx * layout::qo_tile::rows + warpgroup::warpid() * SUBTILE;
                const int k_tile_end = args.iter * layout::kv_tile::rows + layout::kv_tile::rows - 1;
                if (k_tile_end > q_warp_start) {
                    const int q_blk = q_warp_start / SUBTILE;
                    const int k_blk_base = args.iter * (layout::kv_tile::rows / SUBTILE);
                    #pragma unroll
                    for (int j = 0; j < layout::kv_tile::rows / SUBTILE; j++) {
                        const int k_blk = k_blk_base + j;
                        auto &attn_subtile = reinterpret_cast<rt_fl<16, 16>&>(args.state.att_block.tiles[0][j]);
                        if (k_blk > q_blk) {
                            warp::neg_infty(attn_subtile);
                        }
                        else if (k_blk == q_blk) {
                            warp::make_causal(attn_subtile, attn_subtile, kittens::base_types::constants<float>::neg_infty());
                        }
                        __syncwarp();
                    }
                }
            }
            // softmax
            warp::right_fill(args.state.att_block, args.state.att_block, args.globals.K.rows() - args.iter*layout::kv_tile::rows, base_types::constants<float>::neg_infty());
            args.state.max_vec = warp::max<axis::COL>(args.state.att_block, args.state.max_vec); // accumulate onto the max_vec
            args.state.max_vec_scaled = args.state.max_vec * TEMPERATURE_SCALE;
            args.state.att_block = warp::exp2((args.state.att_block*TEMPERATURE_SCALE) - args.state.max_vec_scaled);
            args.state.max_vec_last_scaled = warp::exp2(args.state.max_vec_last_scaled - args.state.max_vec_scaled);
            args.state.norm_vec *= args.state.max_vec_last_scaled;
            args.state.norm_vec = warp::sum<axis::COL>(args.state.att_block, args.state.norm_vec); // accumulate onto the norm_vec
            args.state.o_reg *= args.state.max_vec_last_scaled; // normalize o_reg before mma
            args.state.att_block_mma = args.state.att_block; // convert to bf16 for mma
            // O += A @ V
            wait(args.inputs_arrived_v, get_phasebit<0>(semaphore_bitfield, input_ring)); // wait for memory to arrive, phase changes at half the rate of the ring
            update_phasebit<0>(semaphore_bitfield, input_ring);
            warpgroup::mma<transpose::N, transpose::N>(args.state.o_reg, args.state.att_block_mma, args.input.v);
            warpgroup::mma_async_wait();
            if(laneid() == 0) arrive(args.inputs_finished); // done!
        }
        __device__ static inline void finish(consumer_finish_args<layout> args) {
            if((args.common.seq*NUM_WORKERS+warpgroup::groupid())*layout::qo_tile::rows < args.globals.Q.rows()) { // out of bounds?
                args.state.o_reg /= args.state.norm_vec;
                auto &o_smem = reinterpret_cast<typename layout::qo_tile&>(args.scratch.q[warpgroup::groupid()]);
                warpgroup::store(o_smem, args.state.o_reg);
                warpgroup::sync(warpgroup::groupid());
                if(warpgroup::warpid() == 0)
                    warp::tma::store_async(args.globals.O, o_smem, {args.common.batch, args.common.q_head, args.common.seq*NUM_WORKERS+warpgroup::groupid(), 0});
                warp::tma::store_async_read_wait();
            }
            __syncwarp();
            if(laneid() == 0) arrive(args.finish_finished); // done!
        }
    };
};

#ifdef TORCH_COMPILE

#include "pyutils/torchutils.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Functions.h>
#include <iostream>

std::vector<at::Tensor> 
attention_forward(at::Tensor q, at::Tensor k, at::Tensor v, bool causal)
{
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);

    auto batch    = q.size(0);
    auto seq_len  = q.size(2);
    auto head_dim = q.size(3);
    auto is_causal = causal;
    auto qo_heads = q.size(1);
    auto kv_heads = k.size(1);

    // check to see that these dimensions match for all inputs
    TORCH_CHECK(q.size(0) == batch, "Q batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(k.size(0) == batch, "K batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(v.size(0) == batch, "V batch dimension - idx 0 - must match for all inputs");

    TORCH_CHECK(q.size(2) == seq_len, "Q sequence length dimension - idx 2 - must match for all inputs");
    TORCH_CHECK(k.size(2) == seq_len, "K sequence length dimension - idx 2 - must match for all inputs");
    TORCH_CHECK(v.size(2) == seq_len, "V sequence length dimension - idx 2 - must match for all inputs");

    TORCH_CHECK(q.size(3) == head_dim, "Q head dimension - idx 3 - must match for all non-vector inputs");
    TORCH_CHECK(k.size(3) == head_dim, "K head dimension - idx 3 - must match for all non-vector inputs");
    TORCH_CHECK(v.size(3) == head_dim, "V head dimension - idx 3 - must match for all non-vector inputs");

    TORCH_CHECK(qo_heads >= kv_heads, "QO heads must be greater than or equal to KV heads");
    TORCH_CHECK(qo_heads % kv_heads == 0, "QO heads must be divisible by KV heads");
    TORCH_CHECK(q.size(1) == qo_heads, "QO head dimension - idx 1 - must match for all inputs");
    TORCH_CHECK(k.size(1) == kv_heads, "KV head dimension - idx 1 - must match for all inputs");
    TORCH_CHECK(v.size(1) == kv_heads, "KV head dimension - idx 1 - must match for all inputs");  

    auto hr = qo_heads / kv_heads;

    c10::BFloat16* q_ptr = q.data_ptr<c10::BFloat16>();
    c10::BFloat16* k_ptr = k.data_ptr<c10::BFloat16>();
    c10::BFloat16* v_ptr = v.data_ptr<c10::BFloat16>();

    bf16*  d_q = reinterpret_cast<bf16*>(q_ptr);
    bf16*  d_k = reinterpret_cast<bf16*>(k_ptr);
    bf16*  d_v = reinterpret_cast<bf16*>(v_ptr);
    
    // for the returned outputs
    at::Tensor o     = at::empty({static_cast<const uint>(batch), 
                                        static_cast<const uint>(qo_heads), 
                                        static_cast<const uint>(seq_len), 
                                        static_cast<const uint>(head_dim)}, v.options());
    
    at::Tensor l_vec = at::empty({static_cast<long>(batch),
                                        static_cast<long>(qo_heads),
                                        static_cast<long>(seq_len),
                                        static_cast<long>(1)},
                                        q.options().dtype(at::kFloat));
        

    bf16*  o_ptr = reinterpret_cast<bf16*>(o.data_ptr<c10::BFloat16>());
    bf16*  d_o   = reinterpret_cast<bf16*>(o_ptr);

    float* l_ptr = reinterpret_cast<float*>(l_vec.data_ptr<float>());
    float* d_l   = reinterpret_cast<float*>(l_ptr);

    cudaDeviceSynchronize();
    auto stream = at::cuda::getCurrentCUDAStream().stream(); 

    auto launch = [&]<int HEAD_DIM, bool IS_CAUSAL>() {
        using ker_template = attn_fwd_template<HEAD_DIM, IS_CAUSAL>;
        using layout = typename ker_template::layout;

        typename layout::qo_global Qg(d_q, (size_t)batch, (size_t)qo_heads, (size_t)seq_len, nullptr);
        typename layout::kv_global Kg(d_k, (size_t)batch, (size_t)kv_heads, (size_t)seq_len, nullptr);
        typename layout::kv_global Vg(d_v, (size_t)batch, (size_t)kv_heads, (size_t)seq_len, nullptr);
        typename layout::qo_global Og(d_o, (size_t)batch, (size_t)qo_heads, (size_t)seq_len, nullptr);
        typename layout::globals globals = {Og, Qg, Kg, Vg};

        unsigned long mem_size = kittens::MAX_SHARED_MEMORY - 2000;
        cudaFuncSetAttribute(
            prototype::lcf::kernel<ker_template>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            mem_size
        );

        constexpr int BLOCK_SIZE = prototype::detail::NUM_THREADS_v<ker_template>;
        dim3 grid(132, 1, 1);
        prototype::lcf::kernel<ker_template><<<grid, BLOCK_SIZE, mem_size, stream>>>(globals);

        CHECK_CUDA_ERROR(cudaGetLastError());
        cudaStreamSynchronize(stream);
    };

    if (head_dim == 64) {
        if (is_causal) launch.template operator()<64, true>();
        else           launch.template operator()<64, false>();
    } else if (head_dim == 128) {
        if (is_causal) launch.template operator()<128, true>();
        else           launch.template operator()<128, false>();
    }

    return {o, l_vec};
    cudaDeviceSynchronize();
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mha_forward",  attention_forward, "Bidirectional forward MHA. Takes Q,K,V,O in (B,H,N,D) where D must be 64 or 128, and N must be a multiple of 64. Additionally writes out norm vector L of shape (B,H,N), used in backward pass.");
}

#else

#include "harness.impl"

#endif