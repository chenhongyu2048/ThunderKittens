#pragma once

#include "../include/kittens.cuh"
#include "../common/common.cuh"
#include "templates.cuh"

namespace kittens {
namespace prototype {
namespace lcf {

template<typename lcft> concept kernel_template = requires {
    typename lcft::layout;
    typename lcft::producer;
    typename lcft::consumer;
    lcft::common_setup;
    lcft::producer::setup;
    lcft::producer::load;
    lcft::consumer::setup;
    lcft::consumer::compute_prologue;
    lcft::consumer::compute_mainloop_masked;
    lcft::consumer::compute_mainloop_unmasked;
    lcft::consumer::compute_epilogue;
    lcft::consumer::finish;
} && kittens_layout<typename lcft::layout>;

template<typename lcft> // load-compute-store-finish template
__global__ __launch_bounds__(detail::NUM_THREADS_v<lcft>, detail::NUM_BLOCKS_v<lcft>)
#ifdef KITTENS_SM10X
__cluster_dims__(detail::CLUSTER_BLOCKS_v<lcft>)
#endif
void kernel(const __grid_constant__ typename lcft::layout::globals globals) {
    static_assert(kernel_template<lcft>, "lcf kernel template parameter does not satisfy concept requirements");
    using L              = typename lcft::layout;
    using CKL            = complete_kittens_layout<L>; // complete the layout by filling in the optional types with empty
    using common_state   = typename CKL::common_state_t;
    using producer_state = typename CKL::producer_state_t;
    using consumer_state = typename CKL::consumer_state_t;
    using input_block    = typename CKL::input_block_t;
    using input_block_v  = typename CKL::input_block_v_t;
    using scratch_block  = typename CKL::scratch_block_t;
    using finish_block   = typename CKL::finish_block_t;
    using input_alloc_block   = typename CKL::input_alloc_block_t;
    using input_alloc_block_v = typename CKL::input_alloc_block_v_t;
    using scratch_alloc_block = typename CKL::scratch_alloc_block_t;
    constexpr int MAX_SHARED_MEMORY = detail::MAX_SHARED_MEMORY_v<lcft>;
    constexpr int INPUT_PIPE_STAGES = detail::INPUT_PIPE_STAGES_v<lcft>;
    static_assert(INPUT_PIPE_STAGES >= 1 && INPUT_PIPE_STAGES <= 16, "Invalid number of input pipe stages");
    static_assert(
        INPUT_PIPE_STAGES*(sizeof(input_alloc_block)+sizeof(input_alloc_block_v)) + sizeof(scratch_alloc_block)
        <= MAX_SHARED_MEMORY-1024, "Shared memory usage exceeds limits"
    );
    constexpr int NUM_CONSUMER_WARPS = detail::NUM_CONSUMER_WARPS_v<lcft>;
    constexpr int NUM_PRODUCER_WARPS = detail::NUM_PRODUCER_WARPS_v<lcft>;

#ifdef KITTENS_SM10X
    constexpr int NCTA_TENSOR_ALLOC = detail::CLUSTER_BLOCKS_v<lcft> > 1 ? 2 : 1;
    tensor_allocator<detail::NUM_BLOCKS_v<lcft>, NCTA_TENSOR_ALLOC> tensor_alloc{};
#endif
    
    extern __shared__ int __shm[];
    shared_allocator alloc(&__shm[0]); // allocate shared memory
    scratch_alloc_block (&scratch_smem)              = alloc.allocate<scratch_alloc_block>();
    input_alloc_block   (&k_smem)[INPUT_PIPE_STAGES] = alloc.allocate<input_alloc_block, INPUT_PIPE_STAGES>();
    input_alloc_block_v (&v_smem)[INPUT_PIPE_STAGES] = alloc.allocate<input_alloc_block_v, INPUT_PIPE_STAGES>();

    // figure out where we're going to put the finish block
    constexpr int FINISH_BLOCK_OFFSET = (MAX_SHARED_MEMORY-1024)/detail::NUM_BLOCKS_v<lcft> - sizeof(finish_block);
    static_assert(FINISH_BLOCK_OFFSET >= 0, "Finish block is too large for shared memory.");
    constexpr int NON_FINISH_BLOCK_SPACE = FINISH_BLOCK_OFFSET - 1024 - sizeof(scratch_alloc_block); // including the losses from alignment
    constexpr int COMBINED_INPUT_SIZE = sizeof(input_alloc_block) + sizeof(input_alloc_block_v);
    constexpr int SAFE_STAGES_BETWEEN_BLOCKS = NON_FINISH_BLOCK_SPACE/COMBINED_INPUT_SIZE < INPUT_PIPE_STAGES
                                             ? NON_FINISH_BLOCK_SPACE/COMBINED_INPUT_SIZE : INPUT_PIPE_STAGES;
    finish_block (*finish_smem) = reinterpret_cast<finish_block*>((((uint64_t)&__shm[0] + FINISH_BLOCK_OFFSET)/1024)*1024); // alignment

    if constexpr (detail::DEBUG_v<lcft>) {
        if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
            printf("DEBUG REPORT FOR PRODUCER TEMPLATE KERNEL:\n");
            printf("    BLOCK INFORMATION\n");
            printf("        gridDim.x:                         %d\n", gridDim.x);
            printf("        gridDim.y:                         %d\n", gridDim.y);
            printf("        gridDim.z:                         %d\n", gridDim.z);
            printf("        blockDim.x:                        %d\n", blockDim.x);
            printf("        blockDim.y:                        %d\n", blockDim.y);
            printf("        blockDim.z:                        %d\n", blockDim.z);
            printf("        num_blocks per SM:                 %d\n", detail::NUM_BLOCKS_v<lcft>);
            printf("        num_threads per SM:                %d\n", detail::NUM_THREADS_v<lcft>);
            printf("        num_warps per SM:                  %d\n", detail::NUM_WARPS_v<lcft>);
            printf("        num_consumer_warpgroups:           %d\n", detail::NUM_CONSUMER_WARPGROUPS_v<lcft>);
            printf("        num_consumer_warps:                %d\n", detail::NUM_CONSUMER_WARPS_v<lcft>);
            printf("        num_producer_warps:                %d\n", detail::NUM_PRODUCER_WARPS_v<lcft>);
            printf("    PIPELINE INFORMATION\n"); 
            printf("        input_pipe_stages:                 %d\n", INPUT_PIPE_STAGES);
            printf("        safe_stages_between_blocks:        %d\n", SAFE_STAGES_BETWEEN_BLOCKS);
            printf("    SHARED MEMORY INFORMATION\n"); 
            printf("        input_smem block size:             %llu\n", sizeof(input_block));
            printf("        input_smem block size (aligned):   %llu\n", sizeof(input_alloc_block));
            printf("        input_smem block_v size:           %llu\n", sizeof(input_block_v));
            printf("        input_smem block_v size (aligned): %llu\n", sizeof(input_alloc_block_v));
            printf("        k_smem:                            %p\n", (void*)&k_smem);
            printf("        v_smem:                            %p\n", (void*)&v_smem);
            printf("        input_smem size:                   %llu\n", INPUT_PIPE_STAGES*sizeof(input_alloc_block));
            printf("        input_smem_v size:                 %llu\n", INPUT_PIPE_STAGES*sizeof(input_alloc_block_v));
            printf("        scratch_smem block size:           %llu\n", sizeof(scratch_block));
            printf("        scratch_smem block size (aligned): %llu\n", sizeof(scratch_alloc_block));
            printf("        scratch_smem:                      %p\n", (void*)&scratch_smem);
            printf("        finish_smem:                       %p\n", (void*)finish_smem);
            printf("        finish_smem size:                  %llu\n", sizeof(finish_block));
            printf("        dynamic shared memory usage:       %llu\n", sizeof(scratch_alloc_block) + uint64_t(&scratch_smem) - uint64_t(&__shm[0]));
        }
        everyone::sync(15);
    }

    // Initialize semaphores. This is constant for all two-stage producer-consumer kernels.
    __shared__ kittens::semaphore inputs_arrived_k[INPUT_PIPE_STAGES], inputs_arrived_v[INPUT_PIPE_STAGES];
    __shared__ kittens::semaphore inputs_finished_k[INPUT_PIPE_STAGES], inputs_finished_v[INPUT_PIPE_STAGES];
    __shared__ kittens::semaphore finish_finished;
    uint32_t semaphore_bitfield_k = 0xFFFF0000;
    uint32_t semaphore_bitfield_v = 0xFFFF0000;
    common_state common;

    if(warpid() >= NUM_CONSUMER_WARPS) { // code path for producer warps
        warpgroup::producer_registers();
        using producers = group<NUM_PRODUCER_WARPS>;
        if (warpid() == NUM_CONSUMER_WARPS) { // a single warp (in fact a single thread) does these.
            for(int i = 0; i < INPUT_PIPE_STAGES; i++) {
                init_semaphore(inputs_arrived_k[i], detail::PRODUCER_BARRIER_ARRIVALS_v<lcft>, 0); // needs to wait on each producer warp
                init_semaphore(inputs_arrived_v[i], detail::PRODUCER_BARRIER_ARRIVALS_v<lcft>, 0); // needs to wait on each producer warp
                init_semaphore(inputs_finished_k[i], detail::CONSUMER_BARRIER_ARRIVALS_v<lcft>, 0); // needs to wait on each consumer warp
                init_semaphore(inputs_finished_v[i], detail::CONSUMER_BARRIER_ARRIVALS_v<lcft>, 0); // needs to wait on each consumer warp
            }
            init_semaphore(finish_finished, detail::CONSUMER_BARRIER_ARRIVALS_v<lcft>, 0); // consumer warps must say they are done with the finish block
        }
        // all warps must arrive here, confirming semaphore initialization is visible to all threads.
        if constexpr (detail::CLUSTER_BLOCKS_v<lcft> > 1) everyone::tma::cluster::sync();
        else everyone::sync(15);
        producer_state p_state;
        for(int task_iter = 0; true; task_iter++) {
            int num_iters = -1;
#ifdef KITTENS_SM10X
            common_setup_args<L> unif{common, task_iter, num_iters, globals, *scratch_smem, tensor_alloc};
#else
            common_setup_args<L> unif{common, task_iter, num_iters, globals, *scratch_smem};
#endif
            lcft::common_setup(unif);
            if(num_iters < 0) break; // no work to do
            int k_ring = 0, v_ring = 0; // tracking which input block is being loaded
            int load_iter;
            lcft::producer::setup({p_state, unif});
            for(load_iter = 0; load_iter < SAFE_STAGES_BETWEEN_BLOCKS && load_iter < num_iters; load_iter++) { // fill the pipeline
                wait(inputs_finished_k[k_ring], get_phasebit<1>(semaphore_bitfield_k, k_ring));
                update_phasebit<1>(semaphore_bitfield_k, k_ring);
                wait(inputs_finished_v[v_ring], get_phasebit<1>(semaphore_bitfield_v, v_ring));
                update_phasebit<1>(semaphore_bitfield_v, v_ring);
                lcft::producer::load({p_state, *k_smem[k_ring], *v_smem[v_ring],
                                      inputs_arrived_k[k_ring], inputs_arrived_v[v_ring], num_iters - 1 - load_iter, unif}); // load in reverse order
                k_ring = ring_advance<INPUT_PIPE_STAGES>(k_ring);
                v_ring = ring_advance<INPUT_PIPE_STAGES>(v_ring);
            }
            wait(finish_finished, (task_iter%2)^1); // wait for consumer to finish their finish stage before we can do the rest.
            for(; load_iter < num_iters; load_iter++) { // fill the pipeline
                wait(inputs_finished_k[k_ring], get_phasebit<1>(semaphore_bitfield_k, k_ring));
                update_phasebit<1>(semaphore_bitfield_k, k_ring);
                wait(inputs_finished_v[v_ring], get_phasebit<1>(semaphore_bitfield_v, v_ring));
                update_phasebit<1>(semaphore_bitfield_v, v_ring);
                lcft::producer::load({p_state, *k_smem[k_ring], *v_smem[v_ring],
                                      inputs_arrived_k[k_ring], inputs_arrived_v[v_ring], num_iters - 1 - load_iter, unif});
                k_ring = ring_advance<INPUT_PIPE_STAGES>(k_ring);
                v_ring = ring_advance<INPUT_PIPE_STAGES>(v_ring);
            }
            producers::sync(13); // producer warps must finish before consumer warps can proceed
        } // task iter loop
    } // producer warpgroup
    else { // code path for consumer warps
        warpgroup::consumer_registers<NUM_CONSUMER_WARPS / 4>();
        using consumers = group<NUM_CONSUMER_WARPS>;
        // all warps must arrive here, confirming semaphore initialization is visible to all threads.
        if constexpr (detail::CLUSTER_BLOCKS_v<lcft> > 1) everyone::tma::cluster::sync();
        else everyone::sync(15);
        lcft::consumer::warp_scheduler_barrier_init();
        consumer_state c_state;
        for(int task_iter = 0; true; task_iter++) {
            int num_iters = -1;
#ifdef KITTENS_SM10X
            common_setup_args<L> unif{common, task_iter, num_iters, globals, *scratch_smem, tensor_alloc};
#else
            common_setup_args<L> unif{common, task_iter, num_iters, globals, *scratch_smem};
#endif
            lcft::common_setup(unif);
            if(num_iters < 0) break; // no work to do
            // calculate num of mask/unmask blocks
            int n_unmasked;
            if constexpr (lcft::IS_CAUSAL) {
                const int q_min_start = common.seq * (NUM_CONSUMER_WARPS / 4) * L::qo_tile::rows;
                n_unmasked = q_min_start / L::kv_tile::rows;
                if (n_unmasked > num_iters - 1)
                    n_unmasked = num_iters - 1; // minus 1 for first prologue iteration
            } else {
                n_unmasked = num_iters - 1; // minus 1 for first prologue iteration
            }
            const int n_masked_main = (num_iters - 1) - n_unmasked;
            // setup consumer state
            int k_ring = 0, v_ring = 0;
            lcft::consumer::setup({c_state, unif});
            // Prologue: iter num_iters-1, only K
            {
                lcft::consumer::compute_prologue({c_state, *k_smem[k_ring],
                                                inputs_arrived_k[k_ring], inputs_finished_k[k_ring],
                                                num_iters - 1, unif},
                                                semaphore_bitfield_k, k_ring);
                k_ring = ring_advance<INPUT_PIPE_STAGES>(k_ring);
            }
            // ---- Mainloop (masked) ----
            int j = 1; // record how many iterations we've done
            #pragma unroll 1
            for (int m = 0; m < n_masked_main; m++, j++) {
                lcft::consumer::compute_mainloop_masked({c_state, *k_smem[k_ring], *v_smem[v_ring],
                                                         inputs_arrived_k[k_ring], inputs_arrived_v[v_ring],
                                                         inputs_finished_k[k_ring], inputs_finished_v[v_ring],
                                                         num_iters - 1 - j, unif},
                                                        semaphore_bitfield_k, semaphore_bitfield_v,
                                                        k_ring, v_ring);
                k_ring = ring_advance<INPUT_PIPE_STAGES>(k_ring);
                v_ring = ring_advance<INPUT_PIPE_STAGES>(v_ring);
            }
            // ---- Mainloop (no mask) ----
            #pragma unroll 1
            for (int m = 0; m < n_unmasked; m++, j++) {
                lcft::consumer::compute_mainloop_unmasked({c_state, *k_smem[k_ring], *v_smem[v_ring],
                                                         inputs_arrived_k[k_ring], inputs_arrived_v[v_ring],
                                                         inputs_finished_k[k_ring], inputs_finished_v[v_ring],
                                                         num_iters - 1 - j, unif},
                                                        semaphore_bitfield_k, semaphore_bitfield_v,
                                                        k_ring, v_ring);
                k_ring = ring_advance<INPUT_PIPE_STAGES>(k_ring);
                v_ring = ring_advance<INPUT_PIPE_STAGES>(v_ring);
            }
            // Epilogue: last V
            {
                lcft::consumer::compute_epilogue({c_state, *v_smem[v_ring],
                                                inputs_arrived_v[v_ring], inputs_finished_v[v_ring],
                                                unif},
                                                semaphore_bitfield_v, v_ring);
                v_ring = ring_advance<INPUT_PIPE_STAGES>(v_ring);
            }
            // consumers::sync(14); // not needed
            lcft::consumer::finish({c_state, *finish_smem, finish_finished, unif});
            // consumers::sync(14); // not needed
        } // task iter loop
    } // consumer warpgroup
    // all warps must arrive here, confirming semaphore initialization is visible to all threads.
    if constexpr (detail::CLUSTER_BLOCKS_v<lcft> > 1) everyone::tma::cluster::sync();
#ifdef KITTENS_SM10X
    else everyone::sync(15);
#endif
}

} // namespace lcf
} // namespace prototype
} // namespace kittens
