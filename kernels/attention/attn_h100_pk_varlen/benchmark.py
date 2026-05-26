import torch
import numpy as np
import argparse
import pandas as pd

import _C as tk

try:
    from flash_attn import flash_attn_varlen_func as fa2_varlen_func
    print("Successfully imported flash_attn (FA2 varlen)")
except ImportError:
    fa2_varlen_func = None
    print("Could not import flash_attn. pip install flash-attn for FA2 comparison")

try:
    from flash_attn_interface import _flash_attn_forward as _fa3_fwd
    fa3_available = True
    print("Successfully imported flash_attn_interface (FA3)")
except ImportError:
    _fa3_fwd = None
    fa3_available = False
    print("Could not import flash_attn_interface for FA3 comparison")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def get_varlen_flops(seqlens, nheads, headdim, causal):
    """Sum of flops across variable-length sequences."""
    total = 0
    for n in seqlens:
        total += 4 * n * n * nheads * headdim
    if causal:
        total //= 2
    return total


def make_varlen_inputs(seqlens, qo_heads, kv_heads, d, dtype=torch.bfloat16):
    """Create packed QKV tensors and cu_seqlens for varlen attention."""
    total_len = sum(seqlens)
    cu_seqlens = torch.zeros(len(seqlens) + 1, dtype=torch.int32, device='cuda')
    for i, s in enumerate(seqlens):
        cu_seqlens[i + 1] = cu_seqlens[i] + s

    q = torch.randn(1, qo_heads, total_len, d, dtype=dtype, device='cuda')
    k = torch.randn(1, kv_heads, total_len, d, dtype=dtype, device='cuda')
    v = torch.randn(1, kv_heads, total_len, d, dtype=dtype, device='cuda')
    return q, k, v, cu_seqlens


def bench_tk(q, k, v, cu_seqlens, causal, num_warmup=50, num_iters=50):
    """Benchmark TK varlen attention."""
    for _ in range(num_warmup):
        tk.mha_forward(q, k, v, cu_seqlens, causal)
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

    for i in range(num_iters):
        start_events[i].record()
        tk.mha_forward(q, k, v, cu_seqlens, causal)
        end_events[i].record()
    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return np.mean(times)


def bench_fa2(q, k, v, cu_seqlens, causal, num_warmup=10, num_iters=10):
    """Benchmark FA2 varlen attention."""
    if fa2_varlen_func is None:
        return None
    total_len = q.shape[2]
    qo_heads = q.shape[1]
    kv_heads = k.shape[1]
    d = q.shape[3]
    max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())

    q_fa2 = q.squeeze(0).transpose(0, 1).contiguous().view(total_len, qo_heads, d)
    k_fa2 = k.squeeze(0).transpose(0, 1).contiguous().view(total_len, kv_heads, d)
    v_fa2 = v.squeeze(0).transpose(0, 1).contiguous().view(total_len, kv_heads, d)

    try:
        for _ in range(num_warmup):
            fa2_varlen_func(q_fa2, k_fa2, v_fa2, cu_seqlens, cu_seqlens,
                            max_seqlen, max_seqlen, causal=causal)
        torch.cuda.synchronize()

        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

        for i in range(num_iters):
            start_events[i].record()
            fa2_varlen_func(q_fa2, k_fa2, v_fa2, cu_seqlens, cu_seqlens,
                            max_seqlen, max_seqlen, causal=causal)
            end_events[i].record()
        torch.cuda.synchronize()

        times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        return np.mean(times)
    except Exception as e:
        print(f"  FA2 error: {e}")
        return None


def bench_fa3(q, k, v, cu_seqlens, causal, num_warmup=10, num_iters=10):
    """Benchmark FA3 varlen attention using _flash_attn_forward."""
    if not fa3_available:
        return None
    total_len = q.shape[2]
    qo_heads = q.shape[1]
    kv_heads = k.shape[1]
    d = q.shape[3]
    max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
    softmax_scale = d ** (-0.5)

    q_fa3 = q.squeeze(0).transpose(0, 1).contiguous().view(total_len, qo_heads, d)
    k_fa3 = k.squeeze(0).transpose(0, 1).contiguous().view(total_len, kv_heads, d)
    v_fa3 = v.squeeze(0).transpose(0, 1).contiguous().view(total_len, kv_heads, d)

    try:
        for _ in range(num_warmup):
            _fa3_fwd(q_fa3, k_fa3, v_fa3,
                     softmax_scale=softmax_scale, causal=causal,
                     window_size_left=-1, window_size_right=-1, softcap=0.0,
                     cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                     max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen)
        torch.cuda.synchronize()

        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

        for i in range(num_iters):
            start_events[i].record()
            _fa3_fwd(q_fa3, k_fa3, v_fa3,
                     softmax_scale=softmax_scale, causal=causal,
                     window_size_left=-1, window_size_right=-1, softcap=0.0,
                     cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                     max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen)
            end_events[i].record()
        torch.cuda.synchronize()

        times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        return np.mean(times)
    except Exception as e:
        print(f"  FA3 error: {e}")
        return None


def run_benchmark(seqlens, qo_heads, kv_heads, d, causal):
    """Run benchmark for a given set of sequence lengths."""
    q, k, v, cu_seqlens = make_varlen_inputs(seqlens, qo_heads, kv_heads, d)
    flops = get_varlen_flops(seqlens, qo_heads, d, causal)

    results = {}

    tk_time = bench_tk(q, k, v, cu_seqlens, causal)
    results['TK'] = (tk_time, flops / (tk_time * 1e-3) / 1e12)

    fa2_time = bench_fa2(q, k, v, cu_seqlens, causal)
    if fa2_time is not None:
        results['FA2'] = (fa2_time, flops / (fa2_time * 1e-3) / 1e12)

    fa3_time = bench_fa3(q, k, v, cu_seqlens, causal)
    if fa3_time is not None:
        results['FA3'] = (fa3_time, flops / (fa3_time * 1e-3) / 1e12)

    torch.cuda.empty_cache()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Varlen attention benchmark")
    parser.add_argument("--qo_head", type=int, default=32)
    parser.add_argument("--kv_head", type=int, default=32)
    parser.add_argument("--d", type=int, default=128, choices=[64, 128])
    parser.add_argument("--num_seqs", type=int, default=16)
    parser.add_argument("--causal", action="store_true", default=True)
    parser.add_argument("--no_causal", action="store_true")
    parser.add_argument("--mode", choices=["uniform", "variable", "both"], default="both")
    args = parser.parse_args()

    causal = not args.no_causal
    print(f"Config: qo_head={args.qo_head}, kv_head={args.kv_head}, d={args.d}, "
          f"num_seqs={args.num_seqs}, causal={causal}")
    print("=" * 80)

    uniform_lengths = [384, 768, 1536, 3072, 6144, 12288]
    all_results = []

    if args.mode in ("uniform", "both"):
        print("\n[Uniform lengths] All sequences have the same length")
        for seq_len in uniform_lengths:
            seqlens = [seq_len] * args.num_seqs
            total = sum(seqlens)
            print(f"\n  seq_len={seq_len}, num_seqs={args.num_seqs}, total_tokens={total}")
            results = run_benchmark(seqlens, args.qo_head, args.kv_head, args.d, causal)
            row = {'seq_len': seq_len, 'total_tokens': total}
            for method, (time_ms, tflops) in results.items():
                print(f"    {method:4s}: {time_ms:8.3f} ms | {tflops:7.1f} TFLOPS")
                row[f'{method}_ms'] = f'{time_ms:.3f}'
                row[f'{method}_tflops'] = f'{tflops:.1f}'
            all_results.append(row)

    if args.mode in ("variable", "both"):
        print("\n[Variable lengths] Random sequence lengths (multiples of 384)")
        np.random.seed(42)
        possible_lens = [384 * i for i in range(1, 17)]
        for trial in range(5):
            seqlens = [int(x) for x in np.random.choice(possible_lens, size=args.num_seqs)]
            total = sum(seqlens)
            desc = f"lens={seqlens[:4]}..."
            print(f"\n  Trial {trial+1}: {desc} (total={total})")
            results = run_benchmark(seqlens, args.qo_head, args.kv_head, args.d, causal)
            row = {'seq_len': f'var_trial_{trial+1}', 'total_tokens': total}
            for method, (time_ms, tflops) in results.items():
                print(f"    {method:4s}: {time_ms:8.3f} ms | {tflops:7.1f} TFLOPS")
                row[f'{method}_ms'] = f'{time_ms:.3f}'
                row[f'{method}_tflops'] = f'{tflops:.1f}'
            all_results.append(row)

    print("\n" + "=" * 80)
    df = pd.DataFrame(all_results)
    if not df.empty:
        print("\nTime (ms):")
        time_cols = ['seq_len', 'total_tokens'] + [c for c in df.columns if c.endswith('_ms')]
        print(df[time_cols].to_string(index=False))
        print("\nTFLOPS:")
        tflops_cols = ['seq_len', 'total_tokens'] + [c for c in df.columns if c.endswith('_tflops')]
        print(df[tflops_cols].to_string(index=False))
