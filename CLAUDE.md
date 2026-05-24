# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working Directory

Primary writable working folder: `kernels/parallel/ring_attn`, `kernels/attention/attn_h100_pk`

## What This Is

ThunderKittens (TK) is a header-only CUDA C++ library for writing fast deep learning kernels using tile primitives. It targets NVIDIA Hopper (SM90) and Blackwell (SM100/SM103/SM120) GPUs. Include `kittens.cuh` and you're set — no installation step.

## Build Commands

### Unit tests (library primitives)

```bash
cd tests/
make -j32 ARCH=SM90          # default: intensity 2, ~3000 tests
mkdir -p outputs && ./unit_tests printout
```

Makefile knobs: `ARCH` (SM80/SM90/SM100/SM103/SM120), `TEST_INTENSITY` (1-4), `COMP_LEVEL` (fast/debug/profile). Target specific tests with e.g. `-DTEST_WARP_MEMORY` instead of `-DTEST_ALL`.

### Individual kernels

Each kernel lives in its own directory under `kernels/` with its own Makefile:

```bash
cd kernels/gemm/bf16_h100/
make              # compile
make run          # compile + execute
make ncu          # profile with Nsight Compute
make nsys         # profile with Nsight Systems
```

Kernel Makefiles set `ARCH`, `SRC`, `OUT`, `CMD`, and `CONFIG` (standalone | python | pytorch), then include `../../common.mk`.

### Multi-GPU kernels

```bash
cd kernels/parallel/ag_gemm/
export ARCH=SM90
make && make run  # CMD typically uses torchrun --nproc_per_node=8
```

### Environment setup

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
```

Requires CUDA 12.8+, C++20 (gcc-11+), and for Python/PyTorch kernels: PyTorch 2.8+, pybind11.

## Architecture

### Library structure (`include/`)

The library is organized by scope hierarchy matching NVIDIA's programming model:

- `common/` — Base types (`bf16`, `fp8`, `fp4`), base ops, utilities
- `types/` — Data type definitions at each memory level:
  - `register/` — `rt` (register tiles), `rv` (register vectors), with layout types
  - `shared/` — `st` (shared tiles), `sv` (shared vectors), with swizzled layouts
  - `global/` — `gl` (global layouts), TMA descriptor helpers
  - `tensor/` — `tt` (tensor memory tiles, Blackwell TMEM)
  - `system/` — `pgl` (parallel global layouts for multi-GPU), IPC, VMM
- `ops/` — Operations organized by scope:
  - `thread/` — Single-thread TMA ops, sync primitives
  - `group/` — Warp/warpgroup-level: memory transfers, MMA, maps, reductions
- `pyutils/` — PyTorch/pybind11 integration helpers

### Prototype templates (`prototype/`)

Structured kernel patterns that handle producer/consumer synchronization:

- **LCF** (Load-Compute-Finish) — The primary template. You define a `layout` struct, a `producer` (loads data via TMA), and a `consumer` (computes + stores). The template handles pipelining, barriers, and persistent grid scheduling.
- **LCSC** (Load-Compute-Store-Compute) — Two-phase compute variant
- **LCSF** (Load-Compute-Store-Finish) — Store then finish variant
- **Interpreter** — Dynamic dispatch variant

To use a prototype template, define a struct satisfying the concept (layout + producer + consumer), then launch with `kittens::prototype::lcf::kernel<YourTemplate><<<grid, block>>>`.

### Kernel conventions

- Each kernel directory is self-contained: `.cu` source, `Makefile`, and test/benchmark `.py` files
- Standalone kernels (`CONFIG=standalone`) produce a binary; Python/PyTorch kernels produce a `.so` shared object
- Correctness tests generate reference data via Python (`gentests.py` or inline in `test.py`), then compare against kernel output
- Multi-GPU kernels under `kernels/parallel/` use `torchrun` for launch

### Key type naming conventions

- `st_bf<R,C>` — shared tile, bf16, R rows x C cols
- `rt_fl<R,C>` — register tile, float32
- `rt_bf<R,C>` — register tile, bf16
- `gl<type, B, H, D, S, tile>` — global layout descriptor (batch, head, depth, seq dims; -1 = dynamic)
- `pgl` — parallel global layout (multi-GPU, wraps gl + IPC handles)

### Compilation defines

Exactly one must be set: `KITTENS_SM80`, `KITTENS_SM90`, `KITTENS_SM100`, `KITTENS_SM103`, `KITTENS_SM120`. The `common.mk` build system handles this via the `ARCH` variable.

### Critical nvcc flags

Always use `-O3 --use_fast_math` for kernel performance. The `common.mk` sets these automatically. The `--expt-extended-lambda --expt-relaxed-constexpr` flags are required for TK's template-heavy code.
