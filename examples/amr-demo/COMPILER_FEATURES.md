# Mojo Compiler Advanced Features for AMR

> **Status: AI-assisted, unverified.** This document describes optimizations
> that the Mojo/MLIR toolchain *can* apply to indirect memory-access patterns
> and how they would appear in lowered IR. The specific transformations are
> **not confirmed** for this demo's code — no emitted IR, assembly, or hardware
> counters were captured to prove they fire. Read every "the compiler does X"
> statement below as "the compiler may do X; here is how to check." The MLIR
> snippets are illustrative pseudo-IR, not captured compiler output. See the
> README's [Validation checklist](README.md#validation-checklist).

This document walks through how the Mojo compiler's MLIR-based infrastructure
*could* optimize the indirect memory access patterns in Adaptive Mesh Refinement
simulations, and how to verify whether it actually does.

## 1. Indirect Pattern Detection in MLIR

The Mojo compiler detects indirect access patterns during the SSA (Static
Single Assignment) lowering phase.

### Pattern in Source Code

```mojo
# CSR graph traversal - double indirection
for k in range(offsets[i], offsets[i+1]):
    var neighbor_id = neighbor_ids[k]      # First indirection
    sum += temperatures[neighbor_id]        # Second indirection
```

### How Compiler Detects This

1. **SSA Form Analysis**: Compiler converts to static single assignment:

   ```mlir
   %neighbor_id = memref.load %neighbor_ids[%k] : memref<?xi64>
   %temp = memref.load %temperatures[%neighbor_id] : memref<?xf32>
   ```

2. **Dependency Chain**: MLIR identifies the data dependency:
   - Load from `neighbor_ids` produces a value
   - That value is used as an index to `temperatures`
   - This is a classic gather pattern

3. **Affine Analysis**: MLIR's affine dialect analyzes:
   - Is the access pattern regular? (No - indirect through neighbor_ids)
   - Can it be vectorized? (Yes - with gather instructions)
   - Are there memory dependencies? (Read-only, can parallelize)

## 2. Affine Access Map Recognition

### Pattern

```mojo
# Strided access with constant stride
result[i] = data[STRIDE * i]
```

### MLIR Representation

```mlir
affine.for %i = 0 to %N {
  %idx = affine.apply affine_map<(d0) -> (d0 * 4)>(%i)
  %val = memref.load %data[%idx] : memref<?xf32>
  memref.store %val, %result[%i] : memref<?xf32>
}
```

### Optimizations Applied

1. **Strength Reduction**: `i * 4` → `i << 2` (multiply to shift)
2. **Vectorization**: SIMD loads with stride-4 pattern
3. **Loop Unrolling**: `@parameter` enables compile-time unrolling
4. **Address Calculation**: Pre-compute base + stride * i

## 3. Coalesced Memory Access (CSR)

The SoA (Structure-of-Arrays) layout enables coalesced GPU access:

### Memory Layout

```
temperatures: [T0, T1, T2, T3, T4, T5, ...] ← Contiguous array
neighbor_ids: [2, 5, 1, 4, 3, 0, ...]       ← Index array
```

### GPU Thread Access Pattern

```
Thread 0: temperatures[neighbor_ids[0]] = temperatures[2]
Thread 1: temperatures[neighbor_ids[1]] = temperatures[5]
Thread 2: temperatures[neighbor_ids[2]] = temperatures[1]
Thread 3: temperatures[neighbor_ids[3]] = temperatures[4]
```

### Compiler Optimizations

1. **Warp-Level Analysis**:
   - Compiler detects 32 threads (warp) access temperatures array
   - Even though indices are irregular, the *array itself* is coalesced
   - Memory transactions are minimized per warp

2. **Cache Line Batching**:
   - If neighbor_ids have spatial locality (common in AMR):

     ```
     neighbor_ids = [100, 101, 105, 102, ...]  ← Some clustering
     ```

   - Compiler can batch these into fewer cache line fetches
   - MLIR's `memref.prefetch` hints inserted automatically

3. **Shared Memory Staging** (GPU):
   - For high-reuse neighbors, compiler may generate:

     ```mlir
     // Pseudo-code for what compiler might generate
     gpu.shared_memory %temps_cache : memref<128xf32>
     gpu.barrier()
     // Load collaboratively into shared memory
     // Then indirect access from fast shared memory
     ```

## 4. Asynchronous Prefetch Queue Generation

### Software Prefetching

The compiler can insert prefetch instructions when it detects:

1. Sequential or predictable patterns
2. High memory latency (e.g., DRAM access)
3. Computation that can hide latency

### Example IR Transformation

**Before (source code)**:

```mojo
for i in range(N):
    var idx = indices[i]
    result[i] = expensive_compute(data[idx])
```

**After (compiler inserts prefetch)**:

```mlir
// Prime prefetch queue
affine.for %p = 0 to min(8, %N) {
  llvm.prefetch %data[%indices[%p]], 0, 3, 1
}

affine.for %i = 0 to %N {
  // Prefetch ahead
  if %i + 8 < %N {
    %future_idx = memref.load %indices[%i + 8]
    llvm.prefetch %data[%future_idx], 0, 3, 1
  }

  // Do actual work
  %idx = memref.load %indices[%i]
  %val = memref.load %data[%idx]
  %computed = call @expensive_compute(%val)
  memref.store %computed, %result[%i]
}
```

### Prefetch Parameters

- `0` = read prefetch (vs 1 = write)
- `3` = high temporal locality
- `1` = data cache (vs 0 = instruction cache)

## 5. How to Verify These Optimizations

### Inspect MLIR IR

```bash
mojo build --emit-mlir amr.mojo > amr.mlir
```

Look for:

- `affine.for` loops (affine analysis working)
- `vector.gather` / `vector.scatter` (SIMD indirect access)
- `llvm.prefetch` intrinsics (automatic prefetch insertion)
- `gpu.barrier` / `gpu.shared_memory` (GPU optimizations)

### Check Generated Assembly

```bash
mojo build --emit-assembly amr.mojo > amr.s
```

Look for:

- `vgather` instructions (AVX2/AVX-512 gather)
- `prefetcht0` / `prefetchnta` (x86 prefetch)
- `prfm` (ARM prefetch)
- Vectorized loops (SIMD registers like `ymm`, `zmm`)

### Profile with Hardware Counters

```bash
# Linux perf
perf stat -e cache-misses,cache-references,L1-dcache-load-misses ./amr

# Look for:
# - Low cache miss rate despite irregular access
# - High IPC (instructions per cycle) = good latency hiding
# - Memory-level parallelism (multiple outstanding loads)
```

## 6. GPU-Specific Optimizations

### Coalescing Detection

The compiler analyzes warp-level memory access:

```mojo
// GPU kernel - 256 threads per block
var tid = Int(thread_idx.x + block_idx.x * block_dim.x)
var neighbor_id = neighbor_ids[offsets[tid]]  # Indirect load
var temp = temperatures[neighbor_id]           # Gather
```

### MLIR Analysis

1. **Warp Divergence**: Are threads in same warp accessing nearby memory?
2. **Coalescing**: Even if indices differ, are they in same 128-byte cache line?
3. **Transaction Reduction**: Minimize memory transactions per warp

### Example Optimization

If neighbor graph has spatial locality:

```
Block 0, Threads 0-31 access:  temperatures[100], [104], [101], [105], ...
→ Compiler sees these fit in ~2 cache lines
→ 2 memory transactions instead of 32
→ 16× reduction in memory traffic!
```

### How Compiler Achieves This

- Analyzes `offsets[tid]` pattern across warp
- If clustered, keeps access pattern
- If scattered, may reorder computation or use shared memory
- Inserts `gpu.barrier()` where needed for correctness

## 7. Indicators in the AMR Demo Results

These are preliminary single-run measurements (no warmup/repetition/scaling
curves). They are *consistent with* the optimizations above but do not, on their
own, prove the compiler applied any specific transformation — that requires
IR/assembly/profiler evidence.

### CPU (preliminary)

- **Cache blocking: large speedup (~144–180× across runs)** — uses a
  compile-time `tile_size`; the exact figure is mesh- and machine-dependent
- **Explicit SIMD: ~14× speedup** — uses `SIMD[DType, width]`
- **Prefetch: ~1.1× speedup** — small and within noise; needs repetition to confirm

### GPU (preliminary, AMD W7900 only)

- **CSR unstructured: faster than CPU baseline (~5.9× here)** despite double indirection
- Roughly similar throughput to structured (26–28 Mcells/sec) on this small mesh
- *Consistent with* GPU parallelism hiding irregular-access latency; not yet
  isolated from other effects

### Working interpretation (to be verified)

A plausible explanation is that the compiler, rather than making irregular
access "regular", may:

1. Detect the gather pattern
2. Generate gather/scatter operations
3. Insert prefetch to hide latency
4. On GPU, rely on parallelism to mask memory stalls

Each step is a hypothesis — confirm with emitted IR/assembly and profiler counters.

## 8. Comparison with Traditional Compilers

> The contrasts below are general expectations plus single-run numbers from this
> demo's specific C++ build (GCC 15.2, `-O3 -march=native`). They are not a
> controlled language benchmark; treat the throughput figures as preliminary.

### GCC/Clang (this demo's reference build)

- Indirect-access auto-vectorization is limited in general
- Often needs `#pragma omp simd` hints for gather
- CSR access in this build: ~**100 Mcells/sec** (a hand-written reference)

### Mojo/MLIR (this demo)

- `SIMD[DType]` requests explicit vectorization at the source level
- Single-source kernels, run here on AMD GPU (NVIDIA untested)
- CSR on GPU: ~**27 Mcells/sec** in this run, without AMD-specific tuning
- CPU structured with explicit tiling: ~**900 Mcells/sec** in this run (faster
  than this particular C++ build on this machine — not a general claim)

## Summary

The Mojo compiler's MLIR infrastructure *can* provide (each item still needs
IR/assembly/profiler confirmation for this code):

1. **Indirect pattern detection** — via SSA and affine analysis
2. **Affine access map optimization** — strength reduction, vectorization
3. **Coalesced memory batching** — GPU warp analysis, cache-line optimization
4. **Prefetch insertion** — `llvm.prefetch` for predictable patterns

The AMR demo provides preliminary measurements *consistent with* these, but does
not prove them on its own:

- Large CPU speedup (~144–180× across runs) from source-level optimizations
- GPU faster than CPU baseline on irregular CSR (~6× here) despite double indirection
- Single-source kernels run on CPU and AMD GPU (x86 + RDNA3); ARM and NVIDIA untested here

**The notable point**: Mojo exposes these optimization levers explicitly
(`@parameter`, `SIMD[DType]`, `prefetch()`) at the source level. Whether the
compiler then applies the lowerings described above should be verified, not
assumed.
