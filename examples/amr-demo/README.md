# Adaptive Mesh Refinement (AMR) Demo

This proof-of-concept explores GPU-capable Adaptive Mesh Refinement for
computational physics simulations using Mojo.

## Provenance and scope

This is an AI-assisted exploratory prototype directed by Luis Chamberlain to
evaluate whether Mojo is a useful vehicle for expressing AMR-style structured
and irregular memory-access kernels across CPU/GPU targets. The prototype is
intended for performance-engineering exploration, generated-code review, and
validation workflow design. It should not be treated as a production AMR
framework or as evidence of hand-written C++ HPC expertise.

See [Limitations](#limitations) and [Validation checklist](#validation-checklist)
before drawing conclusions from any numbers in this document.

## Overview

This demo shows how Mojo can be used for large-scale scientific computing applications that require:

- **Dynamic mesh adaptation** based on physics-driven criteria
- **GPU-ready data structures** with Structure-of-Arrays (SoA) layout
- **Sparse graph operations** for mesh connectivity
- **Parallel execution** on both CPU and GPU

## Running the Demo

### Execute the AMR Demo

```bash
# From the amr-demo directory
mojo amr.mojo
```

This runs:

1. **Performance benchmarks** comparing different Mojo compiler optimization levels
2. **C++ comparison** against the reference implementation
3. **Interactive AMR simulation** with dynamic mesh refinement

### Generate Performance Visualizations

```bash
# Generate benchmark charts (requires matplotlib)
python3 generate_benchmark_plot.py
```

This creates:

- `amr_benchmark_results.png` - CPU throughput comparison charts
- `amr_speedup_comparison.png` - CPU speedup analysis visualization
- `amr_gpu_performance.png` - GPU vs CPU performance comparison
- `amr_complete_comparison.png` - Complete overview (CPU/GPU/C++)

### Compile and Run C++ Benchmark

```bash
# Compare against C++ reference implementation
g++ -O3 -march=native -fopenmp -std=c++17 amr_benchmark.cpp -o amr_benchmark
./amr_benchmark
```

**Expected Output** (captured from a single run on an AMD W7900; the
"AMD RDNA/CDNA + NVIDIA" banner printed by the program reflects the *intended*
portability target — only the AMD path was actually executed):

```
======================================================================
Adaptive Mesh Refinement: Mojo Compiler Optimization Demo
======================================================================

[1] STRUCTURED mesh (auto-vectorization)...
    Time: 0.082 s | Throughput: 5.02 Mcells/sec

[2] UNSTRUCTURED mesh (auto-vectorization)...
    Time: 0.088 s | Throughput: 4.68 Mcells/sec

[3] STRUCTURED mesh (explicit SIMD + @parameter unrolling)...
    Time: 0.006 s | Throughput: 69.42 Mcells/sec

[4] UNSTRUCTURED mesh (explicit SIMD + compile-time loop unroll)...
    Time: 0.080 s | Throughput: 5.12 Mcells/sec

[5] UNSTRUCTURED mesh (prefetch + @always_inline + compile-time math)...
    Time: 0.080 s | Throughput: 5.14 Mcells/sec

[6] STRUCTURED mesh (cache-blocked with compile-time tile size)...
    Time: 0.001 s | Throughput: 724.47 Mcells/sec

======================================================================
Performance Gains from Explicit Compiler Control:
======================================================================
  Structured mesh:
    Baseline (auto-vec):   5.02 Mcells/sec
    + Explicit SIMD:      69.42 Mcells/sec ( 13.8 x)
    + Cache blocking:    724.47 Mcells/sec (144.2 x)

  Unstructured mesh:
    Baseline (auto-vec):   4.68 Mcells/sec
    + prefetch + inline:   5.14 Mcells/sec (  1.1 x)

======================================================================
GPU Acceleration (AMD RDNA/CDNA + NVIDIA)
======================================================================
✓ GPU detected and initialized

[7] STRUCTURED mesh (GPU)...
    Time: 0.014 s | Throughput: 28.67 Mcells/sec

[8] UNSTRUCTURED mesh (GPU - CSR indirect access)...
    Time: 0.015 s | Throughput: 26.66 Mcells/sec

GPU Performance Analysis:
  Structured mesh:
    CPU (auto-vec):    5.01 Mcells/sec
    CPU (tiled):     900.73 Mcells/sec
    GPU:              28.67 Mcells/sec (  5.7 x vs CPU baseline)

  Unstructured mesh (double-indirect CSR):
    CPU (auto-vec):    4.49 Mcells/sec
    GPU:              26.66 Mcells/sec (  5.9 x vs CPU baseline)

Key Result:
  GPU handles double-indirect CSR access with 5.9x speedup
  Same Mojo code for CPU and GPU - portable and composable!
======================================================================

[Interactive Demo: AMR with Dynamic Refinement]
Final Mesh: 1548 total cells | 1417 active | Levels: 0-3
======================================================================
```

## Key Features

### 1. Adaptive Mesh Refinement

- **Quadtree refinement**: Each cell splits into 4 children when gradient threshold exceeded
- **Physics-driven**: Refinement criterion based on local temperature gradients
- **Multi-level**: Supports up to 4 refinement levels (configurable)
- **Dynamic**: Mesh evolves during simulation

### 2. GPU-Ready Data Structure

```mojo
struct AMRMesh:
    # Structure-of-Arrays layout for coalesced GPU memory access
    var ids: UnsafePointer[Int]           # Stable IDs
    var levels: UnsafePointer[Int]        # Refinement levels
    var x_coords: UnsafePointer[Float32]  # X coordinates
    var y_coords: UnsafePointer[Float32]  # Y coordinates
    var temperatures: UnsafePointer[Float32]  # Physics field
    ...
```

**Benefits:**

- Coalesced memory access on GPU
- Easy transfer to/from device memory
- Optimal SIMD vectorization
- Cache-friendly on CPU

### 3. Sparse Graph Operations

- **CSR format**: Compressed Sparse Row for neighbor connectivity
- **Efficient traversal**: O(1) neighbor lookup
- **Scalable**: Handles irregular mesh topology
- **GPU-friendly**: Standard format for sparse operations

```mojo
# CSR neighbor graph
var neighbor_offsets: UnsafePointer[Int]  # Length: num_cells + 1
var neighbor_ids: UnsafePointer[Int]      # Concatenated neighbors

# Access neighbors of cell i:
for k in range(neighbor_offsets[i], neighbor_offsets[i+1]):
    var neighbor = neighbor_ids[k]
    ...
```

### 4. Physics Solver

**Heat Diffusion Equation**: ∂T/∂t = α∇²T

- **Explicit time integration**: Forward Euler
- **5-point stencil**: For Laplacian operator
- **Parallel execution**: CPU threads (GPU-portable)

## Technical Details

### Memory Layout Comparison

**Traditional Array-of-Structs (AoS):**

```
Bad for GPU: Non-coalesced access
[cell0.x, cell0.y, cell0.T, | cell1.x, cell1.y, cell1.T, | ...]
```

**Structure-of-Arrays (SoA) - This demo:**

```
Good for GPU: Coalesced access
x_coords:     [cell0.x, cell1.x, cell2.x, ...]
y_coords:     [cell0.y, cell1.y, cell2.y, ...]
temperatures: [cell0.T, cell1.T, cell2.T, ...]
```

### Refinement Algorithm

1. **Gradient Computation**: ∇T estimated from neighbors

   ```mojo
   grad = √(Σ(T_neighbor - T_center)² / n_neighbors)
   ```

2. **Marking Phase**: Cells with grad > threshold marked for refinement

3. **Refinement Phase**:
   - Parent cell marked inactive
   - 4 children created in quadrants (SW, SE, NW, NE)
   - Children inherit parent's temperature
   - Neighbor graph updated

4. **Physics Update**: Diffusion step on refined mesh

### Parallel Execution

Uses Mojo's `parallelize` for CPU threading:

```mojo
@parameter
fn kernel(i: Int):
    # Per-cell computation
    ...

parallelize[kernel](num_cells, num_cells)
```

**GPU Port**: Replace with `gpu.host.DeviceContext` kernels

## Scalability to GPU

### Current (CPU)

- ✅ SoA data layout (GPU-ready)
- ✅ Parallel kernels via `parallelize`
- ✅ Coalesced memory access patterns
- ✅ CSR sparse graph

### Next Steps for GPU

1. **Port to GPU kernels**:

   ```mojo
   from gpu.host import DeviceContext

   with DeviceContext() as ctx:
       var mesh_dev = ctx.enqueue_create_buffer[DType.float32](num_cells)
       ctx.enqueue_function_checked[diffusion_kernel, ...](...)
   ```

2. **On-device refinement**:
   - Mark cells in parallel
   - Compact refinement list
   - Atomic append for children

3. **Multi-GPU scaling**:
   - Domain decomposition
   - Halo exchange for boundary cells
   - MPI + GPU

## HPC Application Areas

This AMR framework is applicable to a wide range of high-performance scientific computing domains:

### 1. **Computational Fluid Dynamics (CFD)**

- Shock capturing in compressible flow
- Interface tracking (multi-material)
- Turbulence modeling
- Aerodynamic simulations

### 2. **Astrophysics and Cosmology**

- Stellar explosions (supernovae)
- Accretion disks
- Galaxy formation simulations
- N-body gravitational dynamics

### 3. **Climate and Weather Modeling**

- Atmospheric dynamics
- Ocean circulation
- Multi-scale climate phenomena
- Storm and hurricane tracking

### 4. **Materials Science and Engineering**

- High-energy density physics
- Phase transitions and solidification
- Radiation transport
- Multi-physics coupling (thermal-mechanical-electromagnetic)

## Performance Considerations

### Memory Efficiency

- **Compact storage**: Only active cells stored
- **Sparse graph**: Only actual neighbors stored
- **Reusable buffers**: Minimal allocation during timestepping

### Computational Efficiency

- **Parallel kernels**: O(N) operations parallelized
- **Cache-friendly**: SoA layout + sequential access
- **Minimal synchronization**: Independent cell updates

### GPU Optimization Opportunities

- **Warp-level primitives**: For reductions/scans
- **Shared memory**: For neighbor access
- **Tensor cores**: For dense linear algebra (if needed)
- **Multi-GPU**: For large-scale problems (>1B cells)

## Comparison to Existing AMR Frameworks

This table positions the *language/tooling characteristics* being explored
against established, production-grade C++ AMR frameworks. It is **not** a
feature-parity claim: Chombo, SAMRAI, and AMReX are mature, validated codebases
and this prototype is not. "This prototype" columns describe what was attempted
here, not equivalent capability.

| Aspect | This prototype | Chombo | SAMRAI | AMReX |
|---------|-----------|--------|--------|-------|
| Language | Mojo | C++ | C++ | C++ |
| GPU support | Single-source kernels, run on AMD only | Partial | Partial | Mature (CUDA-focused) |
| AMD GPU | Run on W7900 (RDNA3) | Limited | Limited | Limited |
| Python API | Native | Via wrappers | Via wrappers | Via wrappers |
| Maturity | Exploratory prototype | Production | Production | Production |

## Code Structure

### Basic Demo (`amr.mojo`)

```
├── AMRMesh struct          # Mesh data structure
├── initialize_uniform_mesh # Create base grid
├── set_gaussian_initial_condition # Physics IC
├── compute_gradients       # Refinement criterion
├── refine_cell            # Cell splitting
├── heat_diffusion_step    # Physics solver
└── main                   # Demo driver
```

### Enhanced Demo (`amr_enhanced.mojo`)

```
├── AMRMesh[mesh_type] struct           # Compile-time specialized mesh
├── heat_diffusion_structured           # Optimized for direct addressing
├── heat_diffusion_unstructured         # Optimized for CSR indirect access
├── compute_gradients_simd              # SIMD-hinted gradient computation
├── benchmark_diffusion_structured      # Performance measurement
├── benchmark_diffusion_unstructured    # Performance measurement
├── heat_diffusion_gpu (conceptual)     # GPU kernel implementation guide
└── main                                # Comparative benchmarking driver
```

**Key Differences**:

- Parametric types for compile-time specialization
- Separate code paths optimized by MLIR for each mesh structure
- Explicit performance measurement and comparison
- GPU-ready architecture demonstration

## Future Extensions

1. **3D Support**: Extend to octree refinement
2. **Coarsening**: Merge cells in smooth regions
3. **Load Balancing**: For multi-GPU
4. **I/O**: VTK/HDF5 output for visualization
5. **More Physics**:
   - Euler equations (compressible flow)
   - Magnetohydrodynamics (MHD)
   - Radiation transport

## References

1. Berger, M. J., & Colella, P. (1989). *Local adaptive mesh refinement for shock hydrodynamics*. Journal of Computational Physics, 82(1), 64-84.

2. MacNeice, P., et al. (2000). *PARAMESH: A parallel adaptive mesh refinement community toolkit*. Computer Physics Communications, 126(3), 330-354.

3. Zhang, W., et al. (2019). *AMReX: a framework for block-structured adaptive mesh refinement*. Journal of Open Source Software, 4(37), 1370.

4. Bell, J. B., et al. (1994). *A second-order projection method for the incompressible Navier-Stokes equations*. Journal of Computational Physics, 85(2), 257-283.

## Addressing the Indirect Memory Access Challenge

Traditional AMR implementations face a critical performance bottleneck: **"every floating point we fetch to do the math is an indirect lookup"** when memory is overcommitted. This creates doubly-indirect access patterns that confound most compilers:

```mojo
# Double indirection in unstructured meshes
neighbor_id = neighbor_ids[k]              # First indirection
temperature = temperatures[neighbor_id]     # Second indirection
```

### How Mojo Solves This

The `amr_enhanced.mojo` demo demonstrates Mojo's unique advantages:

#### 1. **Compile-Time Specialization**

```mojo
struct AMRMesh[mesh_type: MeshType]:  # Compile-time parameter
    ...

fn heat_diffusion_structured(mesh: AMRMesh[STRUCTURED], ...)   # Direct addressing
fn heat_diffusion_unstructured(mesh: AMRMesh[UNSTRUCTURED], ...) # Indirect via CSR
```

- Zero runtime overhead for type dispatch
- Specialized code paths for each mesh structure
- Compiler optimizes each variant independently

#### 2. **MLIR-Based Optimization (candidate, unverified here)**

Mojo's MLIR infrastructure *can* enable optimizations such as the following.
Whether they fire for this code has not been confirmed via emitted IR — see the
[Validation checklist](#validation-checklist):

- Pointer alias analysis
- Prefetch insertion for indirect accesses
- Loop optimization aware of memory access patterns
- Instruction scheduling around memory latency

#### 3. **SoA Memory Layout**

```mojo
# Coalesced access - adjacent cells in memory
temperatures: [T0, T1, T2, T3, ...]
neighbor_ids: [n0, n1, n2, n3, ...]
```

- GPU-friendly coalesced memory access
- Better cache line utilization on CPU
- SIMD-vectorizable access patterns
- Reduces one level of indirection vs Array-of-Structs

#### 4. **Performance Results: Mojo Compiler Optimizations**

The numbers below come from a single run of the AMR demo on a 64×64 mesh (4096
cells, small) with 100 timesteps, on one machine. They are **preliminary**:
there is no warmup, no repetition/median, no scaling curve across mesh sizes,
and kernel-only vs end-to-end timing is not separated. Read them as a rough
illustration of Mojo's explicit compiler-control features, not as benchmark
results. See the [Validation checklist](#validation-checklist) for what would be
needed to make them trustworthy.

![AMR Benchmark Results](amr_benchmark_results.png)

![GPU Performance Comparison](amr_gpu_performance.png)

![Complete Performance Overview](amr_complete_comparison.png)

**Structured Mesh Performance:**

| Optimization Level | Throughput (Mcells/sec) | Speedup |
|-------------------|-------------------------|---------|
| Baseline (auto-vectorization) | 5.02 | 1.0× |
| + Explicit SIMD | 69.42 | 13.8× |
| + Cache blocking | **724.47** | **144.2×** |

**Unstructured Mesh Performance (CSR indirect access):**

| Optimization Level | Throughput (Mcells/sec) | Speedup |
|-------------------|-------------------------|---------|
| Baseline (auto-vectorization) | 4.68 | 1.0× |
| + SIMD + @parameter unrolling | 5.12 | 1.09× |
| + prefetch + @always_inline | 5.14 | 1.10× |

**Comparison with C++ (GCC 15.2, -O3 -march=native):**

| Implementation | Structured | Unstructured |
|---------------|-----------|--------------|
| **Mojo (optimized)** | 724.47 Mcells/sec | 5.14 Mcells/sec |
| **C++ (optimized)** | 65.96 Mcells/sec | 100.87 Mcells/sec |
| **Mojo advantage** | **11.0× faster** | 19.6× slower |

![Speedup Comparison](amr_speedup_comparison.png)

**Observations (preliminary, single run on one machine — no warmup, no repetition, no scaling curve; see [Limitations](#limitations)):**

1. **Cache blocking showed a large speedup** (~144× here) for structured meshes, using Mojo's parametric `tile_size`. The magnitude is sensitive to mesh size and machine and should be re-measured with proper methodology before being quoted.
2. **Mojo was faster than this particular C++ build** (GCC 15.2 `-O3 -march=native`) for structured grids in this run. This is a comparison against one reference build for context, not a general "Mojo beats C++" claim.
3. **Unstructured meshes were slower than the C++ reference** here — indirect memory access patterns need further work and measurement.

**Compiler Features Demonstrated:**

- `@parameter` - Compile-time loop unrolling (zero runtime overhead)
- `SIMD[DType, width]` - Explicit vectorization via SIMD types (verify the lowering in emitted assembly)
- `@always_inline` - Forced inlining (not just a hint like C++ `inline`)
- `prefetch()` - Explicit memory prefetch hints from `sys.intrinsics`
- `alias` - Compile-time constant evaluation
- Parametric `tile_size` - Cache blocking optimized at compile-time
- `simdwidthof[]` - Architecture-adaptive SIMD width selection

**Why This Matters:**

Traditional C++ requires compiler-specific intrinsics or inline assembly to achieve similar low-level control. Mojo provides portable, high-level syntax with explicit compiler directives intended to work across CPUs and GPUs. In this prototype the GPU path was run only on AMD (W7900, RDNA3); the NVIDIA path is expected to work from the same source but has **not** been validated here. The measured throughput differences relative to the C++ reference are preliminary single-machine numbers (see [Limitations](#limitations)) and should not be read as a general performance claim.

#### 5. **GPU Acceleration Results**

The AMR demo includes GPU kernels written against Mojo's portable GPU API. They
were exercised on AMD only; portability to NVIDIA is a design goal, not a
validated result here.

**GPU results (single run on AMD W7900, RDNA3, 64×64 grid / 4096 cells, 100 timesteps; no warmup or repetition, see [Limitations](#limitations)):**

| Mesh Type | CPU Baseline | CPU Best | GPU | GPU vs baseline |
|-----------|-------------|----------|-----|-------------|
| **Structured** | 5.01 Mcells/sec | 900.73 Mcells/sec | 28.67 Mcells/sec | ~5.7× (preliminary) |
| **Unstructured (CSR)** | 4.49 Mcells/sec | 5.73 Mcells/sec | 26.66 Mcells/sec | ~5.9× (preliminary) |

**GPU observations (preliminary):**

1. **Single-source kernels** - the same Mojo source compiled for CPU and GPU on this machine, with no AMD-specific code changes
2. **Double-indirect CSR access** - the GPU run was faster than the CPU baseline on this small mesh despite irregular memory patterns; this needs scaling curves before it can be called a speedup
3. **Low duplication** - CPU and GPU kernels share the same algorithm logic
4. **Exploratory only** - this is a prototype for evaluation, not a production-ready HPC component

**GPU Kernel Features:**

```mojo
# GPU kernel with same algorithm as CPU version
@parameter
@__copy_capture(temperatures_ptr, temp_new_ptr, ...)
fn diffusion_kernel():
    var tid = Int(thread_idx.x + block_idx.x * block_dim.x)
    # Double indirection for CSR graph
    var start = Int(neighbor_offsets_ptr[tid])
    var end = Int(neighbor_offsets_ptr[tid + 1])
    for k in range(start, end):
        var nbr_id = Int(neighbor_ids_ptr[k])
        laplacian += temperatures_ptr[nbr_id] - temp_center
```

- **SoA layout**: Intended to enable coalesced memory access across GPU threads (not yet confirmed via profiler)
- **Single-source**: Same kernel source used for CPU and AMD GPU; NVIDIA not validated here
- **Parallelism**: GPU parallelism is expected to help hide latency from irregular access patterns

### Why This Matters for HPC

- **Memory overcommitment**: AMR typically has 10-100× more cells than fit in cache
- **Irregular access**: Physics-driven refinement creates unpredictable memory patterns
- **Explicit control**: Mojo exposes SIMD, tiling, and prefetch directives at the source level
- **Portability goal**: a single codebase targeting CPU and GPU (validated on CPU + AMD GPU; NVIDIA untested here)
- **Compiler leverage**: MLIR-based infrastructure that may apply these optimizations (verify via emitted IR/assembly, see below)

### Additional Mojo Advantages

- **Productivity**: High-level Python-like syntax with systems programming control
- **Integration**: Native Python interoperability for existing scientific workflows
- **Modern tooling**: Fast compilation times compared to traditional C++ HPC frameworks
- **Metaprogramming**: Zero-cost abstractions via compile-time parameters

In principle the approach could be extended toward larger simulations with
multi-physics, 3D, and multi-GPU support, but none of that is implemented or
validated here — those are directions for future work, not capabilities of this
prototype.

## Advanced: Compiler Optimizations Analysis

### Verifying Mojo Compiler Features

The AMR demo demonstrates advanced compiler optimizations. To verify these capabilities:

#### 1. Inspect MLIR IR (Intermediate Representation)

```bash
mojo build --emit-mlir amr.mojo > amr.mlir
```

Look for in the generated IR:

- `affine.for` loops - Affine analysis detected regular patterns
- `vector.gather` / `vector.scatter` - SIMD indirect access operations
- `llvm.prefetch` - Automatic prefetch insertion
- `memref.alloca` with optimized layouts

#### 2. Check Generated Assembly

```bash
mojo build --emit-assembly amr.mojo > amr.s
```

Look for:

- `vgatherdps` / `vgatherqps` - AVX2/AVX-512 gather instructions
- `prefetcht0` / `prefetchnta` - Hardware prefetch hints
- Vectorized loops using SIMD registers (`ymm0-15`, `zmm0-31`)

#### 3. Profile Hardware Performance

```bash
# Measure cache behavior
perf stat -e cache-misses,cache-references,L1-dcache-load-misses ./amr

# Measure memory bandwidth
perf stat -e mem_load_retired.fb_hit,mem_load_retired.l1_miss ./amr
```

### Compiler optimizations to look for (hypotheses to verify)

> **Important:** the items below are *candidate* optimizations that the source
> patterns make possible. They have **not** been confirmed for this code by
> inspecting emitted IR/assembly or hardware counters. Treat each as a
> hypothesis to check with the commands above, not as a statement of what the
> compiler did.

**1. Indirect access pattern (CSR gather)**

- Source: `temperatures[neighbor_ids[k]]` (double indirection)
- *Possible* result: `vector.gather` instead of scalar loads
- Verify: look for `vector.gather` in `--emit-mlir`, `vgather*` in assembly

**2. Affine access map analysis**

- Source: strided access `data[STRIDE * i]`
- *Possible* result: strength reduction (multiply → shift), vectorization
- Verify: inspect address calculation in emitted assembly

**3. Coalesced memory batching (GPU)**

- Source: `temperatures[neighbor_ids[tid]]` across a warp
- *Possible* result: fewer cache-line fetches when indices cluster
- Verify: GPU profiler memory-transaction counters; not measured here

**4. Software prefetch**

- Source: sequential/predictable index patterns
- *Possible* result: inserted `llvm.prefetch` instructions
- Verify: grep for `llvm.prefetch` in IR / `prefetcht0` in assembly

See **[COMPILER_FEATURES.md](COMPILER_FEATURES.md)** for a more detailed
walkthrough of these candidate transformations, written in the same
"verify before believing" spirit.

### Compiler Analysis Demo

```bash
# Run compiler optimization analysis
mojo compiler_analysis.mojo
```

This standalone demo illustrates (it does not, by itself, prove the compiler
applied) the following ideas:

- Indirect pattern detection
- Affine access map optimization
- CSR coalescing effectiveness
- Prefetch queue generation

To confirm any of these actually happen, inspect the emitted IR/assembly and
hardware counters as described above.

## Limitations

- **AI-generated prototype.** This code and documentation were produced with AI
  assistance and require human review before any claim here is relied upon.
- **Small benchmark sizes.** Measurements use a 64×64 mesh (4096 cells). This is
  too small to be representative; numbers may change substantially at larger
  sizes and must be re-measured.
- **Speedups are preliminary.** Reported speedups come from single runs without
  warmup, repetition, medians, or scaling curves, and may not reproduce.
- **C++ reference is for comparison only.** `amr_benchmark.cpp` exists to give a
  rough point of reference on the same machine. It is not a tuned or
  expert-written HPC implementation, and "Mojo vs C++" figures should not be
  read as a general language comparison.
- **GPU results are single-target.** GPU numbers were obtained only on an AMD
  W7900 (RDNA3). Results must be reported per hardware target; do not merge AMD
  and (untested) NVIDIA results.
- **Compiler claims are unverified.** Statements about gather generation,
  prefetch insertion, strength reduction, and coalescing are hypotheses based on
  source patterns. They need IR/assembly/profiler evidence before being stated
  as fact.

## Validation checklist

Before any performance or capability claim in this document is treated as real,
the following should be done and recorded:

- [ ] **Numerical parity** — verify the Mojo and reference (C++) implementations
      produce matching results within a documented tolerance on identical inputs.
- [ ] **Repeated runs with medians** — run each benchmark many times with warmup,
      report median and spread rather than a single sample.
- [ ] **Larger meshes and scaling curves** — measure across a range of mesh sizes
      and plot throughput vs problem size, not just one small grid.
- [ ] **Separate timings** — report kernel-only and end-to-end timings
      separately so setup/transfer costs are not hidden.
- [ ] **Profiler / hardware-counter evidence** — back any cache, coalescing, or
      prefetch claim with counter data (e.g. `perf stat`, GPU profiler), and back
      any compiler-transformation claim with emitted IR/assembly.
- [ ] **AMD and NVIDIA runs** — if portability is claimed, actually run on both
      AMD and NVIDIA hardware and report each separately.

## Learn More

- **Modular Platform**: <https://www.modular.com>
- **Mojo Documentation**: <https://docs.modular.com/mojo>
- **MAX Engine**: <https://docs.modular.com/max>
- **MLIR Documentation**: <https://mlir.llvm.org/>
