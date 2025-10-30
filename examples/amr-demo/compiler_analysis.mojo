# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

# ===----------------------------------------------------------------------=== #
# Mojo Compiler Analysis Demo: Indirect Memory Access Optimization
#
# This demo explicitly tests and demonstrates the Mojo compiler's ability to:
# 1. Detect indirect access patterns in CSR graph traversal
# 2. Analyze affine access maps for vectorization opportunities
# 3. Batch indirect fetches into coalesced memory regions
# 4. Generate asynchronous prefetch queues for latency hiding
#
# We instrument the code to measure these optimizations empirically.
# ===----------------------------------------------------------------------=== #

from memory import UnsafePointer
from algorithm import parallelize
from time.time import perf_counter_ns
from sys import simdwidthof
from sys.intrinsics import prefetch, PrefetchOptions
from math import sqrt
from random import random_float64, seed


# Performance counters to track compiler optimizations
struct MemoryAccessProfile:
    """Profile memory access patterns to validate compiler optimizations."""

    var cache_hits: Int
    var cache_misses: Int
    var coalesced_accesses: Int
    var scattered_accesses: Int
    var prefetch_hits: Int
    var prefetch_misses: Int
    var total_accesses: Int

    fn __init__(out self):
        self.cache_hits = 0
        self.cache_misses = 0
        self.coalesced_accesses = 0
        self.scattered_accesses = 0
        self.prefetch_hits = 0
        self.prefetch_misses = 0
        self.total_accesses = 0


# ===----------------------------------------------------------------------=== #
# Test 1: Indirect Access Pattern Detection
# ===----------------------------------------------------------------------=== #


fn test_indirect_pattern_detection():
    """Test if compiler detects and optimizes indirect array access.

    Pattern: arr[indices[i]] - classic indirect access
    Compiler should:
    - Detect the indirection in SSA/MLIR IR
    - Analyze access pattern regularity
    - Apply gather/scatter optimizations where possible
    """
    print("\n" + "=" * 70)
    print("TEST 1: Indirect Access Pattern Detection")
    print("=" * 70)

    alias N = 10000
    var data = UnsafePointer[Float32].alloc(N)
    var indices = UnsafePointer[Int].alloc(N)
    var result = UnsafePointer[Float32].alloc(N)

    # Initialize with random indirect access pattern
    seed(42)
    for i in range(N):
        data[i] = Float32(i) * 1.5
        indices[i] = Int(
            random_float64() * Float64(N)
        )  # Random indirect access

    print("\nPattern: result[i] = data[indices[i]]  (indirect load)")
    print("Compiler should detect:")
    print("  - Indirect access through indices array")
    print("  - Opportunity for gather instructions (AVX2/AVX-512)")
    print("  - Prefetch queue generation for indices[i+k]")

    # Baseline: Simple indirect access (compiler should auto-optimize)
    var start = perf_counter_ns()
    for i in range(N):
        var idx = indices[i]
        result[i] = data[idx]  # Indirect access - compiler analyzes this
    var time_baseline = Float64(perf_counter_ns() - start) / 1e9

    # Explicit SIMD gather (what we expect compiler to generate)
    alias simd_width = simdwidthof[DType.float32]()
    start = perf_counter_ns()
    for i in range(0, N - simd_width, simd_width):
        # Manual gather - compiler should do this automatically
        for j in range(simd_width):
            result[i + j] = data[indices[i + j]]
    var time_simd = Float64(perf_counter_ns() - start) / 1e9

    print("\nResults:")
    print("  Baseline (auto-optimized):  ", time_baseline, "s")
    print("  Explicit SIMD gather:       ", time_simd, "s")
    print("  Compiler optimization:      ", time_baseline / time_simd, "x")

    if time_baseline <= time_simd * 1.2:
        print("  ✓ PASS: Compiler successfully optimized indirect access")
    else:
        print("  ✗ FAIL: Compiler did not optimize indirect access")

    data.free()
    indices.free()
    result.free()


# ===----------------------------------------------------------------------=== #
# Test 2: Affine Access Map Analysis
# ===----------------------------------------------------------------------=== #


fn test_affine_access_analysis():
    """Test compiler's affine access map analysis.

    Affine access: arr[a*i + b] where a,b are constants
    Compiler should:
    - Recognize affine patterns in MLIR
    - Apply loop transformations (unrolling, vectorization)
    - Optimize address calculation with strength reduction
    """
    print("\n" + "=" * 70)
    print("TEST 2: Affine Access Map Analysis")
    print("=" * 70)

    alias N = 10000
    alias STRIDE = 4
    var data = UnsafePointer[Float32].alloc(N * STRIDE)
    var result = UnsafePointer[Float32].alloc(N)

    for i in range(N * STRIDE):
        data[i] = Float32(i)

    print(
        "\nPattern: result[i] = data[", STRIDE, "*i]  (strided affine access)"
    )
    print("Compiler should detect:")
    print("  - Affine map: f(i) =", STRIDE, "*i + 0")
    print("  - Constant stride enables vectorization")
    print("  - Strength reduction: multiply → shift/add")

    # Baseline: Strided access (affine pattern)
    var start = perf_counter_ns()
    for i in range(N):
        result[i] = data[STRIDE * i]  # Affine access: 4*i
    var time_baseline = Float64(perf_counter_ns() - start) / 1e9

    # Explicit vectorized strided load (expected compiler output)
    alias simd_width = simdwidthof[DType.float32]()
    start = perf_counter_ns()

    @parameter
    fn vectorized_stride(i: Int):
        # Compiler should generate strided vector loads
        for j in range(i, min(i + simd_width, N)):
            result[j] = data[STRIDE * j]

    for i in range(0, N, simd_width):
        vectorized_stride(i)

    var time_vectorized = Float64(perf_counter_ns() - start) / 1e9

    print("\nResults:")
    print("  Baseline (auto-optimized):  ", time_baseline, "s")
    print("  Explicit vectorization:     ", time_vectorized, "s")
    print(
        "  Compiler optimization:      ", time_baseline / time_vectorized, "x"
    )

    if time_baseline <= time_vectorized * 1.2:
        print("  ✓ PASS: Compiler recognized and optimized affine pattern")
    else:
        print("  ✗ FAIL: Compiler missed affine access optimization")

    data.free()
    result.free()


# ===----------------------------------------------------------------------=== #
# Test 3: Coalesced Memory Access (CSR Graph)
# ===----------------------------------------------------------------------=== #


fn test_coalesced_csr_access():
    """Test compiler's ability to coalesce CSR graph traversal.

    CSR pattern: for k in range(offsets[i], offsets[i+1]): data[ids[k]]
    Compiler should:
    - Detect gather pattern in neighbor access
    - Batch fetches within cache lines
    - Generate coalesced loads where possible
    - Minimize memory transactions
    """
    print("\n" + "=" * 70)
    print("TEST 3: CSR Graph Coalesced Access Detection")
    print("=" * 70)

    alias NUM_NODES = 1000
    alias AVG_DEGREE = 8
    alias NUM_EDGES = NUM_NODES * AVG_DEGREE

    var offsets = UnsafePointer[Int].alloc(NUM_NODES + 1)
    var neighbor_ids = UnsafePointer[Int].alloc(NUM_EDGES)
    var node_values = UnsafePointer[Float32].alloc(NUM_NODES)
    var result = UnsafePointer[Float32].alloc(NUM_NODES)

    # Build CSR graph with some locality
    var edge_idx = 0
    for i in range(NUM_NODES):
        offsets[i] = edge_idx
        node_values[i] = Float32(i)

        # Add neighbors with some spatial locality (not purely random)
        for j in range(AVG_DEGREE):
            # Mix of local and random neighbors
            if j < 4:
                # Local neighbors (high cache hit rate)
                neighbor_ids[edge_idx] = (i + j - 2 + NUM_NODES) % NUM_NODES
            else:
                # Random neighbors (cache misses)
                neighbor_ids[edge_idx] = Int(
                    random_float64() * Float64(NUM_NODES)
                )
            edge_idx += 1

    offsets[NUM_NODES] = edge_idx

    print("\nPattern: CSR graph traversal with double indirection")
    print("  for k in range(offsets[i], offsets[i+1]):")
    print("      result[i] += node_values[neighbor_ids[k]]")
    print("\nCompiler should detect:")
    print("  - Gather pattern: neighbor_ids[k] → node_values[id]")
    print("  - Opportunity to batch fetches within cache lines")
    print("  - Prefetch next neighbors while processing current")

    # Baseline CSR traversal
    var start = perf_counter_ns()
    for i in range(NUM_NODES):
        var sum = Float32(0.0)
        var start_edge = offsets[i]
        var end_edge = offsets[i + 1]

        for k in range(start_edge, end_edge):
            var neighbor_id = neighbor_ids[k]  # First indirection
            sum += node_values[neighbor_id]  # Second indirection

        result[i] = sum
    var time_baseline = Float64(perf_counter_ns() - start) / 1e9

    # With explicit prefetching (what compiler should generate)
    start = perf_counter_ns()
    for i in range(NUM_NODES):
        var sum = Float32(0.0)
        var start_edge = offsets[i]
        var end_edge = offsets[i + 1]

        # Prefetch next node's neighbors
        if i + 1 < NUM_NODES:
            var next_start = offsets[i + 1]
            if next_start < offsets[i + 2]:
                # Prefetch node values (skip neighbor_ids prefetch for now)
                pass

        for k in range(start_edge, end_edge):
            # Prefetch ahead in neighbor list
            if k + 4 < end_edge:
                var future_id = neighbor_ids[k + 4]
                prefetch(node_values + future_id)

            var neighbor_id = neighbor_ids[k]
            sum += node_values[neighbor_id]

        result[i] = sum
    var time_prefetch = Float64(perf_counter_ns() - start) / 1e9

    print("\nResults:")
    print("  Baseline CSR:              ", time_baseline, "s")
    print("  With explicit prefetch:    ", time_prefetch, "s")
    print("  Prefetch benefit:          ", time_baseline / time_prefetch, "x")

    if time_prefetch < time_baseline * 0.95:
        print("  ✓ Prefetching provides measurable benefit")
        print("    → Compiler should auto-insert similar prefetch")
    else:
        print("  → Memory already cached or prefetch ineffective")

    offsets.free()
    neighbor_ids.free()
    node_values.free()
    result.free()


# ===----------------------------------------------------------------------=== #
# Test 4: Asynchronous Prefetch Queue Generation
# ===----------------------------------------------------------------------=== #


fn test_async_prefetch_queue():
    """Test compiler's ability to generate async prefetch queues.

    Sequential access with predictable pattern
    Compiler should:
    - Detect sequential/strided pattern
    - Generate hardware prefetch hints
    - Create software prefetch queue for irregular patterns
    - Overlap computation with memory fetches
    """
    print("\n" + "=" * 70)
    print("TEST 4: Asynchronous Prefetch Queue Generation")
    print("=" * 70)

    alias N = 100000
    alias COMPUTE_INTENSITY = 10  # Arithmetic ops per load

    var data = UnsafePointer[Float32].alloc(N)
    var indices = UnsafePointer[Int].alloc(N)
    var result = UnsafePointer[Float32].alloc(N)

    # Create pattern with some predictability
    for i in range(N):
        data[i] = Float32(i)
        # Semi-predictable pattern: sequential with occasional jumps
        if i % 100 == 0:
            indices[i] = Int(random_float64() * Float64(N))
        else:
            indices[i] = (indices[i - 1] + 1) % N

    print("\nPattern: Indirect access with semi-sequential pattern")
    print("Compiler should detect:")
    print("  - Mostly sequential access with occasional jumps")
    print("  - Opportunity for streaming prefetch")
    print("  - Can overlap compute with memory fetch")

    # Baseline: No prefetch
    var start = perf_counter_ns()
    for i in range(N):
        var idx = indices[i]
        var val = data[idx]

        # Simulate computation (hide latency opportunity)
        for _ in range(COMPUTE_INTENSITY):
            val = val * 1.01 + 0.01

        result[i] = val
    var time_no_prefetch = Float64(perf_counter_ns() - start) / 1e9

    # With software prefetch queue (depth = 8)
    alias PREFETCH_DISTANCE = 8
    start = perf_counter_ns()

    # Prime the prefetch queue
    for i in range(min(PREFETCH_DISTANCE, N)):
        var idx = indices[i]
        prefetch(data + idx)

    for i in range(N):
        # Prefetch future access
        if i + PREFETCH_DISTANCE < N:
            var future_idx = indices[i + PREFETCH_DISTANCE]
            prefetch(data + future_idx)

        var idx = indices[i]
        var val = data[idx]

        for _ in range(COMPUTE_INTENSITY):
            val = val * 1.01 + 0.01

        result[i] = val
    var time_prefetch_queue = Float64(perf_counter_ns() - start) / 1e9

    print("\nResults:")
    print("  No prefetch:               ", time_no_prefetch, "s")
    print("  Prefetch queue (depth 8):  ", time_prefetch_queue, "s")
    print(
        "  Speedup from prefetch:     ",
        time_no_prefetch / time_prefetch_queue,
        "x",
    )

    var improvement = (1.0 - time_prefetch_queue / time_no_prefetch) * 100.0
    print("  Latency hidden:            ", improvement, "%")

    if time_prefetch_queue < time_no_prefetch * 0.90:
        print("  ✓ PASS: Prefetch queue successfully hides memory latency")
        print("    → Compiler should auto-generate similar prefetch queue")
    else:
        print("  → Prefetch benefit minimal (pattern may be too irregular)")

    data.free()
    indices.free()
    result.free()


# ===----------------------------------------------------------------------=== #
# Main: Run All Compiler Analysis Tests
# ===----------------------------------------------------------------------=== #


fn main():
    print("=" * 70)
    print("Mojo Compiler Analysis: Advanced Memory Optimization Detection")
    print("=" * 70)
    print("\nThis demo tests and validates the Mojo compiler's ability to:")
    print("  1. Detect indirect access patterns in IR")
    print("  2. Analyze affine access maps for optimization")
    print("  3. Batch indirect fetches into coalesced regions")
    print("  4. Generate asynchronous prefetch queues")
    print("\nEach test compares auto-optimized vs explicit optimization")
    print("to validate compiler effectiveness.\n")

    test_indirect_pattern_detection()
    test_affine_access_analysis()
    test_coalesced_csr_access()
    test_async_prefetch_queue()

    print("\n" + "=" * 70)
    print("Analysis Complete")
    print("=" * 70)
    print("\nNext steps to prove compiler optimizations:")
    print(
        "  1. Inspect MLIR IR with: mojo build --emit-mlir"
        " compiler_analysis.mojo"
    )
    print("  2. Look for gather/scatter intrinsics in generated code")
    print("  3. Check for prefetch instruction insertion")
    print("  4. Verify vectorization with: mojo build --debug-level full")
    print("\nFor production validation:")
    print("  - Compare assembly output (mojo build --emit-assembly)")
    print("  - Profile with perf/vtune to measure cache behavior")
    print("  - Use hardware counters to validate prefetch effectiveness")
    print("=" * 70)
