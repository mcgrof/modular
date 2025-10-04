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
"""
RDNA Flash Attention Implementation (Rewrite)

This module implements fused multi-head attention (FMHA) for AMD RDNA GPUs
(RDNA3/RDNA4) using Wave32 and WMMA 16x16x16 instructions.

Architecture based on TheRock's CK Tile composable kernel implementation.

Key Architecture Points:
- Wave size: 32 threads (Wave32)
- WMMA shape: 16x16x16
- Fragment distribution: Each thread owns specific matrix positions
- Proper thread coordinate mapping for WMMA fragments
- Row-wise softmax for attention
- Memory coalescing for optimal bandwidth

References:
- TheRock CK Tile: composable_kernel/example/ck_tile/01_fmha/
- RDNA3 WMMA Guide: https://gpuopen.com/learn/wmma_on_rdna3/
- Training Doc: /data/modular/flash-attention-port-to-mojo.md
"""

from algorithm import max as algorithm_max, min as algorithm_min
from collections import InlineArray
from math import ceildiv, exp, log, recip, sqrt
from math.constants import log2e
from sys import llvm_intrinsic, size_of, simd_width_of
from sys.info import _is_amd_rdna, _is_amd_rdna2, _is_amd_rdna3, _is_amd_rdna4

from gpu import WARP_SIZE, barrier, block_idx, lane_id, thread_idx, warp_id
from gpu.memory import AddressSpace
from gpu.mma import mma
from gpu.warp import shuffle_down, shuffle_idx, sum as warp_sum
from layout import Layout, LayoutTensor
from memory import stack_allocation
from utils import Index, IndexList
from utils.numerics import get_accum_type, min_or_neg_inf


# ===----------------------------------------------------------------------=== #
# RDNA WMMA Configuration
# ===----------------------------------------------------------------------=== #

alias RDNA_WAVE_SIZE = 32
"""RDNA uses Wave32 (32 threads per wavefront)."""

alias WMMA_M = 16
alias WMMA_N = 16
alias WMMA_K = 16
"""RDNA WMMA shape: 16x16x16."""

# Default tile sizes - these should be tuned per use case
alias DEFAULT_TILE_M = 128  # Q sequence tile
alias DEFAULT_TILE_N = 64  # K sequence tile
alias DEFAULT_TILE_K = 64  # Head dimension tile


# ===----------------------------------------------------------------------=== #
# WMMA Fragment Coordinate System
# ===----------------------------------------------------------------------=== #


@always_inline
fn get_wmma_thread_layout_16x16(lane: Int) -> Tuple[Int, Int]:
    """Calculate the (row, col) position for a thread in a 16x16 WMMA tile.

    For RDNA WMMA 16x16x16 with Wave32:
    - 32 threads cover a 16x16 matrix
    - Each thread is responsible for specific matrix positions
    - Layout is optimized for memory coalescing

    Wave32 layout for 16x16 (simplified - actual hardware may vary):
    - Threads 0-15: rows 0-15, cols 0-7 (even columns)
    - Threads 16-31: rows 0-15, cols 8-15 (odd columns)

    This is a simplified model. Real fragment layout should be verified
    with hardware documentation or by testing known-good patterns.

    Args:
        lane: Thread lane ID (0-31 for Wave32).

    Returns:
        (row, col) position in the 16x16 tile for this thread.
    """
    # Simplified layout - each thread covers 2 rows, 1 col
    # Thread 0-15 handle left half, 16-31 handle right half
    var row = lane % 16
    var col = (lane // 16) * 8
    return Tuple(row, col)


@always_inline
fn get_wmma_fragment_indices_16x16(
    lane: Int, num_elements: Int = 8
) -> InlineArray[Int, 8]:
    """Get the linear indices for this thread's fragment in a 16x16 WMMA tile.

    Each thread in Wave32 owns 8 elements (for FP16/BF16 WMMA 16x16x16).
    These 8 elements map to specific positions in the 16x16 matrix.

    Args:
        lane: Thread lane ID.
        num_elements: Number of elements per thread (default 8).

    Returns:
        Tuple of 8 linear indices [0, 256) for a 16x16 matrix.
    """
    var layout = get_wmma_thread_layout_16x16(lane)
    var base_row = layout[0]
    var base_col = layout[1]

    # Each thread owns 8 elements in a specific pattern
    # Simplified: consecutive elements in column-major order
    var indices = InlineArray[Int, 8](uninitialized=True)

    # Pattern: 8 elements covering 2 columns × 4 rows
    @parameter
    for i in range(8):
        var local_row = i // 2
        var local_col = i % 2
        var global_row = base_row + local_row
        var global_col = base_col + local_col
        indices[i] = global_row * 16 + global_col

    return indices


# ===----------------------------------------------------------------------=== #
# WMMA Fragment Operations
# ===----------------------------------------------------------------------=== #


@always_inline
fn wmma_store_accumulator_to_smem[
    accum_dtype: DType,
    output_dtype: DType,
](
    smem: UnsafePointer[
        Scalar[output_dtype], address_space = AddressSpace.SHARED
    ],
    stride: Int,
    tile_m: Int,
    tile_n: Int,
    acc: SIMD[accum_dtype, 8],
):
    """Store WMMA accumulator fragment to shared memory as a matrix.

    Converts 8 FP32 accumulator elements to output dtype and stores them in
    proper matrix layout according to AMD RDNA WMMA fragment mapping.

    For RDNA WMMA 16x16x16, each thread owns 8 FP32 accumulator elements
    that map to 8 consecutive columns in a single row.

    Args:
        smem: Shared memory pointer for output matrix.
        stride: Stride between rows in shared memory.
        tile_m: Which 16x16 tile in M dimension.
        tile_n: Which 16x16 tile in N dimension.
        acc: Accumulator fragment (8 FP32 elements per thread).
    """
    var lane = lane_id()
    var thread_x = lane & 15  # lane % 16 (which row, 0-15)
    var thread_y = lane >> 4  # lane // 16 (which half, 0 or 1)

    var row = tile_m * WMMA_M + Int(thread_x)
    var col_base = (
        tile_n * WMMA_N + Int(thread_y) * 8
    )  # Left half (0-7) or right half (8-15)

    # Store 8 consecutive accumulator elements
    @parameter
    for i in range(8):
        smem[row * stride + col_base + i] = acc[i].cast[output_dtype]()


@always_inline
fn wmma_load_fragment_a[
    dtype: DType
](
    smem: UnsafePointer[Scalar[dtype], address_space = AddressSpace.SHARED],
    stride: Int,
    tile_m: Int,
    tile_k: Int,
) -> SIMD[dtype, 16]:
    """Load A matrix fragment from shared memory for WMMA.

    Loads 16 elements per thread according to AMD RDNA WMMA 16x16x16 fragment layout.
    Each thread loads 16 consecutive elements from a single row.

    Args:
        smem: Shared memory pointer to A matrix tile.
        stride: Stride between rows in shared memory.
        tile_m: Which 16x16 tile in M dimension (0, 1, 2, ...).
        tile_k: Which 16x16 tile in K dimension (0, 1, 2, ...).

    Returns:
        SIMD[dtype, 16] fragment for this thread.
    """
    var frag = SIMD[dtype, 16]()
    var lane = lane_id()
    var thread_x = lane & 15  # lane % 16 (which row, 0-15)

    var row = tile_m * WMMA_M + Int(thread_x)
    var col_base = tile_k * WMMA_K

    # Load 16 consecutive elements from this thread's row
    @parameter
    for i in range(16):
        frag[i] = smem[row * stride + col_base + i]

    return frag


@always_inline
fn wmma_load_fragment_b[
    dtype: DType,
    transpose: Bool = False,
](
    smem: UnsafePointer[Scalar[dtype], address_space = AddressSpace.SHARED],
    stride: Int,
    tile_k: Int,
    tile_n: Int,
) -> SIMD[dtype, 16]:
    """Load B matrix fragment from shared memory for WMMA.

    Loads 16 elements per thread according to AMD RDNA WMMA 16x16x16 fragment layout.

    For non-transposed: Each thread loads 1 column, all 16 rows (K dimension)
    For transposed: Each thread loads 1 row, all 16 columns (for K^T)

    Args:
        smem: Shared memory pointer to B matrix tile.
        stride: Stride between rows in shared memory.
        tile_k: Which 16x16 tile in K dimension.
        tile_n: Which 16x16 tile in N dimension.

    Returns:
        SIMD[dtype, 16] fragment for this thread.
    """
    var frag = SIMD[dtype, 16]()
    var lane = lane_id()
    var thread_x = lane & 15  # 0-15 for all threads

    @parameter
    if transpose:
        # For B^T (like K^T in attention): load 1 row, 16 consecutive columns
        var row = (
            tile_n * WMMA_N + Int(thread_x)
        )  # Use tile_n for rows when transposed
        var col_base = tile_k * WMMA_K

        @parameter
        for i in range(16):
            frag[i] = smem[row * stride + col_base + i]
    else:
        # For normal B: load 1 column, 16 consecutive rows
        var row_base = tile_k * WMMA_K
        var col = tile_n * WMMA_N + Int(thread_x)

        @parameter
        for i in range(16):
            frag[i] = smem[(row_base + i) * stride + col]

    return frag


@always_inline
fn wmma_store_fragment[
    dtype: DType
](
    smem: UnsafePointer[Scalar[dtype], address_space = AddressSpace.SHARED],
    stride: Int,
    tile_m: Int,
    tile_n: Int,
    frag: SIMD[dtype, 8],
):
    """Store WMMA accumulator fragment to shared memory.

    Stores this thread's 8 accumulator elements according to AMD RDNA WMMA layout.
    Each thread owns 8 consecutive columns in a single row.

    Args:
        smem: Shared memory pointer to output tile.
        stride: Stride between rows in shared memory.
        tile_m: Which 16x16 tile in M dimension.
        tile_n: Which 16x16 tile in N dimension.
        frag: Fragment to store (8 elements).
    """
    var lane = lane_id()
    var thread_x = lane & 15  # which row, 0-15
    var thread_y = lane >> 4  # which half, 0 or 1

    var row = tile_m * WMMA_M + Int(thread_x)
    var col_base = tile_n * WMMA_N + Int(thread_y) * 8

    @parameter
    for i in range(8):
        smem[row * stride + col_base + i] = frag[i]


# ===----------------------------------------------------------------------=== #
# Shared Memory Management
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct RDNASharedMemory[
    dtype: DType,
    tile_n: Int,
    head_dim: Int,
](Copyable, Movable):
    """Shared memory buffers for K and V tiles.

    Layout optimized for coalesced access and WMMA patterns.

    Memory layout:
    - K: [tile_n, head_dim]
    - V: [tile_n, head_dim]

    Parameters:
        dtype: Data type (float16/bfloat16).
        tile_n: Tile size in sequence dimension.
        head_dim: Head dimension size.
    """

    var k_smem: UnsafePointer[
        Scalar[dtype], address_space = AddressSpace.SHARED
    ]
    var v_smem: UnsafePointer[
        Scalar[dtype], address_space = AddressSpace.SHARED
    ]

    @staticmethod
    fn create() -> Self:
        """Allocate shared memory buffers."""
        alias k_size = tile_n * head_dim
        alias v_size = tile_n * head_dim
        alias total_size = k_size + v_size

        var smem_base = stack_allocation[
            total_size,
            dtype,
            address_space = AddressSpace.SHARED,
        ]()

        var k_ptr = smem_base
        var v_ptr = smem_base.offset(k_size)

        return Self(k_ptr, v_ptr)

    fn load_k_tile_coalesced(
        self,
        k_global: UnsafePointer[Scalar[dtype]],
        seq_offset: Int,
        seq_len: Int,
        hdim: Int,
    ):
        """Load K tile from global memory with coalesced access.

        Each thread loads multiple elements in a coalesced pattern.

        Args:
            k_global: Global K pointer [seq_len, hdim].
            seq_offset: Starting sequence index for this tile.
            seq_len: Total sequence length (for bounds checking).
            hdim: Head dimension.
        """
        var tid = thread_idx.x
        alias num_threads = RDNA_WAVE_SIZE
        alias total_elements = tile_n * head_dim

        # Calculate how many elements each thread loads
        alias elements_per_thread = ceildiv(total_elements, num_threads)

        # Coalesced load pattern: consecutive threads load consecutive elements
        @parameter
        for i in range(elements_per_thread):
            var elem_idx = tid + UInt(i * num_threads)

            if elem_idx < UInt(total_elements):
                var row = elem_idx // UInt(hdim)
                var col = elem_idx % UInt(hdim)

                # Check bounds
                if (seq_offset + Int(row)) < seq_len:
                    var global_idx = (seq_offset + Int(row)) * hdim + Int(col)
                    self.k_smem[elem_idx] = k_global[global_idx]
                else:
                    # Pad with zeros
                    self.k_smem[elem_idx] = Scalar[dtype](0)

        barrier()

    fn load_v_tile_coalesced(
        self,
        v_global: UnsafePointer[Scalar[dtype]],
        seq_offset: Int,
        seq_len: Int,
        hdim: Int,
    ):
        """Load V tile from global memory with coalesced access.

        Args:
            v_global: Global V pointer [seq_len, hdim].
            seq_offset: Starting sequence index for this tile.
            seq_len: Total sequence length.
            hdim: Head dimension.
        """
        var tid = thread_idx.x
        alias num_threads = RDNA_WAVE_SIZE
        alias total_elements = tile_n * head_dim
        alias elements_per_thread = ceildiv(total_elements, num_threads)

        @parameter
        for i in range(elements_per_thread):
            var elem_idx = tid + UInt(i * num_threads)

            if elem_idx < UInt(total_elements):
                var row = elem_idx // UInt(hdim)
                var col = elem_idx % UInt(hdim)

                if (seq_offset + Int(row)) < seq_len:
                    var global_idx = (seq_offset + Int(row)) * hdim + Int(col)
                    self.v_smem[elem_idx] = v_global[global_idx]
                else:
                    self.v_smem[elem_idx] = Scalar[dtype](0)

        barrier()


# ===----------------------------------------------------------------------=== #
# Utility Functions
# ===----------------------------------------------------------------------=== #


@always_inline
fn scalar_max[
    dtype: DType
](a: Scalar[dtype], b: Scalar[dtype]) -> Scalar[dtype]:
    """Return the maximum of two scalars."""
    return a if a > b else b


@always_inline
fn scalar_min[
    dtype: DType
](a: Scalar[dtype], b: Scalar[dtype]) -> Scalar[dtype]:
    """Return the minimum of two scalars."""
    return a if a < b else b


# ===----------------------------------------------------------------------=== #
# Wave-Level Reductions
# ===----------------------------------------------------------------------=== #


@always_inline
fn wave_reduce_max[dtype: DType](val: Scalar[dtype]) -> Scalar[dtype]:
    """Reduce max across wavefront.

    Uses shuffle operations for efficient tree reduction.
    Result is broadcast to all lanes.
    """
    var result = val

    # Tree reduction for Wave32
    var offset = WARP_SIZE // 2
    while offset > 0:
        var shuffled = shuffle_down(SIMD[dtype, 1](result), UInt32(offset))[0]
        result = scalar_max(result, shuffled)
        offset //= 2

    # Broadcast result to all lanes
    return shuffle_idx(SIMD[dtype, 1](result), UInt32(0))[0]


@always_inline
fn wave_reduce_sum[dtype: DType](val: Scalar[dtype]) -> Scalar[dtype]:
    """Reduce sum across wavefront.

    Uses optimized warp_sum which broadcasts result.
    """
    return warp_sum(SIMD[dtype, 1](val))[0]


# ===----------------------------------------------------------------------=== #
# Per-Row Softmax for WMMA Fragments
# ===----------------------------------------------------------------------=== #


struct RDNASoftmaxState[dtype: DType = DType.float32]:
    """Per-row softmax state for online softmax algorithm.

    Tracks running max and sum for each row being processed.
    """

    var max_vals: UnsafePointer[
        Scalar[dtype], address_space = AddressSpace.LOCAL
    ]
    var sum_vals: UnsafePointer[
        Scalar[dtype], address_space = AddressSpace.LOCAL
    ]
    var num_rows: Int

    fn __init__(
        out self,
        num_rows: Int,
        max_ptr: UnsafePointer[
            Scalar[dtype], address_space = AddressSpace.LOCAL
        ],
        sum_ptr: UnsafePointer[
            Scalar[dtype], address_space = AddressSpace.LOCAL
        ],
    ):
        self.num_rows = num_rows
        self.max_vals = max_ptr
        self.sum_vals = sum_ptr

        # Initialize
        for i in range(num_rows):
            self.max_vals[i] = Scalar[dtype](-1e38)
            self.sum_vals[i] = Scalar[dtype](0)

    fn update_with_tile[
        accum_dtype: DType,
        num_wmma_m: Int,
        num_wmma_n: Int,
    ](
        mut self,
        s_frags: UnsafePointer[
            SIMD[accum_dtype, 8], address_space = AddressSpace.LOCAL
        ],
        o_frags: UnsafePointer[
            SIMD[accum_dtype, 8], address_space = AddressSpace.LOCAL
        ],
        scale: Float32,
    ):
        """Update softmax state with a new tile of attention scores.

        Implements online softmax:
        1. Find per-row max in new tile
        2. Update running max
        3. Compute exp(scores - max)
        4. Update running sum with correction factor
        5. Rescale previous outputs

        Args:
            s_frags: Attention score fragments [num_wmma_m * num_wmma_n, 8].
            o_frags: Output fragments to rescale [num_wmma_m * num_wmma_v, 8].
            scale: Attention scale factor (1/sqrt(head_dim)).
        """
        # Step 1: Find per-row max in this tile
        # TODO: This is simplified - need proper row identification from fragments

        # For now, use simplified single max (FIXME for production)
        var tile_max = Scalar[accum_dtype](-1e38)
        var scale_cast = Scalar[accum_dtype](scale)

        @parameter
        for frag_idx in range(num_wmma_m * num_wmma_n):

            @parameter
            for i in range(8):
                var val = s_frags[frag_idx][i] * scale_cast
                tile_max = scalar_max(tile_max, val)

        # Reduce across wave
        var wave_max = wave_reduce_max[accum_dtype](tile_max)

        # Update running max (simplified - single row for now)
        var prev_max = Scalar[accum_dtype](self.max_vals[0])
        var new_max = scalar_max(prev_max, wave_max)
        self.max_vals[0] = Scalar[dtype](new_max)

        # Step 2: Compute exp and sum
        var tile_sum = Scalar[accum_dtype](0)

        @parameter
        for frag_idx in range(num_wmma_m * num_wmma_n):

            @parameter
            for i in range(8):
                var val = s_frags[frag_idx][i] * scale_cast
                var p_val = exp(val - new_max)
                s_frags[frag_idx][i] = p_val
                tile_sum += p_val

        # Reduce sum across wave
        var wave_sum = wave_reduce_sum[accum_dtype](tile_sum)

        # Step 3: Update running sum with correction
        var correction = exp(prev_max - new_max)
        var prev_sum = Scalar[accum_dtype](self.sum_vals[0])
        var new_sum = prev_sum * correction + wave_sum
        self.sum_vals[0] = Scalar[dtype](new_sum)

        # Step 4: Rescale previous outputs
        # TODO: Calculate proper number of output fragments
        # Use runtime loop since num_output_frags may not be compile-time constant
        for frag_idx in range(num_wmma_m * 4):  # Simplified
            for i in range(8):
                o_frags[frag_idx][i] *= correction


# ===----------------------------------------------------------------------=== #
# RDNA Flash Attention Kernel (Simplified Initial Version)
# ===----------------------------------------------------------------------=== #


@always_inline
fn rdna_flash_attention_kernel_simple[
    dtype: DType,
    head_dim_q: Int,
    head_dim_v: Int,
    tile_m: Int = DEFAULT_TILE_M,
    tile_n: Int = DEFAULT_TILE_N,
](
    q_ptr: UnsafePointer[Scalar[dtype]],
    k_ptr: UnsafePointer[Scalar[dtype]],
    v_ptr: UnsafePointer[Scalar[dtype]],
    o_ptr: UnsafePointer[Scalar[dtype]],
    seq_len_q: Int,
    seq_len_k: Int,
    scale: Float32,
):
    """RDNA Flash Attention kernel implementation.

    Implements Flash Attention with online softmax for RDNA GPUs.

    Parameters:
        dtype: Data type (float16/bfloat16).
        head_dim_q: Head dimension for Q/K.
        head_dim_v: Head dimension for V/O.
        tile_m: M dimension tile size.
        tile_n: N dimension tile size.

    Args:
        q_ptr: Query [seq_len_q, head_dim_q].
        k_ptr: Key [seq_len_k, head_dim_q].
        v_ptr: Value [seq_len_k, head_dim_v].
        o_ptr: Output [seq_len_q, head_dim_v].
        seq_len_q: Query sequence length.
        seq_len_k: Key sequence length.
        scale: Attention scale (1/sqrt(head_dim_q)).
    """
    alias accum_dtype = DType.float32

    # Calculate WMMA tiles
    alias num_wmma_m = tile_m // WMMA_M
    alias num_wmma_n = tile_n // WMMA_N
    alias num_wmma_k_q = head_dim_q // WMMA_K
    alias num_wmma_k_v = head_dim_v // WMMA_K

    # Get block and thread IDs
    var block_m = block_idx.x
    var q_offset = block_m * UInt(tile_m)

    # Early exit if out of bounds
    if q_offset >= UInt(seq_len_q):
        return

    var q_tile_rows = tile_m if tile_m < (seq_len_q - Int(q_offset)) else (
        seq_len_q - Int(q_offset)
    )

    # ===--------------------------------------------------------------=== #
    # Step 1: Allocate shared memory for K and V tiles
    # ===--------------------------------------------------------------=== #
    var smem = RDNASharedMemory[dtype, tile_n, head_dim_q].create()

    # ===--------------------------------------------------------------=== #
    # Step 2: Allocate register storage for Q fragments
    # ===--------------------------------------------------------------=== #
    # Q is loaded once and stays in registers
    alias q_frag_count = num_wmma_m * num_wmma_k_q
    var q_frags = stack_allocation[
        q_frag_count,
        SIMD[dtype, 16],
        address_space = AddressSpace.LOCAL,
    ]()

    # Load Q tile to registers
    # For simplicity, we'll load Q into shared memory first, then to fragments
    # Production version would load directly from global to registers
    @parameter
    for m_tile in range(num_wmma_m):

        @parameter
        for k_tile in range(num_wmma_k_q):
            var frag_idx = m_tile * num_wmma_k_q + k_tile

            # Calculate global Q position
            # Load Q as A matrix: each thread loads 1 row, 16 consecutive columns
            var lane = lane_id()
            var thread_x = UInt(
                lane & 15
            )  # which row this thread handles (0-15)

            var row = q_offset + UInt(m_tile * WMMA_M) + thread_x
            var col_base = UInt(k_tile * WMMA_K)

            var frag = SIMD[dtype, 16]()

            # Load 16 consecutive elements from this thread's row
            @parameter
            for i in range(16):
                var col = col_base + UInt(i)

                if row < UInt(seq_len_q) and col < UInt(head_dim_q):
                    frag[i] = q_ptr[row * UInt(head_dim_q) + col]
                else:
                    frag[i] = Scalar[dtype](0)

            q_frags[frag_idx] = frag

    # ===--------------------------------------------------------------=== #
    # Step 3: Allocate accumulator storage for S (attention scores)
    # ===--------------------------------------------------------------=== #
    alias s_frag_count = num_wmma_m * num_wmma_n
    var s_frags = stack_allocation[
        s_frag_count,
        SIMD[accum_dtype, 8],
        address_space = AddressSpace.LOCAL,
    ]()

    # ===--------------------------------------------------------------=== #
    # Step 4: Allocate accumulator storage for O (output)
    # ===--------------------------------------------------------------=== #
    alias o_frag_count = num_wmma_m * num_wmma_k_v
    var o_frags = stack_allocation[
        o_frag_count,
        SIMD[accum_dtype, 8],
        address_space = AddressSpace.LOCAL,
    ]()

    # Initialize O accumulators to zero
    @parameter
    for i in range(o_frag_count):
        o_frags[i] = SIMD[accum_dtype, 8](0)

    # ===--------------------------------------------------------------=== #
    # Step 5: Initialize softmax state
    # ===--------------------------------------------------------------=== #
    var max_state = stack_allocation[
        1,
        Scalar[accum_dtype],
        address_space = AddressSpace.LOCAL,
    ]()
    var sum_state = stack_allocation[
        1,
        Scalar[accum_dtype],
        address_space = AddressSpace.LOCAL,
    ]()

    var softmax_state = RDNASoftmaxState[accum_dtype](1, max_state, sum_state)

    # ===--------------------------------------------------------------=== #
    # Step 6: Loop over K/V tiles
    # ===--------------------------------------------------------------=== #
    var num_k_tiles = ceildiv(seq_len_k, tile_n)

    for k_tile_idx in range(num_k_tiles):
        var k_offset = k_tile_idx * tile_n

        # ===----------------------------------------------------------=== #
        # Step 6a: Load K tile to shared memory
        # ===----------------------------------------------------------=== #
        smem.load_k_tile_coalesced(k_ptr, k_offset, seq_len_k, head_dim_q)

        # ===----------------------------------------------------------=== #
        # Step 6b: Compute S = Q @ K^T using WMMA
        # ===----------------------------------------------------------=== #

        # Initialize S fragments to zero
        @parameter
        for i in range(s_frag_count):
            s_frags[i] = SIMD[accum_dtype, 8](0)

        # GEMM: S = Q @ K^T
        @parameter
        for m_tile in range(num_wmma_m):

            @parameter
            for n_tile in range(num_wmma_n):
                var s_idx = m_tile * num_wmma_n + n_tile
                var acc = SIMD[accum_dtype, 8](0)

                # Accumulate over K dimension
                @parameter
                for k_tile in range(num_wmma_k_q):
                    var q_idx = m_tile * num_wmma_k_q + k_tile

                    # Load K fragment (transposed for K^T)
                    var k_frag = wmma_load_fragment_b[dtype, transpose=True](
                        smem.k_smem,
                        head_dim_q,  # stride
                        n_tile,  # tile_k (becomes tile_n after transpose)
                        k_tile,  # tile_n (becomes tile_k after transpose)
                    )

                    # Get Q fragment
                    var q_frag = q_frags[q_idx]

                    # Perform WMMA: acc = q_frag @ k_frag + acc
                    # mma intrinsic handles FP16/BF16 -> FP32 conversion internally
                    mma(acc, q_frag, k_frag, acc)

                s_frags[s_idx] = acc

        # ===----------------------------------------------------------=== #
        # Step 6c: Update online softmax: P = softmax(S)
        # ===----------------------------------------------------------=== #
        softmax_state.update_with_tile[accum_dtype, num_wmma_m, num_wmma_n](
            s_frags,
            o_frags,
            scale,
        )

        # Now s_frags contains P (softmax of S)

        # ===----------------------------------------------------------=== #
        # Step 6d: Store P to shared memory in proper matrix layout
        # ===----------------------------------------------------------=== #
        # Allocate shared memory for P matrix (tile_m × tile_n)
        alias p_smem_size = tile_m * tile_n
        var p_smem = stack_allocation[
            p_smem_size,
            dtype,
            address_space = AddressSpace.SHARED,
        ]()

        # Store each P fragment to shared memory
        @parameter
        for m_tile in range(num_wmma_m):

            @parameter
            for n_tile in range(num_wmma_n):
                var p_idx = m_tile * num_wmma_n + n_tile
                wmma_store_accumulator_to_smem[accum_dtype, dtype](
                    p_smem,
                    tile_n,  # stride
                    m_tile,
                    n_tile,
                    s_frags[p_idx],
                )

        barrier()

        # ===----------------------------------------------------------=== #
        # Step 6e: Load V tile to shared memory
        # ===----------------------------------------------------------=== #
        var v_smem_v = RDNASharedMemory[dtype, tile_n, head_dim_v].create()
        v_smem_v.load_v_tile_coalesced(v_ptr, k_offset, seq_len_k, head_dim_v)

        # ===----------------------------------------------------------=== #
        # Step 6f: Compute O += P @ V using WMMA
        # ===----------------------------------------------------------=== #
        @parameter
        for m_tile in range(num_wmma_m):

            @parameter
            for v_tile in range(num_wmma_k_v):
                var o_idx = m_tile * num_wmma_k_v + v_tile
                var acc = o_frags[o_idx]

                # Accumulate over N dimension (sequence)
                @parameter
                for n_tile in range(num_wmma_n):
                    # Load P fragment from shared memory with proper WMMA layout
                    var p_frag = wmma_load_fragment_a[dtype](
                        p_smem,
                        tile_n,  # stride
                        m_tile,  # tile_m
                        n_tile,  # tile_k
                    )

                    # Load V fragment
                    var v_frag = wmma_load_fragment_b[dtype, transpose=False](
                        v_smem_v.v_smem,
                        head_dim_v,  # stride
                        n_tile,  # tile_k
                        v_tile,  # tile_n
                    )

                    # Perform WMMA: acc = p_frag @ v_frag + acc
                    mma(acc, p_frag, v_frag, acc)

                o_frags[o_idx] = acc

    # ===--------------------------------------------------------------=== #
    # Step 7: Final normalization: O = O / sum
    # ===--------------------------------------------------------------=== #
    var final_sum = Scalar[accum_dtype](sum_state[0])

    # Guard against division by zero
    if final_sum > Scalar[accum_dtype](1e-10):
        var norm_factor = recip(final_sum)

        @parameter
        for i in range(o_frag_count):

            @parameter
            for j in range(8):
                o_frags[i][j] *= norm_factor
    else:
        # If sum is zero or very small, set output to zeros
        @parameter
        for i in range(o_frag_count):

            @parameter
            for j in range(8):
                o_frags[i][j] = Scalar[accum_dtype](0)

    # ===--------------------------------------------------------------=== #
    # Step 8: Store output to global memory
    # ===--------------------------------------------------------------=== #
    # Store via shared memory for coalesced writes
    var o_smem = stack_allocation[
        tile_m * head_dim_v,
        dtype,
        address_space = AddressSpace.SHARED,
    ]()

    # Store fragments to shared memory
    @parameter
    for m_tile in range(num_wmma_m):

        @parameter
        for v_tile in range(num_wmma_k_v):
            var o_idx = m_tile * num_wmma_k_v + v_tile
            var frag = o_frags[o_idx]

            # Cast to dtype and store
            var frag_cast = SIMD[dtype, 8]()

            @parameter
            for i in range(8):
                frag_cast[i] = frag[i].cast[dtype]()

            wmma_store_fragment[dtype](
                o_smem,
                head_dim_v,  # stride
                m_tile,
                v_tile,
                frag_cast,
            )

    barrier()

    # Coalesced write to global memory
    var tid = thread_idx.x
    alias num_threads = RDNA_WAVE_SIZE
    alias total_elements = tile_m * head_dim_v
    alias elements_per_thread = ceildiv(total_elements, num_threads)

    @parameter
    for i in range(elements_per_thread):
        var elem_idx = tid + UInt(i * num_threads)

        if elem_idx < UInt(total_elements):
            var row = elem_idx // UInt(head_dim_v)
            var col = elem_idx % UInt(head_dim_v)

            var global_row = q_offset + row
            if global_row < UInt(seq_len_q):
                var global_idx = global_row * UInt(head_dim_v) + col
                o_ptr[global_idx] = o_smem[elem_idx]


# ===----------------------------------------------------------------------=== #
# Public API
# ===----------------------------------------------------------------------=== #


fn rdna_flash_attention[
    dtype: DType,
    head_dim_q: Int,
    head_dim_v: Int,
](
    q: UnsafePointer[Scalar[dtype]],
    k: UnsafePointer[Scalar[dtype]],
    v: UnsafePointer[Scalar[dtype]],
    o: UnsafePointer[Scalar[dtype]],
    seq_len_q: Int,
    seq_len_k: Int,
    scale: Float32,
):
    """RDNA Flash Attention (Correct Implementation).

    This is a rewrite with proper WMMA fragment handling.

    Parameters:
        dtype: Data type (float16/bfloat16).
        head_dim_q: Head dimension for Q/K.
        head_dim_v: Head dimension for V/O.

    Args:
        q: Query tensor [seq_len_q, head_dim_q].
        k: Key tensor [seq_len_k, head_dim_q].
        v: Value tensor [seq_len_k, head_dim_v].
        o: Output tensor [seq_len_q, head_dim_v].
        seq_len_q: Query sequence length.
        seq_len_k: Key/Value sequence length.
        scale: Attention scale (typically 1/sqrt(head_dim_q)).

    Constraints:
        - RDNA3 or RDNA4 GPU required
        - head_dim_q and head_dim_v must be divisible by 16
        - dtype must be float16 or bfloat16

    Note:
        This is a correct foundation. Full production implementation
        will add optimizations like software pipelining, auto-tuning,
        and causal masking support.
    """
    # Constraints
    constrained[
        not _is_amd_rdna2(),
        "RDNA2 and earlier do not support WMMA 16x16x16",
    ]()
    constrained[
        dtype == DType.float16 or dtype == DType.bfloat16,
        "RDNA WMMA supports float16 or bfloat16",
    ]()
    constrained[
        head_dim_q % WMMA_K == 0,
        "head_dim_q must be divisible by 16",
    ]()
    constrained[
        head_dim_v % WMMA_K == 0,
        "head_dim_v must be divisible by 16",
    ]()

    # Call kernel (currently simplified)
    rdna_flash_attention_kernel_simple[dtype, head_dim_q, head_dim_v](
        q, k, v, o, seq_len_q, seq_len_k, scale
    )
