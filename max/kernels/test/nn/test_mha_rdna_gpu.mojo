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
"""GPU execution test for RDNA Flash Attention implementation."""

from math import sqrt
from gpu.host import DeviceContext
from nn.mha_rdna import rdna_flash_attention
from testing import assert_true


fn test_rdna_flash_attention_small() raises:
    """Test RDNA flash attention with small matrices on GPU.

    Tests a minimal flash attention operation:
    - seq_len = 32 tokens
    - head_dim = 64
    - FP16 precision

    Verifies the kernel executes without errors and produces valid output.
    """
    print("Testing RDNA Flash Attention on GPU (FP16, 32x64)...")

    with DeviceContext() as ctx:
        # Small test dimensions (all divisible by 16)
        alias seq_len_q = 32
        alias seq_len_k = 32
        alias head_dim_q = 64
        alias head_dim_v = 64
        alias dtype = DType.float16

        # Calculate scale (1/sqrt(head_dim))
        var scale = Float32(1.0 / sqrt(Float64(head_dim_q)))

        # Allocate host memory
        var q_size = seq_len_q * head_dim_q
        var k_size = seq_len_k * head_dim_q
        var v_size = seq_len_k * head_dim_v
        var o_size = seq_len_q * head_dim_v

        var q_host = UnsafePointer[Float16].alloc(q_size)
        var k_host = UnsafePointer[Float16].alloc(k_size)
        var v_host = UnsafePointer[Float16].alloc(v_size)
        var o_host = UnsafePointer[Float16].alloc(o_size)

        # Initialize with simple pattern
        # Q: each position gets a unique small value
        for i in range(q_size):
            q_host[i] = Float16(0.1 + 0.01 * (i % 10))

        # K: similar pattern (will create some attention scores)
        for i in range(k_size):
            k_host[i] = Float16(0.1 + 0.01 * (i % 10))

        # V: ones (makes it easy to verify non-zero output)
        for i in range(v_size):
            v_host[i] = Float16(1.0)

        # O: initialize to zero
        for i in range(o_size):
            o_host[i] = Float16(0.0)

        # Allocate device buffers
        var q_dev = ctx.enqueue_create_buffer[dtype](q_size)
        var k_dev = ctx.enqueue_create_buffer[dtype](k_size)
        var v_dev = ctx.enqueue_create_buffer[dtype](v_size)
        var o_dev = ctx.enqueue_create_buffer[dtype](o_size)

        # Copy inputs to device
        ctx.enqueue_copy(q_dev, q_host)
        ctx.enqueue_copy(k_dev, k_host)
        ctx.enqueue_copy(v_dev, v_host)
        ctx.enqueue_copy(o_dev, o_host)

        # Launch flash attention kernel
        # Grid: 1 block per query tile, Block: 128 threads per block
        alias tile_m = 16  # Process 16 query tokens per block
        var num_blocks = (seq_len_q + tile_m - 1) // tile_m

        alias kernel = rdna_flash_attention[dtype, head_dim_q, head_dim_v]
        ctx.enqueue_function_checked[kernel, kernel](
            q_dev,
            k_dev,
            v_dev,
            o_dev,
            seq_len_q,
            seq_len_k,
            scale,
            grid_dim=(num_blocks,),
            block_dim=128,
        )

        # Copy result back
        ctx.enqueue_copy(o_host, o_dev)

        # Verify output is valid (not NaN, not all zeros)
        var non_zero_count = 0
        var nan_count = 0
        var sum = Float32(0.0)

        for i in range(o_size):
            var val = Float32(o_host[i])
            if val != 0.0:
                non_zero_count += 1
            if val != val:  # NaN check
                nan_count += 1
            sum += val

        # Print diagnostics
        print("  Output statistics:")
        print("    Non-zero elements:", non_zero_count, "/", o_size)
        print("    NaN count:", nan_count)
        print("    Sum:", sum)
        print("    First 10 values:", end="")
        for i in range(min(10, o_size)):
            print(" ", Float32(o_host[i]), end="")
        print()

        # Cleanup
        _ = q_dev
        _ = k_dev
        _ = v_dev
        _ = o_dev
        q_host.free()
        k_host.free()
        v_host.free()
        o_host.free()

        # Basic sanity checks
        # TODO: Flash attention kernel has bugs producing NaN values
        # This test documents the current state - WMMA works but the
        # flash attention algorithm needs debugging
        if nan_count > 0:
            print("⚠️  KNOWN ISSUE: Flash attention produces NaN values")
            print(
                "   WMMA operations work correctly (see"
                " test_mma_fp16_fp32.mojo)"
            )
            print("   Flash attention algorithm needs debugging")
            raise Error("Flash attention kernel produces NaN - needs fixing")

        assert_true(non_zero_count > 0, "Output is all zeros")
        assert_true(sum > 0.0, "Output sum should be positive (V is all ones)")

        print("✅ RDNA Flash Attention GPU test PASSED")


fn main() raises:
    """Run GPU flash attention tests."""
    test_rdna_flash_attention_small()
    print("✓ All RDNA flash attention GPU tests passed!")
