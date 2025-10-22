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
"""GPU tests for activation functions (GELU, SiLU/Swish, etc.)."""

from math import iota
from random import rand, seed

from gpu import block_idx, thread_idx, block_dim
from gpu.host import DeviceContext
from nn.activations import gelu, gelu_approximate, silu, swish
from testing import assert_almost_equal


fn test_gelu_gpu[dtype: DType](ctx: DeviceContext) raises:
    """Test GELU activation on GPU."""
    print("== test_gelu_gpu[", dtype, "]")

    alias N = 1024
    var input_h = UnsafePointer[Scalar[dtype]].alloc(N)
    var output_h = UnsafePointer[Scalar[dtype]].alloc(N)
    var expected_h = UnsafePointer[Scalar[dtype]].alloc(N)

    # Generate test data
    seed(42)
    rand[dtype](input_h, N)

    # Scale to [-4, 4] range for better coverage
    for i in range(N):
        input_h[i] = (input_h[i] - 0.5) * 8.0

    # Compute expected values on CPU
    for i in range(N):
        expected_h[i] = gelu[dtype, 1](input_h[i])[0]

    # Allocate GPU buffers
    var input_d = ctx.enqueue_create_buffer[dtype](N)
    var output_d = ctx.enqueue_create_buffer[dtype](N)

    # Copy input to GPU
    ctx.enqueue_copy(input_d, input_h)

    # Get raw pointers for kernel
    var input_ptr = input_d.unsafe_ptr()
    var output_ptr = output_d.unsafe_ptr()

    # Launch GPU kernel
    @parameter
    @__copy_capture(input_ptr, output_ptr)
    fn gelu_kernel():
        var tid = Int(thread_idx.x + block_idx.x * block_dim.x)
        if tid < N:
            var val = input_ptr[tid]
            output_ptr[tid] = gelu[dtype, 1](val)[0]

    alias kernel = gelu_kernel
    ctx.enqueue_function_checked[kernel, kernel](
        grid_dim=(N + 255) // 256,
        block_dim=256,
    )

    # Copy result back
    ctx.enqueue_copy(output_h, output_d)
    ctx.synchronize()

    # Verify results
    var max_error = Scalar[dtype](0)
    var errors = 0
    for i in range(N):
        var diff = abs(output_h[i] - expected_h[i])
        if diff > max_error:
            max_error = diff
        if diff > 0.001:
            errors += 1
            if errors <= 5:
                print("  Error at", i, ":", output_h[i], "vs", expected_h[i])

    print("  Max error:", max_error, "  Total errors:", errors, "/", N)

    if errors > 0:
        raise Error("GELU GPU test failed")

    # Cleanup
    input_h.free()
    output_h.free()
    expected_h.free()
    _ = input_d
    _ = output_d
    print("  ✓ GELU GPU test PASSED")


fn test_silu_gpu[dtype: DType](ctx: DeviceContext) raises:
    """Test SiLU/Swish activation on GPU."""
    print("== test_silu_gpu[", dtype, "]")

    alias N = 1024
    var input_h = UnsafePointer[Scalar[dtype]].alloc(N)
    var output_h = UnsafePointer[Scalar[dtype]].alloc(N)
    var expected_h = UnsafePointer[Scalar[dtype]].alloc(N)

    # Generate test data
    seed(123)
    rand[dtype](input_h, N)

    # Scale to [-4, 4] range
    for i in range(N):
        input_h[i] = (input_h[i] - 0.5) * 8.0

    # Compute expected values on CPU
    for i in range(N):
        expected_h[i] = silu[dtype, 1](input_h[i])[0]

    # Allocate GPU buffers
    var input_d = ctx.enqueue_create_buffer[dtype](N)
    var output_d = ctx.enqueue_create_buffer[dtype](N)

    # Copy input to GPU
    ctx.enqueue_copy(input_d, input_h)

    # Get raw pointers for kernel
    var input_ptr = input_d.unsafe_ptr()
    var output_ptr = output_d.unsafe_ptr()

    # Launch GPU kernel
    @parameter
    @__copy_capture(input_ptr, output_ptr)
    fn silu_kernel():
        var tid = Int(thread_idx.x + block_idx.x * block_dim.x)
        if tid < N:
            var val = input_ptr[tid]
            output_ptr[tid] = silu[dtype, 1](val)[0]

    alias kernel = silu_kernel
    ctx.enqueue_function_checked[kernel, kernel](
        grid_dim=(N + 255) // 256,
        block_dim=256,
    )

    # Copy result back
    ctx.enqueue_copy(output_h, output_d)
    ctx.synchronize()

    # Verify results
    var max_error = Scalar[dtype](0)
    var errors = 0
    for i in range(N):
        var diff = abs(output_h[i] - expected_h[i])
        if diff > max_error:
            max_error = diff
        if diff > 0.001:
            errors += 1
            if errors <= 5:
                print("  Error at", i, ":", output_h[i], "vs", expected_h[i])

    print("  Max error:", max_error, "  Total errors:", errors, "/", N)

    if errors > 0:
        raise Error("SiLU GPU test failed")

    # Cleanup
    input_h.free()
    output_h.free()
    expected_h.free()
    _ = input_d
    _ = output_d
    print("  ✓ SiLU GPU test PASSED")


fn test_swish_alias_gpu[dtype: DType](ctx: DeviceContext) raises:
    """Test that Swish is an alias for SiLU."""
    print("== test_swish_alias_gpu[", dtype, "]")

    # Simple test to verify swish == silu
    var test_values = SIMD[dtype, 4](-2.0) + iota[dtype, 4]()

    for i in range(4):
        var val = test_values[i]
        var silu_result = silu[dtype, 1](val)[0]
        var swish_result = swish[dtype, 1](val)[0]

        if abs(silu_result - swish_result) > 1e-6:
            print("  Mismatch at", i, ":", silu_result, "vs", swish_result)
            raise Error("Swish/SiLU alias test failed")

    print("  ✓ Swish alias test PASSED")


def main():
    print("Testing GPU activation functions on RDNA...")

    with DeviceContext() as ctx:
        # Test GELU
        test_gelu_gpu[DType.float32](ctx)
        test_gelu_gpu[DType.float16](ctx)
        test_gelu_gpu[DType.bfloat16](ctx)

        # Test SiLU/Swish
        test_silu_gpu[DType.float32](ctx)
        test_silu_gpu[DType.float16](ctx)
        test_silu_gpu[DType.bfloat16](ctx)

        # Test alias
        test_swish_alias_gpu[DType.float32](ctx)

    print("")
    print("✅ All GPU activation tests PASSED")
    print("")
    print("Activation functions verified on RDNA:")
    print("  • GELU (exact) - FP32, FP16, BF16")
    print("  • SiLU/Swish - FP32, FP16, BF16")
    print("")
    print("These are critical for modern LLMs:")
    print("  • GELU: GPT, BERT, T5, many transformers")
    print("  • SiLU: LLaMA, Mistral, Qwen, modern architectures")
