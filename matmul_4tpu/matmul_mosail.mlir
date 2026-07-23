module {
  func.func @matmul_kernel(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<128x256xf32, #tpu.memory_space<vmem>>, %arg4: memref<256x256xf32, #tpu.memory_space<vmem>>, %arg5: memref<128x256xf32, #tpu.memory_space<vmem>>) attributes {dimension_semantics = [#tpu.dimension_semantics<arbitrary>, #tpu.dimension_semantics<arbitrary>, #tpu.dimension_semantics<arbitrary>], iteration_bounds = array<i64: 2, 4, 4>, scalar_prefetch = 0 : i64, scratch_operands = 0 : i64, tpu.core_type = #tpu.core_type<tc>, window_params = [{transform_indices = @transform_0, window_bounds = array<i64: 128, 256>}, {transform_indices = @transform_1, window_bounds = array<i64: 256, 256>}, {transform_indices = @transform_2, window_bounds = array<i64: 128, 256>}]} {
    %cst = arith.constant dense<0.000000e+00> : vector<128x256xf32>
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.cmpi eq, %arg2, %c0_i32 : i32
    scf.if %0 {
      tpu.vector_store %arg5[%c0, %c0], %cst {strides = array<i32>} : memref<128x256xf32, #tpu.memory_space<vmem>>, vector<128x256xf32>, 
    }
    %1 = vector.load %arg5[%c0, %c0] : memref<128x256xf32, #tpu.memory_space<vmem>>, vector<128x256xf32>
    %2 = vector.load %arg3[%c0, %c0] : memref<128x256xf32, #tpu.memory_space<vmem>>, vector<128x256xf32>
    %3 = vector.load %arg4[%c0, %c0] : memref<256x256xf32, #tpu.memory_space<vmem>>, vector<256x256xf32>
    %4 = tpu.matmul %2, %3, %cst {dimension_numbers = #tpu.dot_dimension_numbers<[1], [0], [0], [1], [0, 0, 1, 1], [], []>, transpose_lhs_hint = false} : vector<128x256xf32>, vector<256x256xf32>, vector<128x256xf32> -> vector<128x256xf32>
    %5 = arith.addf %1, %4 : vector<128x256xf32>
    tpu.vector_store %arg5[%c0, %c0], %5 {strides = array<i32>} : memref<128x256xf32, #tpu.memory_space<vmem>>, vector<128x256xf32>, 
    return
  }
  func.func @transform_0(%arg0: i32, %arg1: i32, %arg2: i32) -> (i32, i32) {
    return %arg0, %arg2 : i32, i32
  }
  func.func @transform_1(%arg0: i32, %arg1: i32, %arg2: i32) -> (i32, i32) {
    return %arg2, %arg1 : i32, i32
  }
  func.func @transform_2(%arg0: i32, %arg1: i32, %arg2: i32) -> (i32, i32) {
    return %arg0, %arg1 : i32, i32
  }
}
