module {
  func.func @matmul_kernel(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<512x512xbf16, #tpu.memory_space<vmem>>, %arg4: memref<512x512xbf16, #tpu.memory_space<vmem>>, %arg5: memref<512x512xbf16, #tpu.memory_space<vmem>>, %arg6: memref<512x512xf32, #tpu.memory_space<vmem>>) attributes {dimension_semantics = [#tpu.dimension_semantics<parallel>, #tpu.dimension_semantics<parallel>, #tpu.dimension_semantics<arbitrary>], iteration_bounds = array<i64: 8, 8, 8>, scalar_prefetch = 0 : i64, scratch_operands = 1 : i64, tpu.core_type = #tpu.core_type<tc>, window_params = [{transform_indices = @transform_0, window_bounds = array<i64: 512, 512>}, {transform_indices = @transform_1, window_bounds = array<i64: 512, 512>}, {transform_indices = @transform_2, window_bounds = array<i64: 512, 512>}]} {
    %c7_i32 = arith.constant 7 : i32
    %cst = arith.constant dense<0.000000e+00> : vector<512x512xf32>
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.cmpi eq, %arg2, %c0_i32 : i32
    scf.if %0 {
      tpu.vector_store %arg6[%c0, %c0], %cst {strides = array<i32>} : memref<512x512xf32, #tpu.memory_space<vmem>>, vector<512x512xf32>, 
    }
    %1 = vector.load %arg6[%c0, %c0] : memref<512x512xf32, #tpu.memory_space<vmem>>, vector<512x512xf32>
    %2 = vector.load %arg3[%c0, %c0] : memref<512x512xbf16, #tpu.memory_space<vmem>>, vector<512x512xbf16>
    %3 = vector.load %arg4[%c0, %c0] : memref<512x512xbf16, #tpu.memory_space<vmem>>, vector<512x512xbf16>
    %4 = tpu.matmul %2, %3, %cst {dimension_numbers = #tpu.dot_dimension_numbers<[1], [0], [0], [1], [0, 0, 1, 1], [], []>, transpose_lhs_hint = false} : vector<512x512xbf16>, vector<512x512xbf16>, vector<512x512xf32> -> vector<512x512xf32>
    %5 = arith.addf %1, %4 : vector<512x512xf32>
    tpu.vector_store %arg6[%c0, %c0], %5 {strides = array<i32>} : memref<512x512xf32, #tpu.memory_space<vmem>>, vector<512x512xf32>, 
    %6 = arith.cmpi eq, %arg2, %c7_i32 : i32
    scf.if %6 {
      %7 = vector.load %arg6[%c0, %c0] : memref<512x512xf32, #tpu.memory_space<vmem>>, vector<512x512xf32>
      %8 = arith.truncf %7 : vector<512x512xf32> to vector<512x512xbf16>
      tpu.vector_store %arg5[%c0, %c0], %8 {strides = array<i32>} : memref<512x512xbf16, #tpu.memory_space<vmem>>, vector<512x512xbf16>, 
    }
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
