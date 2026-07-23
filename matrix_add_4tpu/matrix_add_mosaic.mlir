module {
  func.func @add_kernel(%arg0: i32, %arg1: i32, %arg2: memref<128x256xf32, #tpu.memory_space<vmem>>, %arg3: memref<128x256xf32, #tpu.memory_space<vmem>>, %arg4: memref<128x256xf32, #tpu.memory_space<vmem>>) attributes {dimension_semantics = [#tpu.dimension_semantics<arbitrary>, #tpu.dimension_semantics<arbitrary>], iteration_bounds = array<i64: 2, 4>, scalar_prefetch = 0 : i64, scratch_operands = 0 : i64, tpu.core_type = #tpu.core_type<tc>, window_params = [{transform_indices = @transform_0, window_bounds = array<i64: 128, 256>}, {transform_indices = @transform_1, window_bounds = array<i64: 128, 256>}, {transform_indices = @transform_2, window_bounds = array<i64: 128, 256>}]} {
    %c0 = arith.constant 0 : index
    %0 = vector.load %arg2[%c0, %c0] : memref<128x256xf32, #tpu.memory_space<vmem>>, vector<128x256xf32>
    %1 = vector.load %arg3[%c0, %c0] : memref<128x256xf32, #tpu.memory_space<vmem>>, vector<128x256xf32>
    %2 = arith.addf %0, %1 : vector<128x256xf32>
    tpu.vector_store %arg4[%c0, %c0], %2 {strides = array<i32>} : memref<128x256xf32, #tpu.memory_space<vmem>>, vector<128x256xf32>, 
    return
  }
  func.func @transform_0(%arg0: i32, %arg1: i32) -> (i32, i32) {
    return %arg0, %arg1 : i32, i32
  }
  func.func @transform_1(%arg0: i32, %arg1: i32) -> (i32, i32) {
    return %arg0, %arg1 : i32, i32
  }
  func.func @transform_2(%arg0: i32, %arg1: i32) -> (i32, i32) {
    return %arg0, %arg1 : i32, i32
  }
}