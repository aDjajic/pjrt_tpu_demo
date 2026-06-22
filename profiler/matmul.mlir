module {
  func.func @matmul_kernel(%arg0: memref<256x256xf32, #tpu.memory_space<vmem>>, %arg1: memref<256x256xf32, #tpu.memory_space<vmem>>, %arg2: memref<256x256xf32, #tpu.memory_space<vmem>>) attributes {dimension_semantics = [], scalar_prefetch = 0 : i64, scratch_operands = 0 : i64, tpu.core_type = #tpu.core_type<tc>} {
    %cst = arith.constant dense<0.000000e+00> : vector<256x256xf32>
    %c0 = arith.constant 0 : index
    %0 = vector.load %arg0[%c0, %c0] : memref<256x256xf32, #tpu.memory_space<vmem>>, vector<256x256xf32>
    %1 = vector.load %arg1[%c0, %c0] : memref<256x256xf32, #tpu.memory_space<vmem>>, vector<256x256xf32>
    %2 = tpu.matmul %0, %1, %cst {dimension_numbers = #tpu.dot_dimension_numbers<[1], [0], [0], [1], [0, 0, 1, 1], [], []>, transpose_lhs_hint = false} : vector<256x256xf32>, vector<256x256xf32>, vector<256x256xf32> -> vector<256x256xf32>
    tpu.vector_store %arg2[%c0, %c0], %2 {strides = array<i32>} : memref<256x256xf32, #tpu.memory_space<vmem>>, vector<256x256xf32>, 
    return
  }
}