module {
  func.func @concat_kernel(%arg0: memref<256x512xf32, #tpu.memory_space<vmem>>, %arg1: memref<256x512xf32, #tpu.memory_space<vmem>>, %arg2: memref<256x512xf32, #tpu.memory_space<vmem>>, %arg3: memref<256x512xf32, #tpu.memory_space<vmem>>, %arg4: memref<1024x512xf32, #tpu.memory_space<vmem>>) attributes {dimension_semantics = [], scalar_prefetch = 0 : i64, scratch_operands = 0 : i64, tpu.core_type = #tpu.core_type<tc>} {
    %c768 = arith.constant 768 : index
    %c512 = arith.constant 512 : index
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    %0 = vector.load %arg0[%c0, %c0] : memref<256x512xf32, #tpu.memory_space<vmem>>, vector<256x512xf32>
    tpu.vector_store %arg4[%c0, %c0], %0 {strides = array<i32>} : memref<1024x512xf32, #tpu.memory_space<vmem>>, vector<256x512xf32>, 
    %1 = vector.load %arg1[%c0, %c0] : memref<256x512xf32, #tpu.memory_space<vmem>>, vector<256x512xf32>
    tpu.vector_store %arg4[%c256, %c0], %1 {strides = array<i32>} : memref<1024x512xf32, #tpu.memory_space<vmem>>, vector<256x512xf32>, 
    %2 = vector.load %arg2[%c0, %c0] : memref<256x512xf32, #tpu.memory_space<vmem>>, vector<256x512xf32>
    tpu.vector_store %arg4[%c512, %c0], %2 {strides = array<i32>} : memref<1024x512xf32, #tpu.memory_space<vmem>>, vector<256x512xf32>, 
    %3 = vector.load %arg3[%c0, %c0] : memref<256x512xf32, #tpu.memory_space<vmem>>, vector<256x512xf32>
    tpu.vector_store %arg4[%c768, %c0], %3 {strides = array<i32>} : memref<1024x512xf32, #tpu.memory_space<vmem>>, vector<256x512xf32>, 
    return
  }
}