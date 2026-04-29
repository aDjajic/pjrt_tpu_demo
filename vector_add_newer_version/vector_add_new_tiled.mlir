module attributes {stable_mosaic.version = 11 : i64} {
  func.func @add_kernel(%arg0: memref<8x128xf32, #tpu.tiled<(8,128),[1,1]>, #tpu.memory_space<vmem>>, %arg1: memref<8x128xf32, #tpu.tiled<(8,128),[1,1]>, #tpu.memory_space<vmem>>, %arg2: memref<8x128xf32, #tpu.tiled<(8,128),[1,1]>, #tpu.memory_space<vmem>>) attributes {dimension_semantics = [], scalar_prefetch = 0 : i64, scratch_operands = 0 : i64, tpu.core_type = #tpu.core_type<tc>} {
    %0 = tpu.reinterpret_cast %arg0 : memref<8x128xf32, #tpu.tiled<(8,128),[1,1]>, #tpu.memory_space<vmem>> -> memref<8x128xf32, #tpu.tiled<(1,128),[1,1]>, #tpu.memory_space<vmem>>
    %1 = tpu.erase_memref_layout %0 : memref<8x128xf32, #tpu.tiled<(1,128),[1,1]>, #tpu.memory_space<vmem>> -> memref<8x128xf32, #tpu.memory_space<vmem>>
    %2 = tpu.reinterpret_cast %arg1 : memref<8x128xf32, #tpu.tiled<(8,128),[1,1]>, #tpu.memory_space<vmem>> -> memref<8x128xf32, #tpu.tiled<(1,128),[1,1]>, #tpu.memory_space<vmem>>
    %3 = tpu.erase_memref_layout %2 : memref<8x128xf32, #tpu.tiled<(1,128),[1,1]>, #tpu.memory_space<vmem>> -> memref<8x128xf32, #tpu.memory_space<vmem>>
    %4 = tpu.reinterpret_cast %arg2 : memref<8x128xf32, #tpu.tiled<(8,128),[1,1]>, #tpu.memory_space<vmem>> -> memref<8x128xf32, #tpu.tiled<(1,128),[1,1]>, #tpu.memory_space<vmem>>
    %5 = tpu.erase_memref_layout %4 : memref<8x128xf32, #tpu.tiled<(1,128),[1,1]>, #tpu.memory_space<vmem>> -> memref<8x128xf32, #tpu.memory_space<vmem>>
    %c0 = arith.constant 0 : index
    %6 = vector.load %1[%c0, %c0] : memref<8x128xf32, #tpu.memory_space<vmem>>, vector<8x128xf32>
    %7 = vector.load %3[%c0, %c0] : memref<8x128xf32, #tpu.memory_space<vmem>>, vector<8x128xf32>
    %8 = arith.addf %6, %7 : vector<8x128xf32>
    tpu.vector_store %5[%c0, %c0], %8 {strides = array<i32>} : memref<8x128xf32, #tpu.memory_space<vmem>>, vector<8x128xf32>,
    return
  }
}
