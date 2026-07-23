module {
  func.func @all_reduce_kernel(%arg0: i32, %arg1: memref<8x128xf32, #tpu.memory_space<vmem>>, %arg2: memref<8x128xf32, #tpu.memory_space<vmem>>, %arg3: memref<2x8x128xf32, #tpu.memory_space<any>>, %arg4: memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>, %arg5: memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>, %arg6: memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>, %arg7: memref<!tpu.semaphore, #tpu.memory_space<semaphore_mem>>, %arg8: memref<8x128xf32, #tpu.memory_space<vmem>>) attributes {dimension_semantics = [#tpu.dimension_semantics<arbitrary>], iteration_bounds = array<i64: 4>, scalar_prefetch = 0 : i64, scratch_operands = 5 : i64, tpu.core_type = #tpu.core_type<tc>, window_params = [{pipeline_mode = #tpu.pipeline_mode<synchronous>, transform_indices = @transform_0, window_bounds = array<i64: 8, 128>}, {pipeline_mode = #tpu.pipeline_mode<synchronous>, transform_indices = @transform_1, window_bounds = array<i64: 8, 128>}, {}]} {
    %cst = arith.constant dense<0.000000e+00> : vector<8x128xf32>
    %c3_i32 = arith.constant 3 : i32
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %0 = arith.remsi %arg0, %c2_i32 : i32
    %1 = arith.subi %c1_i32, %0 : i32
    %2 = tpu.device_id : i32
    %3 = arith.remsi %2, %c4_i32 : i32
    %4 = arith.addi %3, %c1_i32 : i32
    %5 = arith.remsi %4, %c4_i32 : i32
    %6 = arith.addi %3, %c3_i32 : i32
    %7 = arith.remsi %6, %c4_i32 : i32
    %8 = arith.cmpi eq, %arg0, %c0_i32 : i32
    scf.if %8 {
      %28 = tpu.sem_barrier : memref<!tpu.semaphore, #tpu.memory_space<semaphore_mem>>
      tpu.sem_signal %28, %c1_i32 device_id %7 : memref<!tpu.semaphore, #tpu.memory_space<semaphore_mem>>
      tpu.sem_signal %28, %c1_i32 device_id %5 : memref<!tpu.semaphore, #tpu.memory_space<semaphore_mem>>
      tpu.sem_wait %28, %c2_i32 : memref<!tpu.semaphore, #tpu.memory_space<semaphore_mem>>
      "tpu.region"() ({
        %35 = tpu.sem_alloc : memref<!tpu.semaphore, #tpu.memory_space<semaphore_mem>>
        tpu.sem_signal %35, %c1_i32 device_id %7 : memref<!tpu.semaphore, #tpu.memory_space<semaphore_mem>>
        tpu.sem_signal %35, %c1_i32 device_id %5 : memref<!tpu.semaphore, #tpu.memory_space<semaphore_mem>>
        tpu.sem_wait %35, %c2_i32 : memref<!tpu.semaphore, #tpu.memory_space<semaphore_mem>>
        tpu.yield
      }) : () -> ()
      tpu.vector_store %arg2[%c0, %c0], %cst {strides = array<i32>} : memref<8x128xf32, #tpu.memory_space<vmem>>, vector<8x128xf32>,
      tpu.vector_store %arg8[%c0, %c0], %cst {strides = array<i32>} : memref<8x128xf32, #tpu.memory_space<vmem>>, vector<8x128xf32>, 
      %29 = tpu.memref_slice %arg3[%0, %c0_i32, %c0_i32] : memref<2x8x128xf32, #tpu.memory_space<any>> -> memref<1x8x128xf32, #tpu.memory_space<any>>
      %30 = tpu.memref_squeeze %29 : memref<1x8x128xf32, #tpu.memory_space<any>> -> memref<8x128xf32, #tpu.memory_space<any>>
      tpu.enqueue_dma source(%arg1 : memref<8x128xf32, #tpu.memory_space<vmem>>) target(%30 : memref<8x128xf32, #tpu.memory_space<any>>) source_semaphore(%arg6 : memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>) target_semaphore(%arg5 : memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>) device_id(%5)
      %31 = tpu.memref_slice %arg3[%0, %c0_i32, %c0_i32] : memref<2x8x128xf32, #tpu.memory_space<any>> -> memref<1x8x128xf32, #tpu.memory_space<any>>
      %32 = tpu.memref_squeeze %31 : memref<1x8x128xf32, #tpu.memory_space<any>> -> memref<8x128xf32, #tpu.memory_space<any>>
      tpu.wait_dma2 semaphore(%arg6 : memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>) src(%32 : memref<8x128xf32, #tpu.memory_space<any>>) dst(%arg1 : memref<8x128xf32, #tpu.memory_space<vmem>>) device_id(%c0_i32) core_id(%c0_i32)
      %33 = tpu.memref_slice %arg3[%0, %c0_i32, %c0_i32] : memref<2x8x128xf32, #tpu.memory_space<any>> -> memref<1x8x128xf32, #tpu.memory_space<any>>
      %34 = tpu.memref_squeeze %33 : memref<1x8x128xf32, #tpu.memory_space<any>> -> memref<8x128xf32, #tpu.memory_space<any>>
      tpu.wait_dma2 semaphore(%arg5 : memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>) src(%arg1 : memref<8x128xf32, #tpu.memory_space<vmem>>) dst(%34 : memref<8x128xf32, #tpu.memory_space<any>>) device_id(%5)
    }
    tpu.sem_signal %arg7, %c1_i32 device_id %7 : memref<!tpu.semaphore, #tpu.memory_space<semaphore_mem>>
    %9 = tpu.memref_slice %arg3[%0, %c0_i32, %c0_i32] : memref<2x8x128xf32, #tpu.memory_space<any>> -> memref<1x8x128xf32, #tpu.memory_space<any>>
    %10 = tpu.memref_squeeze %9 : memref<1x8x128xf32, #tpu.memory_space<any>> -> memref<8x128xf32, #tpu.memory_space<any>>
    tpu.enqueue_dma source(%10 : memref<8x128xf32, #tpu.memory_space<any>>) target(%arg8 : memref<8x128xf32, #tpu.memory_space<vmem>>) target_semaphore(%arg4 : memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>)
    tpu.sem_wait %arg7, %c1_i32 : memref<!tpu.semaphore, #tpu.memory_space<semaphore_mem>>
    %11 = tpu.memref_slice %arg3[%1, %c0_i32, %c0_i32] : memref<2x8x128xf32, #tpu.memory_space<any>> -> memref<1x8x128xf32, #tpu.memory_space<any>>
    %12 = tpu.memref_squeeze %11 : memref<1x8x128xf32, #tpu.memory_space<any>> -> memref<8x128xf32, #tpu.memory_space<any>>
    %13 = tpu.memref_slice %arg3[%0, %c0_i32, %c0_i32] : memref<2x8x128xf32, #tpu.memory_space<any>> -> memref<1x8x128xf32, #tpu.memory_space<any>>
    %14 = tpu.memref_squeeze %13 : memref<1x8x128xf32, #tpu.memory_space<any>> -> memref<8x128xf32, #tpu.memory_space<any>>
    tpu.enqueue_dma source(%14 : memref<8x128xf32, #tpu.memory_space<any>>) target(%12 : memref<8x128xf32, #tpu.memory_space<any>>) source_semaphore(%arg6 : memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>) target_semaphore(%arg5 : memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>) device_id(%5)
    %15 = tpu.memref_slice %arg3[%0, %c0_i32, %c0_i32] : memref<2x8x128xf32, #tpu.memory_space<any>> -> memref<1x8x128xf32, #tpu.memory_space<any>>
    %16 = tpu.memref_squeeze %15 : memref<1x8x128xf32, #tpu.memory_space<any>> -> memref<8x128xf32, #tpu.memory_space<any>>
    tpu.wait_dma2 semaphore(%arg4 : memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>) src(%16 : memref<8x128xf32, #tpu.memory_space<any>>) dst(%arg8 : memref<8x128xf32, #tpu.memory_space<vmem>>)
    %17 = vector.load %arg2[%c0, %c0] : memref<8x128xf32, #tpu.memory_space<vmem>>, vector<8x128xf32>
    %18 = vector.load %arg8[%c0, %c0] : memref<8x128xf32, #tpu.memory_space<vmem>>, vector<8x128xf32>
    %19 = arith.addf %17, %18 : vector<8x128xf32>
    tpu.vector_store %arg2[%c0, %c0], %19 {strides = array<i32>} : memref<8x128xf32, #tpu.memory_space<vmem>>, vector<8x128xf32>, 
    %20 = tpu.memref_slice %arg3[%0, %c0_i32, %c0_i32] : memref<2x8x128xf32, #tpu.memory_space<any>> -> memref<1x8x128xf32, #tpu.memory_space<any>>
    %21 = tpu.memref_squeeze %20 : memref<1x8x128xf32, #tpu.memory_space<any>> -> memref<8x128xf32, #tpu.memory_space<any>>
    %22 = tpu.memref_slice %arg3[%1, %c0_i32, %c0_i32] : memref<2x8x128xf32, #tpu.memory_space<any>> -> memref<1x8x128xf32, #tpu.memory_space<any>>
    %23 = tpu.memref_squeeze %22 : memref<1x8x128xf32, #tpu.memory_space<any>> -> memref<8x128xf32, #tpu.memory_space<any>>
    tpu.wait_dma2 semaphore(%arg6 : memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>) src(%23 : memref<8x128xf32, #tpu.memory_space<any>>) dst(%21 : memref<8x128xf32, #tpu.memory_space<any>>) device_id(%c0_i32) core_id(%c0_i32)
    %24 = tpu.memref_slice %arg3[%1, %c0_i32, %c0_i32] : memref<2x8x128xf32, #tpu.memory_space<any>> -> memref<1x8x128xf32, #tpu.memory_space<any>>
    %25 = tpu.memref_squeeze %24 : memref<1x8x128xf32, #tpu.memory_space<any>> -> memref<8x128xf32, #tpu.memory_space<any>>
    %26 = tpu.memref_slice %arg3[%0, %c0_i32, %c0_i32] : memref<2x8x128xf32, #tpu.memory_space<any>> -> memref<1x8x128xf32, #tpu.memory_space<any>>
    %27 = tpu.memref_squeeze %26 : memref<1x8x128xf32, #tpu.memory_space<any>> -> memref<8x128xf32, #tpu.memory_space<any>>
    tpu.wait_dma2 semaphore(%arg5 : memref<!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>) src(%27 : memref<8x128xf32, #tpu.memory_space<any>>) dst(%25 : memref<8x128xf32, #tpu.memory_space<any>>) device_id(%5)
    return
  }
  func.func @transform_0(%arg0: i32) -> (i32, i32) {
    %c0_i32 = arith.constant 0 : i32
    return %c0_i32, %c0_i32 : i32, i32
  }
  func.func @transform_1(%arg0: i32) -> (i32, i32) {
    %c0_i32 = arith.constant 0 : i32
    return %c0_i32, %c0_i32 : i32, i32
  }
}