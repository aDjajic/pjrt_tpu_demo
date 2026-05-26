#loc1 = loc("a")
module @jit__reduce_sum attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8x128xf32> loc("a")) -> (tensor<128xf32> {jax.result_info = "result"}) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32> loc(#loc4)
    %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0] : (tensor<8x128xf32>, tensor<f32>) -> tensor<128xf32> loc(#loc4)
    return %0 : tensor<128xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc2 = loc("/home/aleksandar/pjrt_tpu_demo/cfg/pallas_example_dynamic_for.py":76:15 to :38)
#loc3 = loc("<module>"(#loc2))
#loc4 = loc("jit(_reduce_sum)/reduce_sum"(#loc3))
