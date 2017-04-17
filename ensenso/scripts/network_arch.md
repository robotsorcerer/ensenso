model is : 	nn.Sequential {
  [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> (9) -> (10) -> (11) -> (12) -> (13) -> (14) -> (15) -> (16) -> output]
  (1): nn.SpatialConvolution(1 -> 64, 4x4, 2,2, 1,1)
  (2): nn.LeakyReLU(0.2)
  (3): nn.SpatialConvolution(64 -> 128, 4x4, 2,2, 1,1)
  (4): nn.SpatialBatchNormalization (4D) (128)
  (5): nn.LeakyReLU(0.2)
  (6): nn.SpatialConvolution(128 -> 256, 4x4, 2,2, 1,1)
  (7): nn.SpatialBatchNormalization (4D) (256)
  (8): nn.LeakyReLU(0.2)
  (9): nn.SpatialConvolution(256 -> 512, 4x4, 2,2, 1,1)
  (10): nn.SpatialBatchNormalization (4D) (512)
  (11): nn.LeakyReLU(0.2)
  (12): nn.SpatialConvolution(512 -> 3, 4x4)
  (13): nn.Sigmoid
  (14): nn.View(3)
  (15): nn.Linear(3 -> 3)
  (16): nn.LogSoftMax
}


## Running training examples at 256 X 256 image size drains my memory. Downsampling to 64 X 64 sized images