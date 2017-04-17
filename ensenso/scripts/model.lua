-----------------------------------------------------------------------
print '==> constructing model'

require 'nn'

local opt = {
	model = 'all-conv',	    -- all-conv, le-Net or maxpool
	loadSize = 96,
  nClasses = 3,           -- number of classes
	fineSize = 64,
	ndf = 64,               -- #  of filters in first conv layer
	gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end

opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.DoubleTensor')

if opt.gpu >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.manualSeed(opt.manualSeed)
  cutorch.setDevice(opt.gpu)                         
  idx       = cutorch.getDevice()
  use_cuda = true  
end

local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m.bias:fill(0)
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end

local nInputPlane = 1   -- The number of expected input planes in the image given into forward().
local nOutputPlane= 64  -- The number of output planes the convolution layer will produce.
local kW = 4            -- The kernel width of the convolution
local kH = 4            -- The kernel height of the convolution
local dW = 2            -- The step of the convolution in the width dimension. Default is 1.
local dH = 2            -- The step of the convolution in the height dimension. Default is 1.
local padW = 1          -- The additional zeros added per width to the input planes. Default is 0, a good number is (kW-1)/2.
local padH = 1          -- The additional zeros added per height to the input planes. Default is padW, a good number is (kH-1)/2.

local SpatialMaxPooling         = nn.SpatialMaxPooling
local SpatialConvolution        = nn.SpatialConvolution
local SpatialBatchNormalization = nn.SpatialBatchNormalization
local UpConvolution             = nn.SpatialFullConvolution
local Identity                  = nn.Identity
local Join                      = nn.JoinTable
local ReLU                      = nn.ReLU
local Dropout                   = nn.Dropout
local nbClasses                 = 3
local nc, nf, scale = 1, opt.fineSize, opt.fineSize
local trx = nn.LeakyReLU(0.2, true)

net = nn.Sequential()  

if opt.model == 'all-conv' then
  -- input is (nc) x 64 x 64
  net:add(SpatialConvolution(nc, nf, kW, kH, dW, dH, padW, padH))
  net:add(trx)
  -- state size: (nf) x 128 x 128
  net:add(SpatialConvolution(nf, nf * 2, kW, kH, dW, dH, padW, padH))
  net:add(SpatialBatchNormalization(nf * 2)):add(trx)
  -- state size: (nf*2) x 256 x 256
  net:add(SpatialConvolution(nf * 2, nf * 4, kW, kH, dW, dH, padW, padH))
  net:add(SpatialBatchNormalization(nf * 4)):add(trx)
  -- state size: (nf*4) x 512 x 512
  net:add(SpatialConvolution(nf * 4, nf * 8, kW, kH, dW, dH, padW, padH))
  net:add(SpatialBatchNormalization(nf * 8)):add(trx)
  
  -- state size: (nf*4) x 512 x 256
  net:add(SpatialConvolution(nf * 8, nf * 4, kW, kH, dW, dH, padW, padH))
  net:add(SpatialBatchNormalization(nf * 4)):add(trx)
  -- state size: (nf*4) x 256 x 128
  net:add(SpatialConvolution(nf * 4, nf * 2, kW, kH, dW, dH, padW, padH))
  net:add(SpatialBatchNormalization(nf * 2)):add(trx)
  
  -- state size: (nf*4) x 128 x 64
  net:add(SpatialConvolution(nf * 2, nf, kW, kH, dW, dH, padW, padH))
  net:add(trx)
  
  --[[
  -- state size: (nf) x 64 x 64
  net:add(SpatialConvolution(nf*8, 1, kW, kH))
  net:add(nn.Sigmoid())
  ]]

  net:add(nn.Reshape(nf))
  net:add(nn.Linear(nf, 3))

  --net:add(nn.View(1):setNumInputDims(3))

  -- Last stage : log probabilities
  net:add(nn.LogSoftMax())

elseif opt.model == 'leNet' then
  net:add(nn.SpatialConvolution(1, 6, 5, 5))  -- 1 input image channel, 6 output channels, 5x5 convolution kernel
  net:add(nn.ReLU())                          -- non-linearity 
  net:add(nn.SpatialMaxPooling(2,2,2,2))      -- A max-pooling operation that looks at 2x2 windows and finds the max.
  net:add(nn.SpatialConvolution(6, 16, 5, 5))
  net:add(nn.ReLU())                          -- non-linearity 
  net:add(nn.SpatialMaxPooling(2,2,2,2))
  net:add(nn.View(16*5*5))                    -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
  net:add(nn.Linear(16*5*5, 120))             -- fully connected layer (matrix multiplication between input and weights)
  net:add(nn.ReLU())                          -- non-linearity 
  net:add(nn.Linear(120, 84))
  net:add(nn.ReLU())                          -- non-linearity 
  net:add(nn.Linear(84, 3))                   -- 10 is the number of outputs of the network (in this case, 10 digits)
  net:add(nn.LogSoftMax()) 

elseif opt.model == 'maxpool' then
  -- hidden units, filter sizes (for ConvNet only):
  local nstates = {16,opt.fineSize}
  local filtsize = {5, 7}
  local poolsize = 4

  ----------------------------------------------------------------------
  print(sys.COLORS.red ..  '==> construct CNN')

  local CNN = nn.Sequential()

  -- stage 1: conv+max
  CNN:add(nn.SpatialConvolutionMM(nc, nstates[1], filtsize[1], filtsize[1]))
  CNN:add(nn.Threshold())
  CNN:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

  -- stage 2: conv+max
  CNN:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize[2], filtsize[2]))
  CNN:add(nn.Threshold())

  local classifier = nn.Sequential()
  -- stage 3: linear
  classifier:add(nn.Reshape(nstates[2]))
  classifier:add(nn.Linear(nstates[2], opt.nClasses))

  -- stage 4 : log probabilities
  classifier:add(nn.LogSoftMax())

  for _,layer in ipairs(CNN.modules) do
     if layer.bias then
        layer.bias:fill(.2)
        if i == #CNN.modules-1 then
           layer.bias:zero()
        end
     end
  end

  net:add(CNN)
  net:add(classifier)

else
   print('==> unknown model: ' .. opt.model)
   os.exit()
end

--all-net implementation
if opt.gpu >= 0 then
  local util = paths.dofile('util.lua')
  net:apply(weights_init)
  -- net = util.cudnn(net);    
  net:cuda(); 
else    
  net:apply(weights_init)
end

local loss = nn.ClassNLLCriterion()

print('model is : ', net)

--Exports
return {
	net = net,
	loss = loss,
  opt = opt
}