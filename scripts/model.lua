-----------------------------------------------------------------------
print '==> constructing model'

require 'nn'

local opt = {
	model = 'all-conv',	
	loadSize = 96,
	fineSize = 256,
	ndf = 256,               -- #  of filters in first conv layer
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
local nOutputPlane= 256  -- The number of output planes the convolution layer will produce.
local kW = 4            -- The kernel width of the convolution
local kH = 4            -- The kernel height of the convolution
local dW = 1            -- The step of the convolution in the width dimension. Default is 1.
local dH = 1            -- The step of the convolution in the height dimension. Default is 1.
local padW = 0          -- The additional zeros added per width to the input planes. Default is 0, a good number is (kW-1)/2.
local padH = 0          -- The additional zeros added per height to the input planes. Default is padW, a good number is (kH-1)/2.

local SpatialMaxPooling         = nn.SpatialMaxPooling
local SpatialConvolution        = nn.SpatialConvolution
local SpatialBatchNormalization = nn.SpatialBatchNormalization
local UpConvolution             = nn.SpatialFullConvolution
local Identity                  = nn.Identity
local Join                      = nn.JoinTable
local ReLU                      = nn.ReLU
local Dropout                   = nn.Dropout
local nbClasses                 = 3
local nc, ndf, scale = 1, opt.fineSize, opt.fineSize
local trx = nn.ReLU(true)

if opt.model == 'all-conv' then

  net = nn.Sequential()  
  -- input is (nc) x 256 x 256
  net:add(SpatialConvolution(nc, ndf, 4, 4, dW, dH, padW, padH))
  net:add(trx)
  -- state size: (ndf) x 128 x 128
  net:add(SpatialConvolution(ndf, ndf * 2, 4, 4, dW, dH, padW, padH))
  net:add(SpatialBatchNormalization(ndf * 2)):add(nn.ReLU(true))
  -- state size: (ndf*2) x 64 x 64
  net:add(SpatialConvolution(ndf * 2, ndf * 4, 4, 4, dW, dH, padW, padH))
  net:add(SpatialBatchNormalization(ndf * 4)):add(nn.ReLU(true))
  -- state size: (ndf*4) x 32 x 32
  net:add(SpatialConvolution(ndf * 4, ndf * 8, 4, 4, dW, dH, padW, padH))
  net:add(SpatialBatchNormalization(ndf * 8)):add(nn.ReLU(true))
  -- state size: (ndf*8) x 16 x 16
  net:add(SpatialConvolution(ndf * 8, ndf*4, 4, 4, dW, dH, padW, padH))
  net:add(SpatialBatchNormalization(ndf*4)):add(nn.ReLU(true))
  -- state size: (ndf*4) x 32 x 32
  net:add(SpatialConvolution(ndf * 4, ndf*2, 4, 4, dW, dH, padW, padH))
  net:add(SpatialBatchNormalization(ndf*2)):add(nn.ReLU(true))
  -- state size: (ndf*2) x 64 x 64
  net:add(SpatialConvolution(ndf * 2, ndf, 4, 4, dW, dH, padW, padH))
  net:add(SpatialBatchNormalization(ndf)):add(nn.ReLU(true))

  net:add(SpatialConvolution(ndf, ndf, 4, 4, dW, dH, padW, padH))
  net:add(nn.Sigmoid())

  net:add(nn.SpatialUpSamplingNearest(scale))

  --all-net implementation
  if opt.gpu >= 0 then
    local util = paths.dofile('util.lua')
    net:apply(weights_init)
    -- net = util.cudnn(net);    
    net:cuda(); 
  end
else
   print('==> unknown model: ' .. opt.model)
   os.exit()
end

local loss = nn.ClassNLLCriterion()

print('model is : ', net)

--return packages
return {
	net = net,
	loss = loss
}