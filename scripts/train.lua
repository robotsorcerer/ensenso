-----------------------------------------------------------------------
print '==> training model'

require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
require 'sys'

-------------------------------------------------------------------------
--some global options
local opt = {	
	nThreads = 4,           -- #  of data loading threads to use
	batchSize = 9,
	loadSize = 96,
	fineSize = 256,
	save     = 'results/',  -- path to save logs of results
	niter = 25,             -- #  of iter at starting learning rate
	lr = 0.0002,            -- initial learning rate for adam
	lrdec = 1e-7,         -- learing Rate decay
	momentum = 0.5,         -- momentum term of adam
	gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
	ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
	display = 1,            -- display samples while training. 0 = false
	display_id = 10        -- display window id.
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end

----------------------------------------------------------------------
-- Model + Loss + Data:
local model = require 'scripts.model'
local Data  = require 'scripts.prepro'

local net = model.net
local criterion = model.loss

----------------------------------------------------------------------

-- This matrix records the current confusion across classes
local confusion = optim.ConfusionMatrix(classes)

-- Log results to files
if not paths.dirp(opt.save) then
  os.execute('mkdir  ' .. opt.save)
end

local trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
---------------------------------------------------------------------------

optimState = {
   learningRate = opt.lr,
   momentum = opt.momentum,
   weightDecay = opt.weightDecay,
   learningRateDecay = opt.lrdec
}

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> allocating minibatch memory')

local input = torch.Tensor(opt.batchSize, 1, opt.fineSize, opt.fineSize)
local label = torch.Tensor(opt.batchSize)
local err
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
----------------------------------------------------------------------------
if opt.gpu > 0 then
   require 'cunn'
   cutorch.setDevice(opt.gpu)
   input = input:cuda(); label = label:cuda()

   if pcall(require, 'cudnn') then
      require 'cudnn'
      cudnn.benchmark = true
      -- cudnn.convert(net, cudnn)
   end
   net:cuda();     criterion:cuda()
end

local params, gradParams = net:getParameters()

local trData = Data.trainData.data:clone()
local trLabels = Data.trainData.labels:clone()

local function getBatch(i)
   local batch = {}
   batch.trDataBatch = trData[{ {i, i+opt.batchSize-1}, {}, {}, {} }]
   batch.trDataLabelsBatch = trDataLabels[{ {i, i+opt.batchSize-1}, {}, {}, {} }]
   -- batch.teDataBatch = teData[{ {i, i+opt.batchSize-1}, {}, {}, {} }]
   return batch
end

if opt.display then disp = require 'display' end

-- print('trData size: ', trData:size())
-- sys.sleep('30')

-- create closure to evaluate f(X) and df/dX of discriminator
local fDx = function(x)
   gradParameters:zero()

   -- train with real
   data_tm:reset(); data_tm:resume()
   local real = data:getBatch()
   data_tm:stop()
   input:copy(real)
   label:fill(real_label)

   local output = netD:forward(input)
   local errD_real = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   netD:backward(input, df_do)

   -- train with fake
   if opt.noise == 'uniform' then -- regenerate random noise
       noise:uniform(-1, 1)
   elseif opt.noise == 'normal' then
       noise:normal(0, 1)
   end
   local fake = netG:forward(noise)
   input:copy(fake)
   label:fill(fake_label)

   local output = netD:forward(input)
   local errD_fake = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   netD:backward(input, df_do)

   errD = errD_real + errD_fake

   return errD, gradParametersD
end