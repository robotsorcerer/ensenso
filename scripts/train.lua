-----------------------------------------------------------------------
print '==> training model'

require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
require 'sys'

-------------------------------------------------------------------------
--some global options
local opt = {	
	nThreads = 4,           -- #  of data loading threads to use
	batchSize = 3,
	loadSize = 96,
	fineSize = 256,
	save     = 'results/',  -- path to save logs of results
	name = 'manikin',
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
   batch.trLabelsBatch = trLabels[{ {i, i+opt.batchSize-1} }]
   return batch
end

function ship2gpu(x)
	if opt.gpu > 0 then
		x:cuda()
	else
		x:double()
	end
end

ship2gpu(trData)
ship2gpu(teData)

if opt.display then disp = require 'display' end

--convenience path where model checkpoints are saved
if not paths.dirp('checkpoints/') then os.execute('mkdir checkpoints/') end
-- print('trData size: ', trData:size())
-- sys.sleep('30')

--train
for epoch = 1, opt.niter do
   epoch_tm:reset()
   local counter = 0
   -- local cv_loss   
   local output

   for i = 1, math.min(trData:size(1), opt.ntrain), opt.batchSize do

   	  collectgarbage()

   	  -- batch fits?
   	  if (i + opt.batchSize - 1) > trData:size(1) then
   	     break
   	  end

      tm:reset()
      local temp, labels;

      -- cv_loss =  cross_val(i)

      --closure for f(x) and df/dx
      local fx = function(x)
         net:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

         gradParams:zero()

         data_tm:reset(); data_tm:resume()

         temp = (getBatch(i)).trDataBatch
         labels = (getBatch(i)).trLabelsBatch

         data_tm:stop()

         input:copy(temp)
         label:copy(labels)


         print('output size: ', net:forward(input))

         output = net:forward(input)
         print('output size: ', output:size())
         err = criterion:forward(output, label)
         -- estimate df/do
         local df_do = criterion:backward(output, label)
         net:backward(input, df_do)
         net:updateGradInput({input}, {df_do})

         -- update confusion
         for i = 1,opt.batchSize do
            confusion:add(output[i],label[i])
         end

         return err, gradParams
      end

      -- update network: max log
      optim.adam(fx, params, optimState)

      --display
      counter = counter + 1
      if counter % 10 == 0 and opt.display then         
         local pred = net:forward(input)
         disp.image(label, {win=opt.display_id, title='face labels'})
         disp.image(output, {win=opt.display_id*10, title='face preds'})
      end

      -- print confusion matrix
      print(confusion)

      loss_history[i] = err
      -- log erroes
      if((i-1)/opt.batchSize) %1 == 0 then
         print(('Epoch: [%d][%3d / %3d] | Time: %.3f | DataTime: %.3f '
            .. ' | Err: %.8f'):format(
            epoch, ((i-1) / input:size(1)), 
            math.floor(math.min(input:size(1), opt.niter))/opt.batchSize,
            tm:time().real, data_tm:time().real, err and err or -1))
      end
   end
   params, gradParams = nil, nil
     util.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_net.t7', net, opt.gpu)
   -- end
   params, gradParams = net:getParameters() --reflatted trhe params
   print(('Epoch end %d/ %d \t Time Taken: %.3f'):format(epoch, opt.niter, epoch_tm:time().real))
end