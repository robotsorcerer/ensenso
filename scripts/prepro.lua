--[[--------------------------------------------------------------------
-- Load the Face Detector training data, and pre-process it to facilitate learning.
--
-- It's a good idea to run this script with the interactive mode:
-- $ torch -i 1_data.lua
-- this will give you a Torch interpreter at the end, that you
-- can use to analyze/visualize the data you've just loaded.


To load this script, from the root of the project, 
  do IMAGES_ROOT=data th scripts/prepro.lua
--
-- Lekan Ogunmolu
-- Wed Jan 11 16 12:38PM CST
----------------------------------------------------------------------
]]

require 'torch'   -- torch
require 'image'   -- to visualize the dataset
require 'nnx'      -- provides a normalization operator

local opt = opt or {
   visualize = true,
   size = 'small',
   patches='all',
}
----------------------------------------------------------------------

if opt.patches ~= 'all' then
   opt.patches = math.floor(opt.patches/3)
end

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> loading dataset')

allfiles = {}
face_files = {}
bg_files = {}
fake_files = {}

--load all the images, crop'em and scale them to 256x256
-- to fit within memory
local data = os.getenv('IMAGES_ROOT') .. '/images/'
for file in paths.files(data, function(nm) return nm:find('face*') end) do
  table.insert(allfiles, file)
  local f2 = paths.concat(data, file)
  local im = image.load(f2)
  local x1, y1 = 190, 850
  local x2, y2 = 1140, 72
  -- local cropped = image.crop(im, x1, y1, x2, y2)
  local scaled = image.scale(im, 256, 256)
  image.save(f2, scaled)
end

for file in paths.files(data, function(nm) return nm:find('face_positive*') end) do
  table.insert(face_files, file)
end

for file in paths.files(data, function(nm) return nm:find('face_fake_*') end) do
  table.insert(fake_files, file)
end

for file in paths.files(data, function(nm) return nm:find('face_bg*') end) do
  table.insert(bg_files, file)
end

print(#allfiles, '\t', #face_files, '\t', #fake_files, '\t', #bg_files)

--get total number of files in the training images directory excluding symbolic files 
-- local trainTotal = tonumber(os.execute('ls ../data/train/images/ -1 | grep -v ^l | wc -l'))
-- local posFacesNum = os.execute('ls ../data/train/images/face_positive* -1 | wc -l')
-- local bgFacesNum = os.execute('ls ../data/train/images/face_bg* -1 | grep -v ^l | wc -l')
-- local fakeFacesNum = os.execute('ls ../data/train/images/face_fake* -1 | grep -v ^l | wc -l')

-- preallocate images and their labels
local imagesAll, labelsAll    = torch.Tensor(#allfiles,1,256,256), torch.Tensor(#allfiles)
local posFaces, labelsPos     = torch.Tensor(#face_files,1,256,256), torch.Tensor(#face_files)
local bgFaces, labelsBg       = torch.Tensor(#bg_files,1,256,256), torch.Tensor(#bg_files)
local fakeFaces, labelsFake   = torch.Tensor(#fake_files,1,256,256), torch.Tensor(#fake_files)

-- classes: GLOBAL var!
classes = {'face','background', 'fake'}

local function reset_iter(iter)
  return 0
end

-- load positive faces:
local iter = 0
for f in paths.files(data, function(nm) return nm:find('face_positive*') end) do--=0,#face_files do
  local temp = paths.concat(data, f)
  posFaces[iter+1] = image.load(temp) 
  labelsPos[iter+1] = 1 -- 1 = face
end

-- load backgrounds:
iter = reset_iter(iter)
for f in paths.files(data, function(nm) return nm:find('face_bg*') end) do
  local temp = paths.concat(data, f)
  bgFaces[iter+1] = image.load(temp) 
  labelsBg[iter+1] = 2        -- 2 = background
end

-- load fake faces:
iter = reset_iter(iter)
for f in paths.files(data, function(nm) return nm:find('face_fake_*') end) do
  local temp = paths.concat(data, f)
  fakeFaces[iter+1] = image.load(temp) 
  labelsFake[iter+1] = 3         -- 3 = fake
end

--concatenate all faces and labels along the 1st dim
imagesAll = torch.cat(posFaces, bgFaces, fakeFaces, 1)
labelsAll      = torch.cat(labelsPos, labelsBg, labelsFake, 1)

-- shuffle dataset: get shuffled indices in this variable:
local labelsShuffle = torch.randperm((#labelsAll)[1])

local portionTrain = 0.8            -- 80% is train data, rest is test data
local trsize = torch.floor(labelsShuffle:size(1)*portionTrain)
local tesize = labelsShuffle:size(1) - trsize

-- create train set:
trainData = {
   data = torch.Tensor(trsize, 1, 256, 256),
   labels = torch.Tensor(trsize),
   size = function() return trsize end
}
--create test set:
testData = {
      data = torch.Tensor(tesize, 1, 256, 256),
      labels = torch.Tensor(tesize),
      size = function() return tesize end
   }

for i=1,trsize do
   trainData.data[i] = imagesAll[labelsShuffle[i]][1]:clone()
   trainData.labels[i] = labelsAll[labelsShuffle[i]]
end
for i=trsize+1,tesize+trsize do
   testData.data[i-trsize] = imagesAll[labelsShuffle[i]][1]:clone()
   testData.labels[i-trsize] = labelsAll[labelsShuffle[i]]
end

-- remove from memory temp image files:
-- imagesAll = nil
-- labelsAll = nil


----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> preprocessing data')

trainData.data = trainData.data:float()
testData.data = testData.data:float()

-- Name channels for convenience
local channels = {'y'}--,'u','v'}

-- Normalize each channel, and store mean/std
-- per channel. These values are important, as they are part of
-- the trainable parameters. At test time, test data will be normalized
-- using these values.
print(sys.COLORS.red ..  '==> preprocessing data: normalize each feature (channel) globally')
local mean = {}
local std = {}
for i,channel in ipairs(channels) do
   -- normalize each channel globally:
   mean[i] = trainData.data[{ {},i,{},{} }]:mean()
   std[i] = trainData.data[{ {},i,{},{} }]:std()
   trainData.data[{ {},i,{},{} }]:add(-mean[i])
   trainData.data[{ {},i,{},{} }]:div(std[i])
end

-- Normalize test data, using the training means/stds
for i,channel in ipairs(channels) do
   -- normalize each channel globally:
   testData.data[{ {},i,{},{} }]:add(-mean[i])
   testData.data[{ {},i,{},{} }]:div(std[i])
end

-- Local contrast normalization is needed in the face dataset as the dataset is already in this form:
print(sys.COLORS.red ..  '==> preprocessing data: normalize all three channels locally')

-- Define the normalization neighborhood:
local neighborhood = image.gaussian1D(5) -- 5 for face detector training

-- Define our local normalization operator (It is an actual nn module, 
-- which could be inserted into a trainable model):
local normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()

-- Normalize all channels locally:
for c in ipairs(channels) do
   for i = 1,trainData:size() do
      trainData.data[{ i,{c},{},{} }] = normalization:forward(trainData.data[{ i,{c},{},{} }]:float())
   end
   for i = 1,testData:size() do
      testData.data[{ i,{c},{},{} }] = normalization:forward(testData.data[{ i,{c},{},{} }]:float())
   end
end

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> verify statistics')

-- It's always good practice to verify that data is properly
-- normalized.

for i,channel in ipairs(channels) do
   local trainMean = trainData.data[{ {},i }]:mean()
   local trainStd = trainData.data[{ {},i }]:std()

   local testMean = testData.data[{ {},i }]:mean()
   local testStd = testData.data[{ {},i }]:std()

   print('training data, '..channel..'-channel, mean: ' .. trainMean)
   print('training data, '..channel..'-channel, standard deviation: ' .. trainStd)

   print('test data, '..channel..'-channel, mean: ' .. testMean)
   print('test data, '..channel..'-channel, standard deviation: ' .. testStd)
end

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> visualizing data')

-- Visualization is quite easy, using image.display(). Check out:
-- help(image.display), for more info about options.

if opt.visualize then
   local first50Samples_y = trainData.data[{ {1,50},1 }]
   image.display{image=first50Samples_y, nrow=16, legend='Some training examples: Y channel'}
   local first50Samples_y = testData.data[{ {1,50},1 }]
   image.display{image=first50Samples_y, nrow=16, legend='Some testing examples: Y channel'}
end

-- Exports
return {
   trainData = trainData,
   testData = testData,
   mean = mean,
   std = std,
   classes = classes
}

