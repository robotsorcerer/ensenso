-- Olalekan Ogunmolu
-- May - August 2016
-- Adapted frommy Kaggle UltraSound Nerve Sehgmentation Entry code
-- Refactorized Jan 11, 2016

require 'lfs'
require 'image'

local ImageLoader = torch.class('ImageLoader')


function ImageLoader:__init(opt, train_data, train_label, test_data, cv_data, cvLabel_dir)
  
  self.train_data = train_data or nil;    self.train_label = train_label or nil
  self.test_data = test_data or nil;      self.cv_data = cv_data or nil;  self.cv_labels = cvLabel_dir or nil

  self.files, self.labelFiles, self.testFiles, self.cvFiles, self.cvlabelFiles = {}, {}, {}, {}, {}
  self.ids,   self.idlabs,      self.idTests,  self.idCvs, self.idCVLabels   = {}, {}, {}, {}, {}

  self.img_dim = 64 --opt.fineSize
  -- read in all the filenames from the folder
  local n, nl, nt, ncv, ncvl = 1, 1, 1, 1, 1
  if( (self.train_data~=nil) or self.train_label ~=nil or self.test_data ~= nil or self.cv_data ~=nil) then     

    for file in paths.files(self.train_data, function(nm) return nm:find('.jpg') end) do
      local path_name = path.join(self.train_data, file) 
      table.insert(self.files, path_name)
      table.insert(self.ids, tostring(n)) -- just order them sequentially
      n=n+1
    end

    for file in paths.files(self.train_label, function(nm) return nm:find('.jpg') end) do
      local path_name = path.join(self.train_label, file) 
      table.insert(self.labelFiles, path_name)
      table.insert(self.idlabs, tostring(nl))
      nl = nl + 1
    end

    for file in paths.files(self.test_data, function(nm) return nm:find('.jpg') end) do
      local path_name = path.join(self.test_data, file) 
      table.insert(self.testFiles, path_name)
      table.insert(self.idTests, tostring(nt))
      nt = nt + 1
    end

    for file in paths.files(self.cv_data, function(nm) return nm:find('.jpg') end) do
      local path_name = path.join(self.cv_data, file) 
      table.insert(self.cvFiles, path_name)
      table.insert(self.idCvs, tostring(ncv))
      ncv = ncv + 1
    end

    for file in paths.files(self.cv_labels, function(nm) return nm:find('.jpg') end) do
      local path_name = path.join(self.cv_labels, file) 
      table.insert(self.cvlabelFiles, path_name)
      table.insert(self.idCVLabels, tostring(ncvl))
      ncvl = ncvl + 1
    end
  end

  self.Ntrain = #self.files
  self.NtrainLabels = #self.labelFiles
  self.Ntest = #self.testFiles
  self.Ncv   = #self.cvFiles
  self.Ncvl  = #self.cvlabelFiles

  --sort the files  
  -- table.sort(self.files,           function(a,b) return a < b end)  
  -- table.sort(self.labelFiles,      function(a,b) return a < b end)
  -- table.sort(self.testFiles,       function(a,b) return a < b end)
  -- table.sort(self.cvFiles,         function(a,b) return a < b end)
  -- table.sort(self.cvlabelFiles,    function(a,b) return a < b end)

  print('ImageLoader found ' .. self.Ntrain       .. ' images in directory ' .. self.train_data)
  print('ImageLoader found ' .. self.NtrainLabels .. ' images in directory ' .. self.train_label)
  print('ImageLoader found ' .. self.Ntest        .. ' images in directory ' .. self.test_data)
  print('ImageLoader found ' .. self.Ncv          .. ' images in directory ' .. self.cv_data)
  print('ImageLoader found ' .. self.Ncvl         .. ' images in directory ' .. self.cv_labels)

  self.iterator = 1
end

function file_exists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end

function ImageLoader:resetIterator()
  self.iterator = 1
end

function ImageLoader:getBatch(opt, x)
  local batchSize
  if x == nil then
    batchSize = opt.batchSize   -- how many images get returned at one time (to go through CNN)
  elseif x == 'all' then
    batchSize = self.Ntrain or self.NtrainLabels
  else
    batchSize = x
  end

  -- pick an index of the datapoint to load next
  local wrapped = false; 
  local infos, infosl = {}, {}
  local img_batch_raw = torch.FloatTensor(batchSize, 3, self.img_dim, self.img_dim)
  local img_batch_rawl = torch.FloatTensor(batchSize, 3, self.img_dim, self.img_dim)
  if( (self.train_data~=nil) or (self.train_label~=nil) ) then
    local max_index = self.Ntrain or self.NtrainLabels
    for i=1,batchSize do
      local ri = self.iterator
      local ri_next = ri + 1 -- increment iterator
      if ri_next > max_index then ri_next = 1; wrapped = true end -- wrap back around
      self.iterator = ri_next

      --[[
      because each pix is labeled by a user with a label tag,
      it is important to ensure the target we are comparing with 
      is same as the user and user tag for the training/cv data

      --So we force self.labelFiles to correspond to training image
      ]]
      local a = string.match(self.files[ri], "%d+_%d+")  --extract trailing #s from filename
      local b = string.match(self.labelFiles[ri], "%d+_%d+") --extract trailing #s from filename
      if (b ~= a) then
        -- print('substituting train id to match labels')
        self.labelFiles[ri] = string.gsub(self.labelFiles[ri], b, a)
      end

      -- print('self.labelFiles[ri]: ', self.labelFiles[ri])
      -- load the image
      local img = image.load(self.files[ri], 3, 'float')
      local imgl = image.load(self.labelFiles[ri], 3, 'float')
      img_batch_raw[i] = image.scale(img, self.img_dim, self.img_dim, 'bilinear')
      img_batch_rawl[i] = image.scale(imgl, self.img_dim, self.img_dim, 'bilinear')

      -- and record associated info as well
      local info_struct, info_structl = {}, {}
      info_struct.id = self.ids[ri]
      info_structl.id = self.idlabs[ri]
      info_struct.file_path = self.files[ri]
      info_structl.file_path = self.labelFiles[ri]
      table.insert(infos, info_struct); 
      table.insert(infosl, info_structl); 
    end
  end
  local data, datal = {}, {}
  data.images, datal.images = img_batch_raw, img_batch_rawl
  data.bounds = {it_pos_now = self.iterator, it_max = self.N, wrapped = wrapped}
  datal.bounds = {{it_pos_now = self.iterator, it_max = self.N, wrapped = wrapped}} 
  data.infos = infos
  datal.infos = infosl 

  return data, datal
end

function ImageLoader:getTestBatch(opt, x)
  local batchSize
  if x == nil then
    batchSize = opt.batchSize   -- how many images get returned at one time (to go through CNN)
  elseif x == 'all' then
    batchSize = self.Ntest
  else
    batchSize = x
  end

  -- pick an index of the datapoint to load next
  local wrapped = false
  local img_batch_raw = torch.FloatTensor(batchSize, 3, self.img_dim, self.img_dim)
    local max_index = self.Ntest
    local infos = {}
    for i=1,batchSize do
      local ri = self.iterator
      local ri_next = ri + 1 -- increment iterator
      if ri_next > max_index then ri_next = 1; wrapped = true end -- wrap back around
      self.iterator = ri_next

      -- load the image
      local img = image.load(self.testFiles[ri], 3, 'float')
      img_batch_raw[i] = image.scale(img, self.img_dim, self.img_dim, 'bilinear')

      -- and record associated info as well
      local info_struct = {}
      info_struct.ids = self.idTests[ri]
      info_struct.file_path = self.testFiles[ri]
      table.insert(infos, info_struct); 
    end
  local data = {}
  data.images = img_batch_raw
  data.bounds = {it_pos_now = self.iterator, it_max = self.N, wrapped = wrapped}  
  data.infos = infos

  return data
end

function ImageLoader:getCVBatch(opt, x)
  local batchSize
  if x == nil then
    batchSize = opt.batchSize   -- how many images get returned at one time (to go through CNN)
  elseif x == 'all' then
    batchSize = self.Ncv
  else
    batchSize = x
  end

  -- pick an index of the datapoint to load next
  local wrapped = false; 
  local infos, infosl = {}, {}
  local img_batch_raw = torch.FloatTensor(batchSize, 3, self.img_dim, self.img_dim)
  local img_batch_rawl = torch.FloatTensor(batchSize, 3, self.img_dim, self.img_dim)
  if( self.cv_data~=nil) then
    local max_index = self.Ncv
    for i=1,batchSize do
      local ri = self.iterator
      local ri_next = ri + 1 -- increment iterator
      if ri_next > max_index then ri_next = 1; wrapped = true end -- wrap back around
      self.iterator = ri_next

      --[[
      because each pix is labeled by a user with a label tag,
      it is important to ensure the target we are comparing with 
      is same as the user and user tag for the training/cv data

      --So we force self.labelFiles to correspond to training image
      ]]
      local a = string.match(self.cvFiles[ri], "%d+_%d+")  --extract trailing #s from filename
      local b = string.match(self.cvlabelFiles[ri], "%d+_%d+") --these will be the labels

      if (b ~= a) then
        -- print('substituting train id to match labels')
        self.cvlabelFiles[ri] = string.gsub(self.cvlabelFiles[ri], b, a)
      end
      -- print('self.labelFiles[ri]: ', self.labelFiles[ri])
      -- load the image
      local img = image.load(self.cvFiles[ri], 3, 'float')
      local imgl = image.load(self.cvlabelFiles[ri], 3, 'float')

      -- print('cv trains: ', self.cvFiles[ri])
      -- print('cv labels: ', self.cvlabelFiles[ri])
      img_batch_raw[i] = image.scale(img, self.img_dim, self.img_dim, 'bilinear')
      img_batch_rawl[i] = image.scale(imgl, self.img_dim, self.img_dim, 'bilinear')

      -- and record associated info as well
      local info_struct, info_structl = {}, {}
      info_struct.id = self.ids[ri]
      info_structl.id = self.idlabs[ri]
      info_struct.file_path = self.files[ri]
      info_structl.file_path = self.labelFiles[ri]
      table.insert(infos, info_struct); 
      table.insert(infosl, info_structl); 
    end
  end
  local data, datal = {}, {}
  data.images, datal.images = img_batch_raw, img_batch_rawl
  data.bounds = {it_pos_now = self.iterator, it_max = self.N, wrapped = wrapped}
  datal.bounds = {{it_pos_now = self.iterator, it_max = self.N, wrapped = wrapped}} 
  data.infos = infos
  datal.infos = infosl 

  return data, datal
end