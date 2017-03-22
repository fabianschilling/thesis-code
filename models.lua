function loadModel(name)

  if args.verbose then
    print('Model: ' .. name)
    print('Activation: ' .. args.activation)
    print('Batchnorm: ' .. tostring(args.batchnorm))
    if args.dropout then
      print('Dropout: ' .. tostring(args.dropout) .. ' (p = ' .. args.dropoutProb .. ')')
    end
  end

  if name == 'MLP' then
    return getMLP()
  elseif name == 'CNN' then
    return getCNN()
  elseif name == 'VGG' then
    return getVGG()
  else
    error('Unrecognized model: ' .. name)
  end
end

function getMLP()

  local hidden = 100

  local model = nn.Sequential()

  model:add(nn.Reshape(args.channels * args.height * args.width))

  model:add(nn.Linear(args.channels * args.height * args.width, hidden))
  if args.batchnorm then model:add(nn.BatchNormalization(hidden)) end
  model:add(nn[args.activation]())
  if args.dropout then model:add(nn.Dropout(args.dropoutProb)) end

  model:add(nn.Linear(hidden, hidden))
  if args.batchnorm then model:add(nn.BatchNormalization(hidden)) end
  model:add(nn[args.activation]())
  if args.dropout then model:add(nn.Dropout(args.dropoutProb)) end

  model:add(nn.Linear(hidden, hidden))
  if args.batchnorm then model:add(nn.BatchNormalization(hidden)) end
  model:add(nn[args.activation]())
  if args.dropout then model:add(nn.Dropout(args.dropoutProb)) end

  return model:add(nn.Linear(hidden, #(args.classes)))
end

function getCNN()

  local model = nn.Sequential()

  -- conv1
  model:add(nn.SpatialConvolution(args.channels, 32, 3, 3, 1, 1, 1, 1))
  if args.batchnorm then model:add(nn.SpatialBatchNormalization(32)) end
  model:add(nn[args.activation]())
  if args.dropout then model:add(nn.Dropout(args.dropoutProb)) end
  model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
  --output 16x16

  -- conv2
  model:add(nn.SpatialConvolution(32, 64, 3, 3, 1, 1, 1, 1))
  if args.batchnorm then model:add(nn.SpatialBatchNormalization(64)) end
  model:add(nn[args.activation]())
  if args.dropout then model:add(nn.Dropout(args.dropoutProb)) end
  model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
  --output 8x8

  -- conv3
  model:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1))
  if args.batchnorm then model:add(nn.SpatialBatchNormalization(128)) end
  model:add(nn[args.activation]())
  if args.dropout then model:add(nn.Dropout(args.dropoutProb)) end
  model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
  --output 4x4

  -- conv4
  model:add(nn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1))
  if args.batchnorm then model:add(nn.SpatialBatchNormalization(256)) end
  model:add(nn[args.activation]())
  if args.dropout then model:add(nn.Dropout(args.dropoutProb)) end
  model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
  --output 2x2

  -- conv5
  model:add(nn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1))
  if args.batchnorm then model:add(nn.SpatialBatchNormalization(512)) end
  model:add(nn[args.activation]())
  if args.dropout then model:add(nn.Dropout(args.dropoutProb)) end
  model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
  --output 1x1

  -- reshape
  model:add(nn.View(512))

  -- fc1
  model:add(nn.Linear(512, 512))
  if args.batchnorm then model:add(nn.BatchNormalization(512)) end
  model:add(nn[args.activation]())
  if args.dropout then model:add(nn.Dropout(args.dropoutProb)) end

  -- fc2
  model:add(nn.Linear(512, #(args.classes)))

  return model
end

function getVGG()

  local model = nn.Sequential()

  local function ConvBNReLU(nInputPlane, nOutputPlane)
    model:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3, 3, 1, 1, 1, 1))
    if args.batchnorm then model:add(nn.SpatialBatchNormalization(nOutputPlane)) end
    model:add(nn[args.activation]())
    return model
  end

  -- conv1
  ConvBNReLU(args.channels, 64)
  if args.dropout then model:add(nn.Dropout(args.dropoutProb)) end
  ConvBNReLU(64, 64)
  model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

  -- conv2
  ConvBNReLU(64, 128)
  if args.dropout then model:add(nn.Dropout(args.dropoutProb)) end
  ConvBNReLU(128, 128)
  model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

  -- conv3
  ConvBNReLU(128, 256)
  if args.dropout then model:add(nn.Dropout(args.dropoutProb)) end
  ConvBNReLU(256, 256)
  model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

  -- conv4
  ConvBNReLU(256, 512)
  if args.dropout then model:add(nn.Dropout(args.dropoutProb)) end
  ConvBNReLU(512, 512)
  model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

  -- conv5
  ConvBNReLU(512, 512)
  if args.dropout then model:add(nn.Dropout(args.dropoutProb)) end
  ConvBNReLU(512, 512)
  model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

  model:add(nn.View(512))

  -- fc1
  model:add(nn.Linear(512, 512))
  if args.batchnorm then model:add(nn.BatchNormalization(512)) end
  model:add(nn[args.activation]())
  if args.dropout then model:add(nn.Dropout(args.dropoutProb)) end

  -- fc2
  model:add(nn.Linear(512, #(args.classes)))

  return model

end
