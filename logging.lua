function createLoggers()

  epochLogger = optim.Logger(paths.concat(args.path, 'epoch.log'))
  modelLogger = optim.Logger(paths.concat(args.path, 'model.log'))
  configLogger = optim.Logger(paths.concat(args.path, 'config.log'))
  batchLogger = optim.Logger(paths.concat(args.path, 'batch.log'))
  outputLogger = optim.Logger(paths.concat(args.path, 'output.log'))
  gradLogger = optim.Logger(paths.concat(args.path, 'grad.log'))
  weightLogger = optim.Logger(paths.concat(args.path, 'weight.log'))
  testLogger = optim.Logger(paths.concat(args.path, 'test.log'))

end

function logConfig(config)

  local inspect = require('inspect')

  configLog = {}

  configLog.config = inspect(config)

  configLogger:add(configLog)

end

function logModel(model)

  local modelLog = {}

  modelLog.model = tostring(model)

  modelLogger:add(modelLog)
end

function logBatch(batch, batchDataLoss, batchRegLoss, learningRate, time)

  local batchLog = {}

  batchLog.batch = batch
  batchLog.loss = batchDataLoss + batchRegLoss
  batchLog.dataLoss = batchDataLoss
  batchLog.regLoss = batchRegLoss
  batchLog.lr = learningRate
  batchLog.time = time

  batchLogger:add(batchLog)
end

function logEpoch(epoch, epochDataLoss, epochRegLoss, accuracy, trainAccuracy, learningRate, time)

  local epochLog = {}

  epochLog.epoch = epoch
  epochLog.loss = epochDataLoss + epochRegLoss
  epochLog.dataLoss = epochDataLoss
  epochLog.regLoss = epochRegLoss
  epochLog.accuracy = accuracy
  epochLog.trainAccuracy = trainAccuracy
  epochLog.lr = learningRate
  epochLog.time = time

  epochLogger:add(epochLog)
end

function getLayerName(layer)
  return string.lower(stringx.split(torch.type(layer), '.')[2])
end

function logOutput(model)

  local outputLog = {}

  for i = 1, #model.modules do

    local layer = model.modules[i]
    local name = getLayerName(layer)

    outputLog[string.format('%02d/%s/mean', i, name)] = layer.output:mean()
    outputLog[string.format('%02d/%s/std', i, name)] = layer.output:std()
    outputLog[string.format('%02d/%s/min', i, name)] = layer.output:min()
    outputLog[string.format('%02d/%s/max', i, name)] = layer.output:max()
  end

  outputLogger:add(outputLog)

end

function logGrads(model)

  local gradLog = {}

  for i = 1, #model.modules do

    local layer = model.modules[i]
    local name = getLayerName(layer)

    if layer.gradWeight ~= nil then
      gradLog[string.format('%02d/%s/weight/mean', i, name)] = layer.gradWeight:mean()
      gradLog[string.format('%02d/%s/weight/std', i, name)] = layer.gradWeight:std()
      gradLog[string.format('%02d/%s/weight/min', i, name)] = layer.gradWeight:min()
      gradLog[string.format('%02d/%s/weight/max', i, name)] = layer.gradWeight:max()
    end

    if layer.gradBias ~= nil then
      gradLog[string.format('%02d/%s/bias/mean', i, name)] = layer.gradBias:mean()
      gradLog[string.format('%02d/%s/bias/std', i, name)] = layer.gradBias:std()
      gradLog[string.format('%02d/%s/bias/min', i, name)] = layer.gradBias:min()
      gradLog[string.format('%02d/%s/bias/max', i, name)] = layer.gradBias:max()
    end
  end

  gradLogger:add(gradLog)
end

function logWeights(model)

  local weightLog = {}

  for i = 1, #model.modules do

    local layer = model.modules[i]
    local name = getLayerName(layer)

    if layer.weight ~= nil then
      weightLog[string.format('%02d/%s/weight/mean', i, name)] = layer.weight:mean()
      weightLog[string.format('%02d/%s/weight/std', i, name)] = layer.weight:std()
      weightLog[string.format('%02d/%s/weight/min', i, name)] = layer.weight:min()
      weightLog[string.format('%02d/%s/weight/max', i, name)] = layer.weight:max()
    end

    if layer.bias ~= nil then
      weightLog[string.format('%02d/%s/bias/mean', i, name)] = layer.bias:mean()
      weightLog[string.format('%02d/%s/bias/std', i, name)] = layer.bias:std()
      weightLog[string.format('%02d/%s/bias/min', i, name)] = layer.bias:min()
      weightLog[string.format('%02d/%s/bias/max', i, name)] = layer.bias:max()
    end
  end

  weightLogger:add(weightLog)
end

function logTest(epoch, accuracy, time)

  local testLog = {}

  testLog.epoch = epoch
  testLog.accuracy = accuracy
  testLog.time = time

  testLogger:add(testLog)
end
