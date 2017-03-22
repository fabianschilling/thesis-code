function train(trainData)

  -- put model in training mode (for things like dropout)
  model:training()

  -- Get dataset size
  dataSize = trainData.inputs:size()[1]

  -- shuffle training data
  local shuffle = torch.randperm(dataSize)

  -- local vars
  local epochDataLoss = 0
  local epochRegLoss = 0

  -- Calculate number of batches
  local numBatches = math.floor(dataSize / args.batchsize)

  -- Train on all batches
  for batch = 1, numBatches do

    if args.progress then xlua.progress(batch, numBatches) end

    local timeStart = torch.tic()

    local inputs = torch.CudaTensor(args.batchsize, args.channels, args.height, args.width)
    local labels = torch.CudaTensor(args.batchsize)

    local batchStart = (batch * args.batchsize) - args.batchsize + 1
    local batchEnd = batchStart + args.batchsize - 1

    local index = 1
    for i = batchStart, batchEnd do
      inputs[index] = trainData.inputs[shuffle[i]]
      labels[index] = trainData.labels[shuffle[i]]
      index = index + 1
    end

    -- closure to evaluate batchLoss and df
    local feval = function(x)

      -- Copy new parameters (if any)
      if x ~= parameters then parameters:copy(x) end

      -- Reset gradients to zero (very important since they get accumulated)
      gradParameters:zero()

      -- Forward and compute batch data loss (accumulate epoch loss)
      local outputs = model:forward(inputs)
      confusion:batchAdd(outputs, labels)
      local batchDataLoss = criterion:forward(outputs, labels)
      epochDataLoss = epochDataLoss + batchDataLoss

      if args.logging then
        logOutput(model)
      end

      -- Backward
      local df = criterion:backward(outputs, labels)
      model:backward(inputs, df)

      -- Compute batch regularization loss
      local batchRegLoss = 0

      -- L1 weight decay
      if args.wDecayL1 > 0 then
        batchRegLoss = batchRegLoss + args.wDecayL1 * torch.norm(parameters, 1)
        gradParameters:add(torch.sign(parameters):mul(args.wDecayL1))
      end

      -- L2 weight decay
      if args.wDecayL2 > 0 then
        batchRegLoss = batchRegLoss + args.wDecayL2 * torch.norm(parameters, 2)^2 / 2
        gradParameters:add(parameters:clone():mul(args.wDecayL2))
      end

      -- Accumulate epoch regularization loss
      epochRegLoss = epochRegLoss + batchRegLoss

      -- Timing
      local time = torch.toc(timeStart)

      -- Log layer activations and gradients
      if args.logging then
        logGrads(model)
        logBatch(batch, batchDataLoss, batchRegLoss, optimConfig.learningRate, time)
      end

      local batchLoss = batchDataLoss + batchRegLoss
      return batchLoss, gradParameters
    end

    -- Update parameters using optimizer
    optim[string.lower(args.optimizer)](feval, parameters, optimConfig, optimState)

    if args.logging then
      logWeights(model)
    end

  end

  -- confustion matrix
  confusion:updateValids()

  if args.confusion then print(confusion) end

  local trainAccuracy = confusion.totalValid * 100

  confusion:zero()

  collectgarbage()

  -- train error
  epochDataLoss = epochDataLoss / numBatches
  epochRegLoss = epochRegLoss / numBatches
  return epochDataLoss, epochRegLoss, trainAccuracy
end

