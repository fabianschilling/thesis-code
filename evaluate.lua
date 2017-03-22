function evaluate(testData)

  collectgarbage()

  -- put model to evaluate mode (for stuff like dropout)
  model:evaluate()

  local numTestBatches = 100
  local sizeTest = testData.inputs:size()[1]
  local testBatchSize = math.floor(sizeTest / numTestBatches)
  local batchStart = 1
  local batchEnd = batchStart + testBatchSize - 1

  for testBatch = 1, numTestBatches do

    if args.progress then xlua.progress(testBatch, numTestBatches) end

    if testBatch == numTestBatches then
      batchEnd = sizeTest
    end

    local preds = model:forward(testData.inputs[{{batchStart, batchEnd}}]:cuda())
    confusion:batchAdd(preds, testData.labels[{{batchStart, batchEnd}}])

    batchStart = batchStart + testBatchSize
    batchEnd = batchEnd + testBatchSize

  end

  -- confusion matrix
  confusion:updateValids()

  if args.confusion then print(confusion) end
  local accuracy = confusion.totalValid * 100

  confusion:zero()

  collectgarbage()

  return accuracy

end
