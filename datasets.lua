function loadDataset(name, validRatio)

  local dataset = torch.load('/home/fabian/datasets/' .. string.lower(name) .. '.t7')

  args.classes = dataset.classes
  args.channels = dataset.channels
  args.width = dataset.width
  args.height = dataset.height

  if validRatio > 0.0 then
    dataset = splitTrain(dataset, validRatio)
  else
    if args.verbose then print('No validation set!') end
  end

  local trainSize = dataset.train.labels:size()[1]
  local validSize = dataset.valid.labels:size()[1]
  local testSize = dataset.test.labels:size()[1]
  if args.verbose then print('Dataset: ' .. name .. ' (train: ' .. trainSize .. ', valid: ' .. validSize .. ', test: ' .. testSize .. ')') end

  return dataset
end

function splitTrain(dataset, validRatio) 

  local size = dataset.train.labels:size()[1]

  local trainStart = 1
  local trainEnd = math.floor(size * (1 - validRatio))

  local validStart = trainEnd + 1
  local validEnd = size

  dataset.valid = {}
  dataset.valid.inputs = dataset.train.inputs[{{validStart, validEnd}}]
  dataset.valid.labels = dataset.train.labels[{{validStart, validEnd}}]
  dataset.train.inputs = dataset.train.inputs[{{trainStart, trainEnd}}]
  dataset.train.labels = dataset.train.labels[{{trainStart, trainEnd}}]

  return dataset

end

function countExamples(labels, classes)

  counts = torch.zeros(#classes)

  for i = 1, labels:size()[1] do
    counts[labels[i]] = counts[labels[i]] + 1
  end

  return counts
end
