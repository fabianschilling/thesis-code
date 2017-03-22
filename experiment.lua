require 'torch' -- torch
require 'cutorch' -- CUDA backend for torch
require 'cudnn' -- cuDNN backend for nn
require 'cunn' -- CUDA backend for nn
require 'optim' -- Optimization methods
require 'paths' -- Paths for directories
require 'xlua' -- Progress bar
require 'pl' -- Useful functions

torch.manualSeed(1337) -- manual torch seed
cutorch.manualSeed(1337) -- manual cutorch seed
cudnn.benchmark = true
--cudnn.fastest = true

-- Load files
dofile('arguments.lua')
dofile('datasets.lua')
dofile('preprocessing.lua')
dofile('models.lua')
dofile('initialization.lua')
dofile('logging.lua')
dofile('train.lua')
dofile('evaluate.lua')

-- Command line arguments
args = getArguments()

-- Load dataset
dataset = loadDataset(args.dataset, args.validRatio)

if args.verbose then print('Batch size: ' .. args.batchsize) end

-- Data preprocessing
dataset = preprocess(dataset, args.preprocess)

-- Load model
model = loadModel(args.model):cuda()
cudnn.convert(model, cudnn)

-- Initialize model
model = initializeModel(model, args.init, args.dist)

-- Define epochLoss function
criterion = nn.CrossEntropyCriterion():cuda()

-- Enable logging
createLoggers()

-- Optimization
optimConfig = {
  learningRate = args.lr,
  learningRateDecay = 0, -- implemented otherwise
  weightDecay = 0, -- implemented otherwise
  momentum = args.momentum,
  dampening = 0 -- don't need this
}

optimState = {}

if args.verbose then print('Optimization method: ' .. args.optimizer .. ' (lr: ' .. args.lr .. ')') end


confusion = optim.ConfusionMatrix(dataset.classes)
parameters, gradParameters = model:getParameters()

bestAccuracy = 0.0 -- keeps track of current best validation accuracy
epoch = 1 -- keeps track of current epoch
epochsSinceImprovement = 0 -- keeps track of epochs without improvemnt
epochsSinceDecay = 0 -- keeps track of epoch since last adaptive decay

if args.logging then
  logConfig(args)
  logModel(model)
end

local timeStart = torch.tic()

-- Run as long as specified or as long as improvement is made
while (epoch <= args.epochs) or (args.epochs < 0)  do

  -- do one epoch
  print('\n *** Epoch ' .. epoch .. ' of ' .. args.epochs .. ' ***')

  -- Timing
  local epochTimeStart = torch.tic()

  -- Train (get data and regularization loss)
  local epochDataLoss, epochRegLoss, trainAccuracy = train(dataset.train)
  local epochLoss = epochDataLoss + epochRegLoss

  -- print train accuracy and epochLoss
  print(string.format('Loss: %f', epochLoss))
  print(string.format('Learning rate: %f', optimConfig.learningRate))

  -- Evaluate on validation set
  local accuracy = evaluate(dataset.valid)

  -- Keep track of accuracy improvement
  if accuracy > bestAccuracy then
    print(string.format('Accuracy improvement: %.2f%% -> %.2f%%', bestAccuracy, accuracy))
    bestAccuracy = accuracy
    epochsSinceImprovement = 0
    epochsSinceDecay = 0
  else
    epochsSinceImprovement = epochsSinceImprovement + 1
    epochsSinceDecay = epochsSinceDecay + 1
    print(string.format('No accuracy improvement for %d epoch(s)', epochsSinceImprovement))
  end

  print(string.format('Accuracy: %.2f%% (best: %.2f%%)', accuracy, bestAccuracy))

  -- Learning rate decay
  if args.lrDecayType == 'adaptive' then
    if epochsSinceDecay > 0 then
      print(string.format('Epochs since last decay: %d of %d', epochsSinceDecay, args.lrDecayWait))
    end
    if epochsSinceDecay >= args.lrDecayWait then
      print(string.format('Dropping learning rate: %f%% -> %f%%', optimConfig.learningRate, optimConfig.learningRate * args.lrDecay))
      optimConfig.learningRate = optimConfig.learningRate * args.lrDecay
      epochsSinceDecay = 0 -- RESET!
    end
  elseif args.lrDecayType == 'power' then
    optimConfig.learningRate = args.lr / (1 + args.lrDecay * epoch)
  elseif args.lrDecayType == 'exponential' then
    optimConfig.learningRate = args.lr * math.exp(-args.lrDecay * epoch)
  elseif args.lrDecayType == 'linear' then
    local step = (args.lr - args.lrMin) / args.lrDecaySat
    optimConfig.learningRate = args.lr - (epoch * step)
    print(optimConfig.learningRate)
  end

  -- Minimum learning rate
  optimConfig.learningRate = math.max(optimConfig.learningRate, args.lrMin)

  -- Timing
  local time = torch.toc(epochTimeStart)

  -- Logging (epoch)
  if args.logging then
    logEpoch(epoch, epochDataLoss, epochRegLoss, accuracy, trainAccuracy, optimConfig.learningRate, time)
  end

  -- Model saving
  if args.save and accuracy > bestAccuracy then
    bestAccuracy = accuracy
    print('Better model found. Saving...')
    torch.save(paths.concat(args.path, 'model.net'), model)
  end

  -- Stop when no improvement
  if epochsSinceImprovement >= args.maxEpochs then
    print('No improvement for ' .. epochsSinceImprovement .. ' epochs. Stopping.')
    break
  end

  epoch = epoch + 1
end

local time = torch.toc(timeStart)

local testAccuracy = evaluate(dataset.test)
print(string.format('Test accuracy: %.2f%%', testAccuracy))
if args.logging then logTest(epoch, testAccuracy, time) end
