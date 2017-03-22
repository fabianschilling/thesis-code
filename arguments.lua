function getArguments()

  local cmd = torch.CmdLine()

  cmd:text('Options:')

  -- General
  cmd:option('-epochs', 100, 'Number of epochs to run')

  -- Model
  cmd:option('-model', 'MLP', 'Model: MLP | CNN | VGG')

  -- Dataset
  cmd:option('-dataset', 'MNIST', 'Dataset: MNIST | CIFAR10 | CIFAR100 | SVHN')
  cmd:option('-batchsize', 64, 'Batch size')

  -- Activation function
  cmd:option('-activation', 'ReLU', 'Activation function: Sigmoid | ReLU | ELU')

  -- Initialization
  cmd:option('-init', 'Kaiming', 'Init: Default | Xavier | Kaiming')
  cmd:option('-dist', 'Normal', 'Init distribution: Normal | Uniform')

  -- Batch normalization
  cmd:option('-batchnorm', false, 'Use batch normalization')

  -- Preprocessing
  cmd:option('-preprocess', 'Standardize', 'Preprocessing: Standardize | None')

  -- Optimization
  cmd:option('-optimizer', 'NAG', 'Optimization: SGD | NAG | Adagrad | Adadelta | Adam | RMSProp')
  cmd:option('-momentum', 0.9, 'Momentum')

  -- Learning rate and decay
  cmd:option('-lr', 0.01, 'Initial learning rate')
  cmd:option('-lrMin', 0.0001, 'Minimum learning rate')
  cmd:option('-lrDecay', 0.5, 'Learning rate decay')
  cmd:option('-lrDecayType', 'adaptive', 'Learning rate decay type: linear | power | exponential | adaptive | none')

  -- Linear learning rate
  cmd:option('-lrDecaySat', 100, 'Epoch when lr should be equal to lrMin')

  -- Adaptive learning rate
  cmd:option('-lrDecayWait', 5 , 'Learning rate drops by lrfactor after lrwait epochs')

  -- Weight decay
  cmd:option('-wDecayL1', 0, 'L1 weight decay')
  cmd:option('-wDecayL2', 0.0005, 'L2 weight decay')

  -- Dropout
  cmd:option('-dropout', false, 'Use dropout')
  cmd:option('-dropoutProb', 0.5, 'Dropout probability')

  -- Printing, logging, and saving
  cmd:option('-verbose', true, 'Verbose mode')
  cmd:option('-progress', true, 'Progress bar')
  cmd:option('-confusion', true, 'Print confusion matrix')
  cmd:option('-logging', true, 'Enable logging')
  cmd:option('-save', false, 'Save model')
  cmd:option('-path', paths.concat(paths.cwd(), 'logs'), 'Path for saving everything')

  cmd:option('-maxEpochs', 20, 'Max number of epochs without improvement')
  cmd:option('-cudnn', true, 'Use cuDNN')

  cmd:option('-validRatio', 0.1, 'Ratio of train set that will be valid set')

  local args = cmd:parse(arg)

  local type = args.batchnorm and 'batchnorm' or 'vanilla'
  local model = string.lower(args.model)
  local dataset = string.lower(args.dataset)
  local activation = string.lower(args.activation)
  local bs = args.batchsize
  local lr = args.lr

  local logname = string.format('log_%s_%s_%s_%s_bs%s_lr%s', model, dataset, activation, type, bs, lr)

  --args.path = paths.concat(paths.cwd(), logname)
  args.path = paths.concat('/home/fabian/logs/', logname)

  return args

end

