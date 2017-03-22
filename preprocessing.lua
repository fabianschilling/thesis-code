function preprocess(dataset, method)

  if args.verbose then print('Preprocessing: ' .. method) end

  if method == 'None' then
    return dataset
  elseif method == 'Standardize' then
    return standardize(dataset)
  else
    error('Unrecognized preprocessing method: ' .. method)
  end
end

function standardize(dataset)

  local mean = dataset.train.inputs:mean()
  local std = dataset.train.inputs:std()

  for i,v in ipairs({'train', 'test', 'valid'}) do
    dataset[v].inputs:add(-mean)
    dataset[v].inputs:div(std)
  end

  return dataset
end
