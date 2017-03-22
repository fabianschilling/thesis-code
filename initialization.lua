function initializeModel(model, method, distribution)

  if args.verbose then
    print('Initialization: ' .. method .. ' (' .. distribution .. ')')
  end

  nninit = require 'nninit' -- Initialization schemes

  for i = 1, #model.modules do

    if args.init == 'Default' then break end -- do nothing when default

    local layer = model.modules[i]
    local layerName = torch.type(layer)

    if layerName:find('Linear') or layerName:find('Convolution') then
      layer:init('weight', nninit[string.lower(args.init)], {dist = string.lower(args.dist)})
      layer:init('bias', nninit.constant, 0.0)
    end
  end

  return model
end
