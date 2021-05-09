import torch
from torch import nn
def replace_module_from_model(model, x, y, display=True, params={}):
    for name, module in model.named_modules():
        for n, c in module.named_children():
            if isinstance(c, x):
                if display:
                    print(f'{name}.{n} =', c, f'-> {y}')
                if 'ReLU' in str(c.__class__):  
                    if params=={}:
                        setattr(module, n, y())
                    else:
                        setattr(module, n, y(**params))
                elif 'batchnorm' in str(c.__class__):
                    if params=={}:
                        setattr(module, n, y(c.num_features))
                    else:
                        setattr(module, n, y(c.num_features, **params))
def weight_reset(m):
    try:
        m.reset_parameters()
    except:
        pass
class GANLoss(nn.Module):
    def __init__(self, label_smoothing: float = 1.0):
        super(GANLoss, self).__init__()
        self.label_smoothing = label_smoothing
        self.criterion = nn.BCEWithLogitsLoss()
    def forward(self, inp, real: bool):
        if real:
            label = torch.full(inp.size(), self.label_smoothing, requires_grad=True)
        else:
            label = torch.full(inp.size(), 1 - self.label_smoothing, requires_grad=True)
        return self.criterion(inp, label)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    