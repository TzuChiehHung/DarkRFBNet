import torch
import torch.nn as nn

from utils.parse_config import parse_model_cfg
from models.modules import create_modules
# from collections import defaultdict

class DarkNet53(nn.Module):
    '''DarkNet 53 backbone'''

    def __init__(self, cfg_path, img_size=416):
        super(DarkNet53, self).__init__()

        self.module_defs = parse_model_cfg(cfg_path)
        self.module_defs[0]['cfg'] = cfg_path
        self.module_defs[0]['height'] = img_size
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.img_size = img_size

    def forward(self, x, targets=None, var=0):
        img_size = x.shape[-1]
        layer_outputs = []
        output = []

        for module_def, module in zip(self.module_defs, self.module_list):
            mtype = module_def['type']
            if mtype in ['convolutional', 'upsample', 'maxpool']:
                x = module(x)
            elif mtype == 'route':
                layer_i = [int(x) for x in module_def['layers'].split(',')]
                if len(layer_i) == 1:
                    x = layer_outputs[layer_i[0]]
                else:
                    x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif mtype == 'shortcut':
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]
                output.append(x)
            layer_outputs.append(x)
        return x
