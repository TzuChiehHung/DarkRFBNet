import torch
import torch.nn as nn
import numpy as np

from collections import defaultdict, OrderedDict


def model_info(model, input_size, batch_size=-1, device='cuda'):

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split('.')[-1].split("'")[0]
            module_idx = len(summary)

            m_key = '%s-%i' % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]['input_shape'] = list(input[0].size())
            summary[m_key]['input_shape'][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]['output_shape'] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]['output_shape'] = list(output.size())
                summary[m_key]['output_shape'][0] = batch_size

            params = 0
            if hasattr(module, 'weight') and hasattr(module.weight, 'size'):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]['trainable'] = module.weight.requires_grad
            if hasattr(module, 'bias') and hasattr(module.bias, 'size'):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]['nb_params'] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        'cuda',
        'cpu',
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == 'cuda' and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print('\n' + '-'*64)
    line_new = '{:>20}  {:>25} {:>15}'.format('Layer (type)', 'Output Shape', 'Param #')
    print(line_new)
    print('='*64)
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = '{:>20}  {:>25} {:>15}'.format(
            layer,
            str(summary[layer]['output_shape']),
            '{0:,}'.format(summary[layer]['nb_params']),
        )
        total_params += summary[layer]['nb_params']
        total_output += np.prod(summary[layer]['output_shape'])
        if 'trainable' in summary[layer]:
            if summary[layer]['trainable'] == True:
                trainable_params += summary[layer]['nb_params']
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print('='*64)
    print('Total params: {0:,}'.format(total_params))
    print('Trainable params: {0:,}'.format(trainable_params))
    print('Non-trainable params: {0:,}'.format(total_params - trainable_params))
    print('-'*64)
    print('Input size (MB): %0.2f' % total_input_size)
    print('Forward/backward pass size (MB): %0.2f' % total_output_size)
    print('Params size (MB): %0.2f' % total_params_size)
    print('Estimated Total Size (MB): %0.2f' % total_size)
    print('-'*64 + '\n')

def parameters_info(module_list):
    '''Print a line-by-line parameters description of a PyTorch model'''
    n_p = sum(x.numel() for x in module_list.parameters())  # number parameters
    n_g = sum(x.numel() for x in module_list.parameters() if x.requires_grad)  # number gradients
    print('\n' + '-'*150)
    print('{:>5s} {:>50s} {:>9s} {:>12s} {:>20s} {:>12s} {:>12s}'.format('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    print('='*150)
    for i, (name, p) in enumerate(module_list.named_parameters()):
        name = name.replace('module_list.', '') #%5g %50s %9s %12g %20s %12.3g %12.3g
        print('{:>5d} {:>50s} {:>9} {:>12} {:>20s} {:>12.3g} {:>12.3g}'.format(
            i, name, str(p.requires_grad), p.numel(), str(list(p.shape)), p.mean(), p.std()))
    print('-'*150)
    print('Model Summary: {:g} layers, {:,} parameters, {:,} gradients'.format(i + 1, n_p, n_g))
    print('-'*150 + '\n')

def layers_info(module_defs):
    '''Print a line-by-line layers description of a model definition config'''
    print('\n' + '-'*64)
    print('{:>5s} {:>20s} {:>10s} {:>15s} {:>10s}'.format('Layer', 'Type', 'Ref.', 'Size','Filters'))
    print('='*64)
    for i, layer in enumerate(module_defs):
        layer = defaultdict(str, layer)
        name = layer['type']
        if name == 'shortcut':
            ref = layer['from']
        elif name == 'route':
            ref = layer['layers']
        else:
            ref = ''
        size = layer['size']
        stride = layer['stride']
        filters = layer['filters']
        
        size = '{:1s} x {:1s} / {:1s}'.format(size, size, stride) if size or stride else ''
        print('{:>5d} {:>20s} {:>10s} {:>15s} {:>10s}'.format(i, name, ref, size, filters))
    print('-'*64 + '\n')
