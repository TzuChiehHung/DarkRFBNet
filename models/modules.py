
import torch
import torch.nn as nn
import torch.nn.functional as F

ONNX_EXPORT = False

def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams['channels'])]
    module_list = nn.ModuleList()
    yolo_layer_count = 0
    for i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def['type'] == 'convolutional':
            bn = int(module_def['batch_normalize'])
            filters = int(module_def['filters'])
            kernel_size = int(module_def['size'])
            pad = (kernel_size - 1) // 2 if int(module_def['pad']) else 0
            modules.add_module('conv_%d' % i, nn.Conv2d(in_channels=output_filters[-1],
                                                        out_channels=filters,
                                                        kernel_size=kernel_size,
                                                        stride=int(module_def['stride']),
                                                        padding=pad,
                                                        bias=not bn))
            if bn:
                modules.add_module('batch_norm_%d' % i, nn.BatchNorm2d(filters))
            if module_def['activation'] == 'leaky':
                modules.add_module('leaky_%d' % i, nn.LeakyReLU(0.1))

        elif module_def['type'] == 'maxpool':
            kernel_size = int(module_def['size'])
            stride = int(module_def['stride'])
            if kernel_size == 2 and stride == 1:
                modules.add_module('_debug_padding_%d' % i, nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules.add_module('maxpool_%d' % i, maxpool)

        elif module_def['type'] == 'upsample':
            # upsample = nn.Upsample(scale_factor=int(module_def['stride']), mode='nearest')  # WARNING: deprecated
            upsample = Upsample(scale_factor=int(module_def['stride']))
            modules.add_module('upsample_%d' % i, upsample)

        elif module_def['type'] == 'route':
            layers = [int(x) for x in module_def['layers'].split(',')]
            filters = sum([output_filters[i + 1 if i > 0 else i] for i in layers])
            modules.add_module('route_%d' % i, EmptyLayer())

        elif module_def['type'] == 'shortcut':
            filters = output_filters[int(module_def['from'])]
            modules.add_module('shortcut_%d' % i, EmptyLayer())

        elif module_def['type'] == 'yolo':
            anchor_idxs = [int(x) for x in module_def['mask'].split(',')]
            # Extract anchors
            anchors = [float(x) for x in module_def['anchors'].split(',')]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            nC = int(module_def['classes'])  # number of classes
            img_size = int(hyperparams['height'])
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, nC, img_size, yolo_layer_count, cfg=hyperparams['cfg'])
            modules.add_module('yolo_%d' % i, yolo_layer)
            yolo_layer_count += 1
        elif module_def['type'] == 'rfb':
            filters = int(module_def['filters'])
            stride = int(module_def['stride'])
            scale = int(module_def['scale'])
            visual = int(module_def['visual'])
            bn = int(module_def['batch_normalize'])
            modules.add_module('rfb_%d' % i, RFBLayer(output_filters[-1], filters,
                stride=stride, scale=scale, visual=visual, bn=bn))

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list

class RFBLayer(nn.Module):
    '''
                                    padding and dilation
    branch 0: 1x1, 3x3              (1)
    branch 1: 1x1, 3x3, 3x3         (1 + visual)
    branch 2: 1x1, 3x3, 3x3, 3x3    (1 + 2*visual)
    concat: branch 0, 1, 2
    conv: 1x1
    shortcut conv: 1x1
    sum: shortcut + scale * conv
    LeakyReLU
    '''

    def __init__(self, in_channels, out_channels, stride=1, scale=1, visual=2, bn=False):
        super(RFBLayer, self).__init__()
        self.scale = scale

        inter_channels = in_channels // 8

        self.branch0 = nn.Sequential(
            ConvLayer(in_channels, 2*inter_channels, kernel_size=1, stride=1, bn=bn),
            ConvLayer(2*inter_channels, 2*inter_channels, kernel_size=3, stride=stride, padding=1, dilation=1, activation=False, bn=bn)
        )
        self.branch1 = nn.Sequential(
            ConvLayer(in_channels, inter_channels, kernel_size=1, stride=1, bn=bn),
            ConvLayer(inter_channels, 2*inter_channels, kernel_size=3, stride=stride, padding=1, dilation=1, bn=bn),
            ConvLayer(2*inter_channels, 2*inter_channels, kernel_size=3, stride=1, padding=1+visual, dilation=1+visual, activation=False, bn=bn)
        )
        self.branch2 = nn.Sequential(
            ConvLayer(in_channels, inter_channels, kernel_size=1, stride=1, bn=bn),
            ConvLayer(inter_channels, (inter_channels//2)*3, kernel_size=3, stride=1, padding=1, dilation=1, bn=bn),
            ConvLayer((inter_channels//2)*3, 2*inter_channels, kernel_size=3, stride=stride, padding=1, dilation=1, bn=bn),
            ConvLayer(2*inter_channels, 2*inter_channels, kernel_size=3, stride=1, padding=1+2*visual, dilation=1+2*visual, activation=False, bn=bn)
        )
        self.concat_conv = ConvLayer(6*inter_channels, out_channels, kernel_size=1, stride=1, activation=False, bn=bn)
        self.shortcut_conv = ConvLayer(in_channels, out_channels, kernel_size=1, stride=stride, activation=False, bn=bn)
        self.activation = nn.LeakyReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)
        out = self.concat_conv(out)
        shortcut = self.shortcut_conv(x)
        out = self.scale * out + shortcut
        out = self.activation(out)

        return out


class ConvLayer(nn.Module):


    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, activation=True, bn=False):
        super(ConvLayer, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias= not bn)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None  # use default setting
        self.activation = nn.LeakyReLU() if activation else None # change ReLU to LeakyReLU, use default setting

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class YOLOLayer(nn.Module):


    def __init__(self, anchors, nC, img_size, yolo_layer, cfg):
        super(YOLOLayer, self).__init__()

        nA = len(anchors)
        self.anchors = torch.FloatTensor(anchors)
        self.nA = nA  # number of anchors (3)
        self.nC = nC  # number of classes (80)
        self.img_size = 0
        # self.coco_class_weights = coco_class_weights()

        if ONNX_EXPORT:  # grids must be computed in __init__
            stride = [32, 16, 8][yolo_layer]  # stride of this layer
            if cfg.endswith('yolov3-tiny.cfg'):
                stride *= 2

            self.nG = int(img_size / stride)  # number grid points
            self.create_grids(img_size, self.nG)

    def forward(self, p, img_size, targets=None, var=None):
        if ONNX_EXPORT:
            bs, nG = 1, self.nG  # batch size, grid size
        else:
            bs, nG = p.shape[0], p.shape[-1]

            if self.img_size != img_size:
                self.create_grids(img_size, nG)

                if p.is_cuda:
                    self.grid_xy = self.grid_xy.cuda()
                    self.anchor_vec = self.anchor_vec.cuda()
                    self.anchor_wh = self.anchor_wh.cuda()

        # p.view(bs, 255, 13, 13) -- > (bs, 3, 13, 13, 80)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.nA, self.nC + 5, nG, nG).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        # xy, width and height
        xy = torch.sigmoid(p[..., 0:2])
        wh = p[..., 2:4]  # wh (yolo method)
        # wh = torch.sigmoid(p[..., 2:4])  # wh (power method)

        # Training
        if targets is not None:
            MSELoss = nn.MSELoss()
            BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
            CrossEntropyLoss = nn.CrossEntropyLoss()

            # Get outputs
            p_conf = p[..., 4]  # Conf
            p_cls = p[..., 5:]  # Class

            txy, twh, mask, tcls = build_targets(targets, self.anchor_vec, self.nA, self.nC, nG)

            tcls = tcls[mask]
            if p.is_cuda:
                txy, twh, mask, tcls = txy.cuda(), twh.cuda(), mask.cuda(), tcls.cuda()

            # Compute losses
            nT = sum([len(x) for x in targets])  # number of targets
            nM = mask.sum().float()  # number of anchors (assigned to targets)
            k = 1  # nM / bs
            if nM > 0:
                lxy = k * MSELoss(xy[mask], txy[mask])
                lwh = k * MSELoss(wh[mask], twh[mask])

                lcls = (k / 4) * CrossEntropyLoss(p_cls[mask], torch.argmax(tcls, 1))
                # lcls = (k * 10) * BCEWithLogitsLoss(p_cls[mask], tcls.float())
            else:
                FT = torch.cuda.FloatTensor if p.is_cuda else torch.FloatTensor
                lxy, lwh, lcls, lconf = FT([0]), FT([0]), FT([0]), FT([0])

            lconf = (k * 64) * BCEWithLogitsLoss(p_conf, mask.float())

            # Sum loss components
            loss = lxy + lwh + lconf + lcls

            return loss, loss.item(), lxy.item(), lwh.item(), lconf.item(), lcls.item(), nT

        else:
            if ONNX_EXPORT:
                grid_xy = self.grid_xy.repeat((1, self.nA, 1, 1, 1)).view((1, -1, 2))
                anchor_wh = self.anchor_wh.repeat((1, 1, nG, nG, 1)).view((1, -1, 2)) / nG

                # p = p.view(-1, 85)
                # xy = xy + self.grid_xy[0]  # x, y
                # wh = torch.exp(wh) * self.anchor_wh[0]  # width, height
                # p_conf = torch.sigmoid(p[:, 4:5])  # Conf
                # p_cls = F.softmax(p[:, 5:85], 1) * p_conf  # SSD-like conf
                # return torch.cat((xy / nG, wh, p_conf, p_cls), 1).t()

                p = p.view(1, -1, 85)
                xy = xy + grid_xy  # x, y
                wh = torch.exp(p[..., 2:4]) * anchor_wh  # width, height
                p_conf = torch.sigmoid(p[..., 4:5])  # Conf
                p_cls = p[..., 5:85]
                # Broadcasting only supported on first dimension in CoreML. See onnx-coreml/_operators.py
                # p_cls = F.softmax(p_cls, 2) * p_conf  # SSD-like conf
                p_cls = torch.exp(p_cls).permute((2, 1, 0))
                p_cls = p_cls / p_cls.sum(0).unsqueeze(0) * p_conf.permute((2, 1, 0))  # F.softmax() equivalent
                p_cls = p_cls.permute(2, 1, 0)
                return torch.cat((xy / nG, wh, p_conf, p_cls), 2).squeeze().t()

            p[..., 0:2] = xy + self.grid_xy  # xy
            p[..., 2:4] = torch.exp(wh) * self.anchor_wh  # wh yolo method
            # p[..., 2:4] = ((wh * 2) ** 2) * self.anchor_wh  # wh power method
            p[..., 4] = torch.sigmoid(p[..., 4])  # p_conf
            p[..., :4] *= self.stride

            # reshape from [1, 3, 13, 13, 85] to [1, 507, 85]
            return p.view(bs, -1, 5 + self.nC)

    def create_grids(self, img_size, nG):
        self.stride = img_size / nG

        # build xy offsets
        grid_x = torch.arange(nG).repeat((nG, 1)).view((1, 1, nG, nG)).float()
        grid_y = grid_x.permute(0, 1, 3, 2)
        self.grid_xy = torch.stack((grid_x, grid_y), 4)

        # build wh gains
        self.anchor_vec = self.anchors / self.stride
        self.anchor_wh = self.anchor_vec.view(1, self.nA, 1, 1, 2)


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()

    def forward(self, x):
        return x


class Upsample(nn.Module):
    # Custom Upsample layer (nn.Upsample gives deprecated warning message)

    def __init__(self, scale_factor=1, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
