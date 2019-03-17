from utils.parse_config import *
from utils.utils  import *
# from models.yolov3 import YOLOv3
from models.dark_rfbnet import DarkRFBNet

# cfg = 'cfg/yolov3.cfg'
# net = YOLOv3(cfg)

# cfg = 'cfg/darknet53.cfg'
cfg = 'cfg/dark53_rfbnet.cfg'
net = DarkRFBNet(cfg)

layers_info(net.module_defs)
parameters_info(net.module_list)
model_info(net, input_size=(3,416,416))

