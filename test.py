from utils.parse_config import *
from utils.utils  import *
# from models.yolov3 import YOLOv3
from models.darknet53 import DarkNet53

# cfg = 'cfg/yolov3.cfg'
# net = YOLOv3(cfg)

cfg = 'cfg/darknet53.cfg'
net = DarkNet53(cfg)

layers_info(net.module_defs)
parameters_info(net.module_list)
model_info(net, input_size=(3,416,416))

