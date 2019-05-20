

import torch.nn as nn


def conv_bn_relu(in_planes, out_planes, kernel=3, stride=1, padding=1):
    net = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=1),
                         nn.BatchNorm2d(num_features=out_planes),
                         nn.ReLU(True))
    return net;

def convtranspose_bn_relu(in_planes, out_planes, kernel=3, stride=2, padding=1):
    net = nn.Sequential(nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=1, output_padding =1),
                         nn.BatchNorm2d(num_features=out_planes),
                         nn.ReLU(True))
    return net;

# input size Bx3x224x224
class SegmenterModel(nn.Module):
    def __init__(self, in_size=3):
        super(SegmenterModel, self).__init__()
        
        self.features = nn.Sequential()

        self.features.add_module('3x3', conv_bn_relu(in_planes = 3, out_planes = 8, stride = 2))
        
        self.features.add_module('downsample_1', conv_bn_relu(8, 16, stride = 2))
        
        self.features.add_module('conv_1', conv_bn_relu(16, 16))
        self.features.add_module('conv_2', conv_bn_relu(16, 16))
        self.features.add_module('downsample_2', conv_bn_relu(16, 32, stride = 2))
        
        self.features.add_module('conv_3', conv_bn_relu(32, 32))
        self.features.add_module('conv_4', conv_bn_relu(32, 32))
        self.features.add_module('downsample_3', conv_bn_relu(32, 64, stride = 2))
        
        self.features.add_module('conv_5', conv_bn_relu(64, 64))
        self.features.add_module('conv_6', conv_bn_relu(64, 64))
    
        self.features.add_module('upsample_1', convtranspose_bn_relu(64, 32))
        self.features.add_module('conv_7', conv_bn_relu(32, 32))
        self.features.add_module('conv_8', conv_bn_relu(32, 32))
        
        self.features.add_module('upsample_2', convtranspose_bn_relu(32, 16))
        self.features.add_module('conv_9', conv_bn_relu(16, 16))
        self.features.add_module('conv_10', conv_bn_relu(16, 16))
        
        self.features.add_module('upsample_3', convtranspose_bn_relu(16, 8))
        
        self.features.add_module('3x3_1', conv_bn_relu(8, 8))
        
        self.features.add_module('upsample_4', convtranspose_bn_relu(8, 1))
        
        self.fc_classifier = nn.Softmax2d()
        
        
        
    def forward(self, input):
        
        input = self.features(input)
        input = self.fc_classifier(input)
        
        return input

