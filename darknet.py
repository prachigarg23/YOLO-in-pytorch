from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from util import *

def parse_cfg(cfgfile):
    """
        This function takes in the yolo v3 configuration file.

        The cfg file is converted into a list of blocks where each block corresponds to a block in the cfg file.

        Each block is in the form of a dictionary which stores the variables and their values as (key, value) pairs inside the dictionary.

        The block format will help us define modules for each layer
    """
    fil = open(cfgfile, 'r')
    lines = fil.read().split('\n')
    lines = [x for x in lines if len(x)>0 ]     #removed null strings
    lines = [x for x in lines if x[0]!='#']     #removed comments
    lines = [x.rstrip().lstrip() for x in lines]       #removed whitespaces before and after

    block = {}
    blocks = []

    for line in lines:
        if line.startswith('[') and line.endswith(']'):
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split('=')
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)    #this appends the last block to blocks

    return blocks


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


def create_modules(blocks):
    """
        This module takes as input the list of all blocks as specified by the cfg file and defines the neural network layers corresponding to each block.
    """
    # first we shall save the information about the network from the block [net].
    # this gives information about the input dimensions and training parameters
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    # nn.ModuleList just stores all submodules(layers) in a single list
    # prev_filters keeps track of the number of feature maps in previous layer so that it helps us define the filter depth for next convolutiona
    # output_filters stores the output depth of each convolution, i.e., the number of feature maps produced as result of each convolution. THis is required as we need to acccess previous feature maps in route layers

    for index, b in enumerate(blocks[1:]):
        module = nn.Sequential()

        """
        procedure:
        1. check the type of block
        2. create a new module for the block
        3. append the new block to module_list
        """

        # 2d conv
        if(b["type"] == 'convolutional'):
            activation = b["activation"]
            try:
                batch_normalize = int(b["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True
            filters = int(b["filters"]) #number of filters
            kernel_size = int(b["size"]) # size of filter
            stride = int(b["stride"])
            padding = int(b["pad"])

            # the padding paramter is either 0 or 1. 0 corresponds to no padding (valid convolution), 1 corresponds to same convolution
            if padding:
                pad = (kernel_size - 1) // 2
                # formula for calculating pad in case of 'same' convolution
            else:
                pad = 0
                # valid convolution

            # add conv layer
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module("conv_{0}".format(index), conv)

            # add batch_norm layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)

            # add activation layer
            # This network has either linear or leaky relu activation. thus, we only need to check for leaky relu
            if activation == 'leaky':
                actn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{0}".format(index), actn)

        # bilinear 2d upsampling
        elif b["type"] == 'upsample':
            stride = int(b["stride"])
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            module.add_module("upsample_{}".format(index), upsample)

        # route layer
        elif b["type"] == 'route':

            # first we need to get the layers (2 or 1)
            b["layers"] = b["layers"].split(',')
            start = int(b["layers"][0])
            try:
                end = int(b["layers"][1])
            except:
                end = 0
            # start, end(if it exists) are the 2 layers
            # now we use feature maps only from layers previous to the current layer
            # hence we assume that if start and end are positive, they have to be less than index

            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            # now the start and end values are negative(given as relative numbers to index)

            # adding an empty layer to the Model
            # THIS CLASS HAS BEEN DEFINED IN THE CODE ABOVE
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)

            # we need to update the number of filters as output of route layer (after concatenation or otherwise)
            if end < 0:
                filters = output_filters[index+start] + output_filters[index+end]
            else:
                filters = output_filters[index+start]

        # shortcut layer - corresponding to the skip connections
        elif b["type"] == 'shortcut':
            shortcut = EmptyLayer()
            module.add_module("shortcut_{0}".format(index), shortcut)


        # yolo layer - detection layer
        elif b["type"] == 'yolo':
            mask = b["mask"]
            mask = mask.split(',')
            mask = [int(x) for x in mask]
            # we got the indices of the anchors to be used
            anchors = b["anchors"].split(',')
            anchors = [int(x) for x in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]
            # we got the anchors to be used from the mask

            # THIS CLASS HAS BEEN DEFINED IN THE CODE ABOVE
            detection_layer = DetectionLayer(anchors)
            # this detection layer holds the anchors for bounding boxes
            module.add_module("Detection_{}".format(index), detection_layer)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
    return (net_info, module_list)

#blocks = parse_cfg("cfg/yolov3.cfg")
#print(create_modules(blocks))

"""
defined the various layers used in YOLO
now we will define the network architecture. The following class defines the forward pass of YOLO
"""
class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        outputs = {} # WE CACHE THE OUTPUT FEATURE MAPS FROM EACH LAYER SO THAT IT CAN BE USED BY ROUTE AND SHORTCUT LAYERS

        write = 0
        for i, module in enumerate(modules):
            module_type = (module["type"])

            if module_type == 'convolutional' or module_type == 'upsample':
                x = self.module_list[i](x)


            elif module_type == 'route':
                layers = module["layers"]
                layers = [int(a) for a in layers]
                if (layers[0]) > 0:
                    layers[0] = layers[0] - i
                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2) , 1)


            elif module_type == 'shortcut':
                from_ = int(module["from"])
                x = outputs[i-1] + outputs[i+from_]


            elif module_type == 'yolo':

                anchors = self.module_list[i][0].anchors
                inp_dim = int(self.net_info["height"])
                num_classes = int(module["classes"])

                # we have the required data
                # transform the data
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)

                # we concatenate the detection maps at different scales (we have detection maps from 3 scales)
                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)

            outputs[i] = x
        return detections

        # function to load weights
    def load_weights(self, weightfile):
        fp = open(weightfile, 'rb')
        """
        The first 160 bytes of the weights file store 5 int32 values which constitute the header of the file
        we read the header separately

        The header is of type int32, the weights are of type float32
        """
        header = np.fromfile(fp, dtype=np.int32, count=5)
        #convert to tensor and store as class attribute
        self.header = torch.from_numpy(header)
        #keep a track of all the images seen by the network
        self.seen = self.header[3]
        weights = np.fromfile(fp, dtype=np.float32)

        # we keep a variable ptr that keeps track of our position in the weights array
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i+1]["type"]
            if module_type == 'convolutional':
                model = self.module_list[i]
                try:
                    batch_n = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_n = 0

                conv = model[0] #the 1st layer was conv, then batch_norm, then activation
                if (batch_n):
                    bn = model[1]

                    # get no of weights of batch norm layer
                    num_bn_biases = bn.bias.numel()

                    # load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases

                    #Cast the loaded weights into dims of model weights.
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    #Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    # conv biases
                    num_biases = conv.bias.numel()

                    conv_biases = torch.from_numpy(weights[ptr:ptr+num_biases])
                    ptr += num_biases
                    conv_biases = conv_biases.view_as(conv.bias.data)
                    conv.bias.data.copy_(conv_biases)

                    # conv weights
                num_weights = conv.weight.numel()

                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr += num_weights
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)






#model = Darknet("cfg/yolov3.cfg")
#model.load_weights("yolov3.weights")
"""inp = get_test_input()
pred = model(inp, torch.cuda.is_available())
print (pred)
print(pred.size())
"""
