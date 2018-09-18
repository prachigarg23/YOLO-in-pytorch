from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

def unique(tensor):
    # we convert the tensor to numpy, get the unique classes and conver it back to tensor
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res

def bbox_iou(box1, box2):
    """

    IMPORTANT NOTE:
    The distances of coordinates in the bounding box system are calculated from the top left corner of the box.
    so the origin is not the usual one but top left corner one.

    """
    # returns the IOU
    # get the 2 box coordinates
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    # get the intersection box coordinates
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    # area of Intersection
    area_intersection = torch.clamp(inter_x2 - inter_x1 + 1, min=0) * torch.clamp(inter_y2 - inter_y1 + 1, min=0)

    # area bboxes
    area_box1 = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    area_box2 = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)

    IOU = area_intersection / (area_box1 + area_box2 - area_intersection)

    return IOU


def predict_transform(prediction, input_dim, anchors, num_classes, CUDA = False):

    """
    here I transform the raw output from the network to a meaningful one with appropriate format for the bounding boxes and labels
    these bounding boxes can later be combined from those sampled at earlier stages (we get bboxes at 3 stages)
    """
    """
    The dimensions of output are: (batch_size, output_image_size, output_image_size, num_anchors*(5+num_classes))
    we need to modify the shape so we can perform operations and changes easily
    """
    batch_size = prediction.size(0)
    stride = input_dim // prediction.size(2)
    grid_size = input_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)


    # the dimensions of anchors are according to the size of input image not predicted image. so we need to divide them by the stride scale_factor
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]
    """
    The new dimensions of prediciton : (batch_size, num_of_cells_in_output * num_of_bboxes_per_cell, number_of_attributes_per_bbox)

    We want the values of the centre of the bboxes to be relative to the size of the grid cell. Taking the dimensions of grid cell to be unity, we apply sigmoid on the x and y centre values to bring it in the range [0,1]
    Also, the object confidence score is a measure of whether an object (centre) is present in the cell or not. therefore, applying sigmoid gives me a probability of it being present
    """

    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])


    #To get the actual bbox coordinates and values, we create a grid and find the position of the grid cells. These are the x and y grid cell coordinates(top left)
    #Add the x, y centres of bboxes to the grid cell top left coordinates to get the position of bbox centres on the grid.
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid, grid)

    #store numpy matrix into pytorch CPU tensor
    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)
    # if GPU available, convert CPU tensor into GPU tensor
    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    # we need coordinate pairs from individual x and y values
    # concat them to get each row storing 1 pair
    # we have 3 bboxes per cell. hence repeat it num_anchors_time
    # lastly, add another dimension for the batch_size using unsqueeze
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

    prediction[:,:,:2] += x_y_offset

    #log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    # we need 3 anchors for each cell
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    """
    we apply the log transform to the box height and width:
    h(required_height) = a[0](height anchor box) * exp(h(height_as_given_by_network_output))
    """
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors

    # apply sigmoid to class scores
    prediction[:,:,5: 5+num_classes] = torch.sigmoid(prediction[:,:,5: 5+num_classes])

    """now we resize the detected boxes to match the original input image size"""
    prediction[:,:,:4] *= stride

    return prediction



def write_results(prediction, confidence, num_classes, nms_conf = 0.4):

    """
    Each bounding box's 1st 4 coordinates are in this order initially: (x_centre, y_centre, width, height)

    In predictions tensor, we have 10647 bounding boxes for each image
    In this function, we reduce this number to the real predictions using object confidence thresholding and Non Max Supression
    The bboxes below a threshold object confidence (probability that an object exists inside that box) will be removed
    The IOU will be calculated for each bbox and NMS will be used to ensure that there is a single bbox corresponding to every unique object
    """
    #the mask needs to be of float type as we will multiply the bbox attributes with it

    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction * conf_mask
    #conf_mask has [batch_size, no.of.bboxes] dimension initially
    #but the actual data is along the z axis and we need to compare along that axis. hence we add another dim
    #pytorch broadcasts the conf_mask tensor before multiplying it elementwise with prediction tensor to yield result

    # the IOU is easier to calculate if predictions are in the form of (top_left_corner_x, top_left_corner_y) and (bottom_right_x, bottom_right_y)
    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2)
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_corner[:,:,:4]

    batch_size = prediction.size(0)

    #to check if any output has been added to output tensor or not
    write = False

    for ind in range(batch_size):
        image_pred = prediction[ind]

        # now we get class with max class score and replace all class scores with it.
        # get max along columns
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+ num_classes], 1)
        # the 2 variables above are 1D tensors. we need tensors of shape [no_of_bboxes, 1] to append it to img_pred
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)

        # we need to remove the bboxes with attributes 0
        non_zero_ind =  (torch.nonzero(image_pred[:,4]))

        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        except:
            continue

        if image_pred_.shape[0] == 0:
            continue

        # we get the unique classes for each image
        img_classes = unique(image_pred_[:,-1])

        for cls in img_classes:
            # we get all detections for 1 class and perform nms
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1,7)

            # sorting in descending order of object scores
            conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)   #Number of detections

            for i in range(idx):
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                except ValueError:
                    break
                except IndexError:
                    break

                """
                In ious, we get the Intersection Over Union of the ith bbox with each one of (i+1)th indices boxes.
                The idea here is to check if any 2 boxes have iou above a threshold or not. if they do, it means they are bounding the same object(of the same class)
                In such a case, we get rid of the one with a lower object score.
                """

                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask

                # we have zeroed all such entries

                # now remove the zero entries
                non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1,7)

            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            #Repeat the batch_id for as many detections of the class cls in the image
            seq = batch_ind, image_pred_class

            if not write:
                output = torch.cat(seq,1)
                write = True
            else:
                out = torch.cat(seq,1)
                output = torch.cat((output,out))

            #for the dog detector, i want only those predictions belonging to the dog class i.e. class index 16
            dog_mask = output*(output[:,-1] == 16).float().unsqueeze(1)
            dog_mask_ind = torch.nonzero(dog_mask[:,-2]).squeeze()
            output = output[dog_mask_ind].view(-1,8)
            if output.size(0) == 0:
                output=0
    try:
        print(output)
        print(output.size())
        return output
    except:
        print("no dog detections")
        return 0

def letterbox_image(img, inp_dim):
    """resize image with unchanged aspect ratio using padding"""
    # img is the image to be resized and inp_dim is dimensions as required by Network
    w, h = inp_dim
    img_w, img_h = img.shape[1], img.shape[0]
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_CUBIC)

    # now we will create a canvas filled with the padding color, and place image at the centre
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)
    canvas[(h-new_h)//2 : (h-new_h)//2 + new_h, (w-new_w)//2 : (w-new_w)//2 + new_w, :] = resized_image
    return canvas

def prep_image(img, inp_dim):
    """
    The images read by opencv are numpy arrays with BGR color format
    Prepare image for inputting to the neural network.
    We need pytorch tensors with RGB format
    The order of data: (batch_size, channels, height, width)

    Returns a Variable
    """
    img = (letterbox_image(img, (inp_dim, inp_dim)))
    img = img[:,:,::-1].transpose((2,0,1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img

def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names
