from PIL import Image,ImageOps,ImageDraw

import os 
import numpy as np
import scipy.io as sio
from loader import load

DIR_TRAIN = './trainset/'
DIR_TEST = './testset/'
#TODO: resize image to be centered on the face, and that each images has the same size. 

#using bounding box to identify faces and then get the mean scale of these images to divide. 

def get_bboxes():
# Just some ugly translations from the very nested MATLAB representation of
# the bounding box information.
    bboxes1 = sio.loadmat('Bounding_boxes/bounding_boxes_helen_trainset.mat')['bounding_boxes']
    bboxes2 = sio.loadmat('Bounding_boxes/bounding_boxes_helen_testset.mat')['bounding_boxes']
    ret = {}
    for bb in bboxes1[0]:
        ret[bb[0][0][0][0]] = list(bb[0][0][1][0])
    for bb in bboxes2[0]:
        ret[bb[0][0][0][0]] = list(bb[0][0][1][0])
    return ret


#get the dictionary where each bounding box is associated to its pic 
ret = get_bboxes()

#get all the images 
print(os.listdir(DIR_TRAIN))




def resize_im(name, bbox, param):
    img_path = DIR_TRAIN  + name + '.png'
    print(img_path)
    img = Image.open(img_path)
    bbox = comput_new_box(im.size, bbox, param)
    im_crop = im.crop(bbox)


def compute_new_bbox(image_size,bbox,parameters):
    '''
    compute the expanded bbox
    a robust function to expand the crop image bbox even the original bbox is
    around the border of the image
    ---------------------------------------------------------------------------
    INPUT:
        image_size: a tuple   ex: (height,width)
        bbox: the ground_truth bounding boxes  ex:[x0,y0,x1,y1]
        parameters: model parameter object
    OUTPUT:
        new bbox: ex:[x0,y0,x1,y1]
    ---------------------------------------------------------------------------
    '''
    x_size,y_size = image_size
    bx0,by0,bx1,by1 = bbox
    bw = by1 - by0
    bh = bx1 - bx0
    if bw > bh:
        delta = parameters.expand_rate * bw
        if by1 + delta > y_size:
            nby1 = y_size
        else:
            nby1 = int(floor(by1 + delta))
        if by0 - delta < 0:
            nby0 = 0
        else:
            nby0 = int(floor(by0 - delta))
        new_w = nby1 - nby0
        delta_h = (new_w - bh) / 2
        if bx0 - delta_h < 0:
            nbx0 = 0
        else:
            nbx0 = int(floor(bx0 - delta_h))
        if bx1 + delta_h > x_size:
            nbx1 = x_size
        else:
            nbx1 = int(floor(bx1 + delta_h))
    else:
        delta = parameters.expand_rate * bh
        if bx1 + delta > x_size:
            nbx1 = x_size
        else:
            nbx1 = int(floor(bx1 + delta))
        if bx0 - delta < 0:
            nbx0 = 0
        else:
            nbx0 = int(floor(bx0 - delta))
        new_h = nbx1 - nbx0
        delta_w = (new_h - bw) / 2
        if by0 - delta_w < 0:
            nby0 = 0
        else:
            nby0 = int(floor(by0 - delta_w))
        if by1 + delta_w > y_size:
            nby1 = y_size
        else:
            nby1 = int(floor(by1 + delta_w))
    return nbx0,nby0,nbx1,nby1
    


    