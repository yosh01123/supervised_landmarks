from PIL import Image,ImageOps,ImageDraw

import os 
import numpy as np
import scipy.io as sio
from loader import load
from math import floor 
from copy import deepcopy

class model_parameters(object):
    
    def __init__(self,
                 N=3,
                 alpha=0.001,
                 new_size=(400,400),
                 expand=50,
                 expand_rate=0.2,
                 orientations=4,
                 pixels_per_cell=3,
                 cells_per_block=2,
                 cells_per_side=1,
                 train_or_test='trainset',
                 hog_no_block=True,
                 demo=False):
        self.N = N
        self.alpha = alpha
        self.new_size=new_size
        self.expand =expand
        self.expand_rate = expand_rate
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.cells_per_side = cells_per_side
        self.train_or_test = train_or_test
        self.hog_no_block = hog_no_block
        self.demo = demo

#first we need to load the data

DIR_TRAIN = './trainset/'
DIR_TEST = './testset/'


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

boxes = get_bboxes()




#now that we have the bounding box, we need to resize the images.

def crop_and_resize_image(image_name,bbox,parameters):
    '''
    crop and resize the image given the ground truth bounding boxes
    also, compute the new coordinates according to transformation
    ---------------------------------------------------------------------------
    INPUT:
        image_name: a string without extension  ex: 'image_0007'
        bbox: the ground_truth bounding boxes  ex:[x0,y0,x1,y1]
        parameters: model parameter object
    OUTPUT:
        grey: a numpy array of grey image after crop and resize
        landmarks: new landmarks accordance with new image
    ---------------------------------------------------------------------------
    '''
    image_path = './' + parameters.train_or_test + '/'  + image_name + '.jpg'
    print(image_path)
    assert os.path.exists(image_path)
    im = Image.open(image_path)
    bbox = compute_new_bbox(im.size,bbox,parameters)
    im_crop = im.crop(bbox)
    Expand = parameters.expand
    im_expand = ImageOps.expand(im_crop,(Expand,Expand,Expand,Expand),fill = 'black')
    im_resize = im_expand.resize(parameters.new_size)
    grey = im_resize.convert('L')
    
    #compute the new landmarks according to transformation procedure
    landmarks = load_landmarks(image_name,parameters)
    landmarks = landmarks - (bbox[:2]) + Expand
    landmarks = landmarks * im_resize.size / im_expand.size
    
    return np.array(grey),landmarks.astype(int)



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



def load_landmarks(image_name,parameters):
    '''
    load the landmarks coordinates from .pts file
    ---------------------------------------------------------------------------
    INPUT:
        image_name: a string without extension   ex: 'image_0122'
        parameters: model parameter object
    OUTPUT:
        a numpy array containing all the points
    ---------------------------------------------------------------------------
    '''   

    file_path = './' + parameters.train_or_test + '/' + image_name + '.pts'
    assert os.path.exists(file_path)
    with open(file_path) as f: rows = [rows.strip() for rows in f]
    coords_set = [point.split() for point in rows[rows.index('{') + 1:rows.index('}')]]
    return np.array([list([float(point) for point in coords]) for coords in coords_set])



parameters = model_parameters()


#we test if the function is woring 

im1 = Image.open('./trainset/2857823310_1.jpg')
print(f'the original image is {im1.size}')
box1 = boxes['2857823310_1.jpg']
im2, landmarks = crop_and_resize_image('2857823310_1', box1, parameters )

im2 = Image.fromarray(im2)



#everthing works fine we successufully converted our image into the same size according to there bounding box
#now we can start to train the algorithm 



#now we need to implement the HOG 

import matplotlib.pyplot as plt 

from skimage.feature import hog 
from skimage import data, exposure 

def get_hog(image):
    mat, hog_img = hog(image, visualize=True)
    return mat, hog_img



mat, hog_img = get_hog(im2)

fig, (ax1, ax2)  = plt.subplots(1, 2, figsize = (8, 4), sharex = True, sharey = True)

ax1.imshow(im2, cmap = plt.cm.gray)
ax2.imshow(hog_img, cmap = plt.cm.gray)
plt.show()

#ok now we have hog for each picture. 


#TODO: now we have all picture of the trainset  resize with
#       ground truth, and the we computed the HOG coefficients
#       now we need to train the cascade model. 

