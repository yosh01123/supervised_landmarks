# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 19:02:31 2016
@author: dingning
"""
from __future__ import division
import os
from PIL import Image,ImageOps,ImageDraw
import numpy as np
from scipy import sqrt, pi, arctan2, io
from scipy.ndimage import uniform_filter
from sklearn.linear_model import Lasso,LinearRegression
from math import floor

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
                 train_or_test='train',
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
        
    def show_parameters(self):
        print('use data in trainset or testset:', self.train_or_test)
        print('the number of iterations:',self.N)
        print('the L1 regularization parameter alpha is:',self.alpha)
        print('the new image size used is:',self.new_size)
        print('how much to expand in data preparation:',self.expand)
        print('the rate of expand in modifing the bbox:',self.expand_rate)
        print('the number of divisions of gradient angle used by hog:',self.orientations)
        print('pixels per cell in hog descriptor:',self.pixels_per_cell)
        print('cells per side in hog decriptor:',self.cells_per_side)
        print('cells per bolck in hog descriptor:',self.cells_per_block)
        print('do not use the block when computing the hog:',self.hog_no_block)
        
def train(parameters):
    
    '''
    the standard SDM training function
    ---------------------------------------------------------------------------
    INPUT:
        parameters: the model parameter object
    OUTPUT:
        coef: a numpy array of R
        inte: a numpy array of b
        initials: a numpy array containing a initial landmarks (without ravel)
    ---------------------------------------------------------------------------
    '''
    parameters.train_or_test = 'train'
    
    #show the parameters which will be used
    parameters.show_parameters()
    
    #get the image path list
    image_path_list = get_image_path_list(parameters)
    
    #get the ground truth bounding boxes
    bbox_dict = load_boxes(parameters)
    
    #compute the hog features of ture landmarks
    mark_list = []
    hog_list = []
    grey_list = []
    print('computing the hog features for ture landmarks...........')
    for path in image_path_list:
        grey,mark = crop_and_resize_image(path[:10],bbox_dict[path],parameters)
        hog_list.append(hog(grey,mark,parameters))
        grey_list.append(grey)
        mark_list.append(mark.ravel())
        
    HOG_TRUE = np.array(hog_list)
    MARK_TRUE = np.array(mark_list)
    
    #compute the initial landmarks by mean
    initials = np.mean(MARK_TRUE,axis = 0).astype(int)
    MARK_x = np.array([initials.tolist()] * len(image_path_list))
    initials = initials.reshape(68,2)
    
    #training
    coef = []
    inte = []
    
    for i in range(parameters.N):
        
        print('Iteration: ',i + 1)
        
        #compute the delta x
        MARK_delta = MARK_TRUE - MARK_x
        
        #compute the hog features
        HOG_x = np.zeros_like(HOG_TRUE)
        for j in range(len(image_path_list)):
            if j+1 % 100 == 0: print('already computed',j+1,'features')
            HOG_x[j,:] = hog(grey_list[j],MARK_x[j,:].reshape(68,2),parameters)
        
        #linear regression
        if parameters.alpha == 0:
            reg = LinearRegression(fit_intercept=False)
        else:
            reg = Lasso(alpha=parameters.alpha)
        print('computing the lasso linear regression.......')
        reg.fit(HOG_x,MARK_delta)  
        coef.append(reg.coef_.T)
        inte.append(reg.intercept_.T)
        
        #compute the sparse rate
        sparse_rate = coef[-1][coef[-1]==0].size / coef[-1].size
        print('the sparse rate of',i+1,'th R is:',sparse_rate        )
        
        #compute new landmarks        
        MARK_x = MARK_x + np.matmul(HOG_x, coef[i]) + inte[i]
    
    coef = np.array(coef)
    inte = np.array(inte)
    io.savemat('train_data',{'R':coef,'B':inte,'I':initials})
    
    return coef,inte,initials
    
    

def test_for_one_image(coef,inte,path,bbox,initials,parameters):
    '''
    given the regressors, predicted the landmarks
    ---------------------------------------------------------------------------
    INPUT:
        coef: the R matrix
        inte: the b vector
        path: the image file name with extension
        bbox: the numpy array of bbox
        initials: the numpy array of initials landmarks
        parameters: model parameter object
    OUTPUT:
        mark_x: predicted landmarks
        mark_t: the true landmarks
        MSE: the mean square error of all the iterations
    ---------------------------------------------------------------------------
    '''
    parameters.train_or_test = 'test'                       
                           
    grey,mark_true = crop_and_resize_image(path[:10],bbox,parameters)
    mark_x = initials.astype(int)
    MSE = []
    
    for i in range(coef.shape[0]):
        hog_x = hog(grey,mark_x,parameters)
        mark_x = (mark_x.ravel() + np.matmul(hog_x,coef[i]).astype(float) + inte[i].astype(float)).reshape(68,2)
        MSE.append((abs(mark_x.astype(int) - mark_true)**2).sum() / len(mark_true))
        
    if parameters.demo:
        im = Image.fromarray(grey)
        draw = ImageDraw.Draw(im)
        width = 5
        for i in range(len(mark_x)):
            circle = [mark_x[i,0]-width,mark_x[i,1]-width,mark_x[i,0]+width,mark_x[i,1]+width]
            draw.ellipse(circle,fill = 'red')
    
        im.show()
    
        
    return mark_x.astype(int),mark_true,MSE



def get_image_path_list(parameters):
    '''
    get a list containing all the paths of images in the trainset
    ---------------------------------------------------------------------------
    INPUT:
        parameters: model parameter object
    OUTPUT:
        a list with all the images' paths
    ---------------------------------------------------------------------------
    '''
    folder_path = './trainset/'
    print(os.listdir(folder_path))
    print('already get all the image path.')
    return os.listdir(folder_path)




def load_boxes(parameters):
    '''
    load the ground truth ground truth boxes coordinates from .mat file
    ---------------------------------------------------------------------------
    INPUT:
        parameters: model parameter object
    OUTPUT:
        a dict with all the ground truth bounding boxes coordinates
        key: a string of filename    ex: 'image_0122.png'
        value: a numpy array of boungding boxes
    ---------------------------------------------------------------------------
    '''
    file_path = 'data/bounding_boxes/bounding_boxes_lfpw_' + parameters.train_or_test + 'set.mat'
    assert os.path.exists(file_path)
    x = io.loadmat(file_path)['bounding_boxes'][0]
    x = [x[0][0] for x in x]
    print('loading ground truth bboxes....................')
    return {x[i][0][0]:x[i][1][0] for i in range(len(x))}
    
    
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
    image_path = 'data/' + parameters.train_or_test + 'set/png/' + image_name + '.png'
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


def hog(image, xys, parameters):
    '''
    Given a grey image in numpy array and a vector of sequence of coordinates,
    return the ndarray of hog feature vectors extract from given locations
    ---------------------------------------------------------------------------
    INPUT:
        image: grey image, numpy array, 8-bit
        xys: coordinates, numpy array, float
        parameters: model parameter object
    OUTPUT:
        features: ndarray of all the features extracted from locations in xy
    ---------------------------------------------------------------------------
    '''
    image = np.atleast_2d(image)    
    if image.ndim > 3: raise ValueError("Currently only supports grey-level images")


    #normalisation
    image = sqrt(image)
    
    
    #compute the gradients of the input grey image
    gx = np.zeros(image.shape)
    gy = np.zeros(image.shape)
    gx[:, :-1] = np.diff(image, n=1, axis=1)
    gy[:-1, :] = np.diff(image, n=1, axis=0)
    
    
    #compute the magnitude and orientation of gradients
    magnitude = sqrt(gx ** 2 + gy ** 2)
    orientation = arctan2(gy, (gx + 1e-15)) * (180 / pi) + 180

    #just for convinients, make the variables shorter
    r = parameters.pixels_per_cell * parameters.cells_per_side
    pc = parameters.pixels_per_cell
    eps = 1e-5
    
    #compute the orientation histogram
    orientation_histogram = np.zeros((len(xys), 
                                      parameters.cells_per_side*2, 
                                      parameters.cells_per_side*2, 
                                      parameters.orientations))    
    for j in range(len(xys)):        
        x, y = xys[j].astype(int)
        for i in range(parameters.orientations):
            # classify the orientation of the gradients
            temp_ori = np.where(orientation <= 180 / parameters.orientations * (i + 1) * 2,
                                orientation, 0)
            temp_ori = np.where(orientation > 180 / parameters.orientations * i * 2,
                                temp_ori, 0)
            # select magnitudes for those orientations
            cond2 = temp_ori > 0
            temp_mag = np.where(cond2, magnitude, 0)
        
            orientation_histogram[j,:,:,i] = uniform_filter(temp_mag, size=pc)[x-r+pc/2:x+r:pc, y-r+pc/2:y+r:pc].T
        
        if parameters.hog_no_block:
            orientation_histogram[j] /= sqrt(orientation_histogram[j].sum()**2 + eps)
    
    if parameters.hog_no_block: return orientation_histogram.ravel()
        
    
    #compute the block normalization
    n_blocks = parameters.cells_per_side * 2 - parameters.cells_per_block + 1
    cb = parameters.cells_per_block
    normalised_blocks = np.zeros((len(xys), n_blocks, n_blocks, cb, cb, parameters.orientations))
    
    for i in range(len(xys)):
        for x in range(n_blocks):
            for y in range(n_blocks):
                block = orientation_histogram[i,x:x + cb, y:y + cb, :]            
                normalised_blocks[i, x, y, :] = block / sqrt(block.sum() ** 2 + eps)
    
    return normalised_blocks.ravel()


def test_after_run_main(n):
    x,t,m = test_for_one_image(R,B,image_path_list[n],bbox_dict[image_path_list[n]],I,parameters)
    return x,t,m

if __name__ == '__main__':
    parameters = model_parameters()
    if os.path.exists('train_data.mat'):
        data = io.loadmat('train_data.mat')
        R = data['R']
        B = data['B']
        I = data['I']
    else:
        R,B,I = train(parameters)
    parameters.train_or_test = 'test'
    image_path_list = get_image_path_list(parameters)
    bbox_dict = load_boxes(parameters)