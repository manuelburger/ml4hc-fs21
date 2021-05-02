# Custom helper functions for loading and manipulating data

# Load libraries
import os
import numpy as np
from tqdm.notebook import tqdm
from sklearn.utils import resample
from skimage.measure import block_reduce
from imblearn.under_sampling import RandomUnderSampler

import nibabel as nib
from joblib import Parallel, delayed, parallel_backend
import dask
import sys


def img_loader(path, img_type, file, test=False):
    """
    Load a single input image given its path (directory), type {imagesTr, labelsTr, imagesTs}
    and the file name
    """
    if test:
        return (file, nib.load(os.path.join(path, img_type, file)).get_fdata())
    else:
        return nib.load(os.path.join(path, img_type, file)).get_fdata()


def img_loader_scaling(path, img_type, file, scaling=False, wid=.7, dep=5., target_im_size = 512, test=False):
    """
    Loads scans as well as their meta data
    Scales scans to target voxel volume with pixel width 'wid' and voxel depth 'dep'; 
    default values .7 and 5. corresponds to the modal values
    """
    scan = nib.load(os.path.join(path, img_type, file))  # loads the scan
    scan_head = scan.header                              # header of scan with metadata
    scan_arr = scan.get_fdata()                          # data array of the scan
    scan_dim = scan_head.get_data_shape()                # array dimension of scan
    vxl_dim = scan_head.get_zooms()                      # voxel dimension

    if scaling:
        factor = np.array([wid, wid, dep])/vxl_dim  # scaling factors
        scan_arr = zoom(scan_arr, factor)  # scale and interpolate scan
        if img_type == 'labelsTr': scan_arr = np.round(scan_arr) # if label mask then round to get of artifacts
        x, y, _ = scan_arr.shape           
        if x > target_im_size:  # crop the image if necessary
            start_x = x//2-(target_im_size//2)
            start_y = y//2-(target_im_size//2)
            scan_arr = scan_arr[start_x:(start_x + target_im_size), start_y:(start_y + target_im_size), :]
        elif x < target_im_size:                    # pad the image
            pad_l = (target_im_size-x)//2           # pad on the left
            pad_r = pad_l + (target_im_size-x) % 2  # pad on the right
            pad_const = 0 if img_type=='labelsTr' else -1024.  # differentiate padding for images and labels
            scan_arr = np.pad(scan_arr, ((pad_l, pad_r), (pad_l, pad_r), (0, 0)), 
                              constant_values=[(pad_const, pad_const), (pad_const, pad_const), (0,0)])
    if test:
        return (file, scan_arr)
    else:
        return scan_arr


def read_training_data_parallel(data_path, njobs=8, frac=None, load_scaled=False):
    """Read training data"""
    imgs = []
    lbls = []

    # Load scaled images or standard
    if load_scaled:
        loader = img_loader_scaling
    else:
        loader = img_loader
          
    files = [f for f in os.listdir(os.path.join(data_path, 'imagesTr')) if not ('.DS_Store' in f or '._' in f)]
    
    # Load only a subset of the data
    if frac:
        files = files[:int(len(files)*frac)]
    
    # Parallel load of the training images
    with parallel_backend('multiprocessing'):
        imgs = Parallel(n_jobs=njobs)(delayed(loader)(data_path, 'imagesTr', f) for f in tqdm(files))
    
    # Parallel load of the training labels
    with parallel_backend('multiprocessing'):
        lbls = Parallel(n_jobs=njobs)(delayed(loader)(data_path, 'labelsTr', f) for f in tqdm(files))
        
        
    return imgs, lbls

def read_testing_data_parallel(data_path, njobs=8, frac=None, load_scaled=False, get_names=False):
    '''Read testing data'''
    imgs = []
    
    # Load scaled or standard
    if load_scaled:
        loader = img_loader_scaling
    else:
        loader = img_loader
    
    files = [f for f in os.listdir(os.path.join(data_path, 'imagesTs')) if not ('.DS_Store' in f or '._' in f)]
    
    # Load only a fraction of the data
    if frac:
        files = files[:int(len(files)*frac)]
     
    # Parallel load of test images
    with parallel_backend('multiprocessing'):
        imgs = Parallel(n_jobs=njobs)(delayed(loader)(data_path, 'imagesTs', f, test=get_names) for f in tqdm(files))
    
    return imgs


def read_training_data_and_discard(
        data_path, frac=None, discard_empty=0.9,
        weak_thresh=1000, discard_weak=0.2,
        reduce_by=0):
    '''Read training data and discard some samples below a threshold of cancerous pixels'''
    imgs = []
    lbls = []
    files = [f for f in os.listdir(os.path.join(data_path, 'imagesTr')) if not (
        '.DS_Store' in f or '._' in f)]
    if frac:
        files = files[:int(len(files)*frac)]
    for f in tqdm(files, total=len(files)):
        img = nib.load(os.path.join(
            data_path, 'imagesTr', f)).get_fdata().astype(np.float32)
        lbl = nib.load(os.path.join(
            data_path, 'labelsTr', f)).get_fdata().astype(np.float32)

        if reduce_by > 0:
            img = block_reduce(img, (reduce_by, reduce_by, 1), np.max)
            lbl = block_reduce(lbl, (reduce_by, reduce_by, 1), np.max)
        s = np.sum(lbl > 0, axis=(0, 1))
        empty = s == 0
        empty_places = np.nonzero(empty)[0]
        empty[np.random.choice(
            empty_places, replace=False,
            size=int(discard_empty * len(empty_places)))] = False
        weak = (s > 0) * (s < weak_thresh)
        weak_places = np.nonzero(weak)[0]
        weak[np.random.choice(
            weak_places, replace=False,
            size=int(discard_weak * len(weak_places)))] = False
        strong = s >= weak_thresh
        imgs.append(img[:, :, empty + weak + strong])
        lbls.append(lbl[:, :, empty + weak + strong])
        del img
        del lbl
    return imgs, lbls


def convert_depth_to_imgs_keras(images, neighbors=0, njobs=8, print_shape=True):
    """
    Convert depth to individual images considering neighboring channels
    """
    # Clamping helper function
    def clamp(n, minn, maxn):
        return max(min(maxn, n), minn)
    
    # Transform a single 3D image into separate layers
    # where we consider some number of neighbors for each layer
    def transform(image):
        out_imgs = []
        for i in range(image.shape[2]):
            new_img = []
            for j in range(i-neighbors, i+neighbors+1):
                i_clamp = clamp(j, 0, image.shape[2]-1)
                new_img.append(image[:,:,i_clamp])
            out_imgs.append(np.moveaxis(np.array(new_img), 0, -1))
        return out_imgs
        
    
    # Convert images
    res_imgs = []
    if neighbors == 0:
        for img in images:
            for i in range(img.shape[2]):
                res_imgs.append(img[:,:,i].reshape((img.shape[0], img.shape[1], 1)))
    else:
        for img in images:
            new = dask.delayed(transform)(img)
            res_imgs.append(new)
                
        res_imgs = dask.compute(*res_imgs, njobs=8)
        res_imgs = [i for sublist in res_imgs for i in sublist]
    
    # Print output shape
    if print_shape:
        print("Data shape single img: ", res_imgs[0].shape)
    
    return res_imgs


def get_cancerous_layers(img):
    """
    Get indeces of cancerous layers in input image
    """
    out = []
        
    for i in range(img.shape[2]):
        if np.sum(img[:,:,i]) > 0:
            out.append(i)
    return out


def keep_only_cancerous(imgs, lbls):
    """
    Keeps only images with cancerous tissue,
     considers a list of images (depths separated)
    """
    
    out_lbls = []
    out_imgs = []
    
    for img, lbl in zip(imgs, lbls):
        if np.sum(lbl) > 0:
            out_lbls.append(lbl)
            out_imgs.append(img)
            
    return out_imgs, out_lbls



def downsample(imgs, lbls, frac=None, strategy=None):
    """
    Downsamples the majority class, either give a direct strategy
    for RandomUnderSampler from imblearn
    or give a fractional value for the desired
    #minority/#majority ratio
    
    lbls are considered to be segmentation masks
    """

    # Compute segmentation masks containing positive labels
    data = np.array(list(range(len(lbls)))).reshape(-1, 1)
    classes = [1 if np.sum(lbl) > 0 else 0 for lbl in lbls]
    
    if not strategy:
        strategy = frac if frac else 'majority'

    rus = RandomUnderSampler(
        random_state=42,
        replacement=False,
        sampling_strategy=strategy
    )
    
    resampled, _ = rus.fit_resample(data, classes)
    
    resampled = sorted(resampled.ravel())

    imgs_res, lbls_res = [], []
    for i in resampled:
        imgs_res.append(imgs[i])
        lbls_res.append(lbls[i])
    
    count = sum([1 for l in lbls_res if np.sum(l) > 0.0])
    print("Downsampled, returning {} images".format(len(lbls_res)))
    print("Ratio of images with canc. tissue {:.2f}%".format(count/len(lbls_res) * 100))
    
    return list(imgs_res), list(lbls_res)
