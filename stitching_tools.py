import tifffile
import pandas as pd
import numpy as np
import os
from natsort import natsorted
import cv2
import tifffile as tiff
from skimage.exposure import rescale_intensity

def get_meta(path):
    with tifffile.TiffFile(path) as tif:
        img_meta = tif.imagej_metadata['Info']
    img_meta_list = img_meta.split('DeviceCoordinatesUm":{"')[1:]
    meta_clean = pd.DataFrame(index=['focus','x','y'])
    for i in img_meta_list:
        s = i.split(',"GridRowIndex"')[0]
        depth,x,y,_,_,label = s.split(',')
        depth = depth.split('[')[1][:-1]
        x = x.split('[')[1]
        y = y[:-1]
        label = label.split(':')[1][1:-1]
        meta_clean[label] = [depth,x,y]
    # meta_clean.to_csv(path.split('/')[-1][:-8]+'_meta.csv')
    return meta_clean

def get_size(path_to_tiff):
    meta_data = get_meta(path_to_tiff)
    print('meta-data loaded')
    x_series = meta_data.iloc[1].values.astype(float)
    # y_series = meta_data.iloc[2].values.astype(float)
    height = np.sum(np.abs(np.diff(x_series).astype(int))<13) + 1
    width = int(len(x_series)/height)
    return width,height

def get_size_json(pos_list):
    pos = np.array(pos_list)
    pos_x = pos[:,0]
    height = np.sum(np.abs(np.diff(pos_x).astype(int))<13) + 1
    width = int(len(pos_x)/height)
    return width,height

def map_loc(width,height):

    total = int(width*height)
    new_loc = np.array([])

    if (height%2) == 0: # even number of rows
        # print('even number of rows')
        snake = 0 # natural row scanning direction 
        range_max = total
        for i in range(height):
            if snake == 0: # left to right
                this_row = np.arange(range_max-width,range_max,1)
                new_loc = np.concatenate((new_loc,this_row))
                range_max = range_max - width - 1
                snake += 1
            else:
                this_row = np.arange(range_max,range_max-width,-1)
                new_loc = np.concatenate((new_loc,this_row))
                range_max = range_max - width + 1
                snake -= 1
    else: # odd number of rows 
        # print('odd number of rows')
        snake = 1 # reversed row scanning direction
        range_max = total - 1
        for i in range(height):
            if snake == 1: # right to left
                this_row = np.arange(range_max,range_max-width,-1)
                new_loc = np.concatenate((new_loc,this_row))
                range_max = range_max - width + 1
                snake -= 1
            else:
                this_row = np.arange(range_max-width,range_max,1)
                new_loc = np.concatenate((new_loc,this_row))
                range_max = range_max - width - 1
                snake += 1

    loc_map = {}
    for i in range(total):
        loc_map[i] = int(new_loc[i])
    return loc_map

def stitch_folder(path_to_region_folder,overlap,output_folder,new_fname,padding):
    # get cover slide meta data from the first patch
    files = os.listdir(path_to_region_folder) # files name
    files_clean = []
    for i in files:
        if i.startswith('.'): # remove hidden files begin with .
            pass
        else:
            files_clean.append(i)
    files = files_clean

    files = natsorted(files) # natural sort image order
    files = [files[-1]] + files[:-1] # swap last image to first
    w,h = get_size(os.path.join(path_to_region_folder,files[0]))

    # calculate canvas size
    canvas_w = int(2048*w) - int(overlap*(w-1))
    canvas_h = int(2048*h) - int(overlap*(h-1))
    # initialize a empty array
    stitch_canvas = np.zeros((canvas_h,canvas_w),np.uint16)
    # generate tile location map
    loc_map = map_loc(w,h)

    for j in range(h):
        for i in range(w):
            d_left,d_up = (0,0)
            img = cv2.imread(os.path.join(path_to_region_folder,files[loc_map[int(w*j)+i]]), cv2.IMREAD_ANYDEPTH)
            # overlap shifting
            # horizontal shifting
            if i == 0:
                d_left = 0
            else:
                d_left = int(overlap*i)
            # vertical shifting
            if j == 0:
                d_up = 0
            else:
                d_up = int(overlap*j)
            # filling in canvas with tiles
            try:
                stitch_canvas[int(j*2048)-d_up:int((j+1)*2048)-d_up,int(i*2048)-d_left:int((i+1)*2048)-d_left] = img
            except:
                print("image damaged")
    # anti-distortion
    if padding is True:
        stitch_canvas = anti_distortion(stitch_canvas)
    else:
        pass

    if new_fname is None:
        stitched_file_name = files[0].split('_MMStack_')[0] + '_stitched.tif'
    else:
        animal_id = files[0].split('_')[0] + '_'
        global_num = str(new_fname['num']) + '_'
        obj_num = 'obj'+str(new_fname['obj']) + '_'
        tissue_num = files[0].split(obj_num)[1].split('MMStack_')[0]
        stitched_file_name = animal_id + global_num + obj_num + tissue_num +'stitched.tif'

    if output_folder is None:
        pass
    else:
        tiff.imsave(os.path.join(output_folder,stitched_file_name),stitch_canvas)
        print(stitched_file_name,' saved')
        print('---------------------')
    return (animal_id + global_num + obj_num + tissue_num,stitch_canvas)

def stitch_stack(pos_list,whole_stack,overlap,stitched_path,downsampled_path,padding):
    w,h = get_size_json(pos_list) # pass in pos_list and get size of whole image
    pop_img = int(w * h)
    # calculate canvas size
    canvas_w = int(2048*w) - int(overlap*(w-1))
    canvas_h = int(2048*h) - int(overlap*(h-1))
    # initialize a empty array
    stitch_canvas = np.zeros((canvas_h,canvas_w),np.uint16)
    # generate tile location map
    loc_map = map_loc(w,h)
    # stitch stack image
    for j in range(h):
        for i in range(w):
            d_left,d_up = (0,0)
            img =whole_stack[loc_map[int(w*j)+i]]
            # overlap shifting
            # horizontal shifting
            if i == 0:
                d_left = 0
            else:
                d_left = int(overlap*i)
            # vertical shifting
            if j == 0:
                d_up = 0
            else:
                d_up = int(overlap*j)
            # filling in canvas with tiles
            try:
                stitch_canvas[int(j*2048)-d_up:int((j+1)*2048)-d_up,int(i*2048)-d_left:int((i+1)*2048)-d_left] = img
            except:
                print("image damaged")
    # apply black margin or not
    if padding is True:
        stitch_canvas = anti_distortion(stitch_canvas)
    else:
        pass
    ## save to full resolution to stitched folder
    tiff.imsave(stitched_path,stitch_canvas)
    print(stitched_path,' stitched')
    ## downsample and save to sharptrack folder for green channel
    if downsampled_path is False:
        pass
    else:
        downsample(stitch_canvas,downsampled_path,(1140,800),(50,700))
    return pop_img

def downsample(input_tiff,output_png,size_tuple,contrast_tuple):
    if isinstance(input_tiff,str): # if input is a file path
        # read file to matrix
        img = cv2.imread(input_tiff,cv2.IMREAD_ANYDEPTH)
    else: # if input itself is a image matrix
        img = input_tiff
    # adjust size
    img_down = cv2.resize(img,size_tuple)
    # adjust brightness
    img_down = rescale_intensity(img_down,contrast_tuple)
    # transform to 8 bit
    img_8 = (img_down >> 8).astype('uint8')
    # save downsampled image
    tifffile.imsave(output_png,img_8)
    print(output_png,' downsampled')
    print('-----')

def anti_distortion(input_array):
    h,w = np.shape(input_array)
    ratio = w/h
    if ratio == 1.425:
        output_array = input_array
    elif ratio < 1.425: # portrait, fill left and right
        dest_w = round(h/800 * 1140)
        if (dest_w % 2) == 0:
            pass
        else:
            dest_w += 1
        d_w = int((dest_w - w)/2)
        output_array = np.pad(input_array,((0,0),(d_w,d_w)),'constant',constant_values=0) # pad with absolute black
    else: # landscape, fill top and bottom
        dest_h = round(w/1140 * 800)
        if (dest_h % 2) == 0:
            pass
        else:
            dest_h += 1
        d_h = int((dest_h - h)/2)
        output_array = np.pad(input_array,((d_h,d_h),(0,0)),'constant',constant_values=0) # pad with absolute black
    return output_array


    
