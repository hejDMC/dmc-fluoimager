import json
from pycromanager import Acquisition
from pycromanager import Core
from pycromanager import Studio
import numpy as np
import pandas as pd
import tifffile
from stitching_tools import *
import os
import cv2
import matplotlib.pyplot as plt
from roi_tools import *


class Scanner():
    def __init__(self):
        self.core = Core()
        self.studio = Studio()
        self.z_default = 24200
        self.filterCube = '5-TX2' # '5-TX2'
        self.pl_json = 'position_list_whole5x.json'
        self.s_shaped_scan = False
        self.filter_name = {'green':'3-L5',
                            'cy3':'5-TX2',
                            'n3':'4-N3',
                            'dapi':'2-A4',
                            'cy5':'6-Y5'}
        self.magnif = 10 # default objecitve lens mag
        # filter parameters for ROIs
        self.roiarea_min = 50
        self.roiarea_max = 2000
        self.roi_w_min = 5
        self.roi_h_max = 1000
        # create empty list for merged subregions
        self.to_delete = []
        self.focus_channel = "cy3"

    def createFolder(self):
        if not os.path.exists(os.path.join(self.root, self.animal_id)):
            os.mkdir(os.path.join(self.root, self.animal_id))
        else:
            pass

    def scan_5x(self):
        # create raw folder if there's none
        if not os.path.exists(os.path.join(self.root,self.animal_id,'raw')):
            os.mkdir(os.path.join(self.root,self.animal_id,'raw'))
        # create obj folder
        os.mkdir(os.path.join(self.root,self.animal_id,'raw',self.obj_id))
        # load general whole5x position list
        with open(self.pl_json,'r') as data:
            pos = json.load(data)
        # generate events
        events = []
        for tile in range(len(pos['pos'])):
            evt = {
                'axes': {'pos': int(tile)}, # axes unique id should be int
                'exposure': 10, # here change exposure time to 10 ms
                'x': float(pos['pos'][tile][0]),
                'y': float(pos['pos'][tile][1]),
                'z': float(self.z_default),
                'keep_shutter_open': False,
                }
            events.append(evt)
        # switch to default filtercube, 5x objective
        self.core.set_property('IL-Turret','Label', self.filterCube)
        self.core.set_property('ObjectiveTurret','Label','2-5x 0.15na')
        # acquire image stack
        with Acquisition(directory=os.path.join(self.root,self.animal_id,'raw',self.obj_id),name=self.obj_id+"_meta") as acq:
            acq.acquire(events)
        print('5X scan finished')
        print('Stitching 5X image, please wait...')
        self.stitch_5x()


    def stitch_5x(self):
        # load position list
        with open(self.pl_json,'r') as data:
            img_meta = json.load(data)
        # get tile-size
        meta_np = np.array(img_meta['pos']).astype(float) # convert 2D string array to float
        x_series = meta_np[:,0]
        h = np.sum(np.abs(np.diff(x_series).astype(int))<13) + 1
        w = int(len(x_series)/h)
        # calculate whole image size
        canvas_w = int(2048*w) - int(205*(w-1))
        canvas_h = int(2048*h) - int(205*(h-1))
        loc_map = map_loc(w,h)
        self.stitch_canvas = np.zeros((canvas_h,canvas_w),np.uint16) # create stitched img here
        img_stack = tifffile.imread(os.path.join(self.root,self.animal_id,'raw',self.obj_id,self.obj_id+'_meta_1',self.obj_id+'_meta_NDTiffStack.tif'))
        tile = 0
        for j in range(h):
            for i in range(w):
                d_left,d_up = (0,0)
                img = img_stack[loc_map[tile]]
                # overlap shifting
                # horizontal shifting
                if i == 0:
                    d_left = 0
                else:
                    d_left = int(205*i)
                # vertical shifting
                if j == 0:
                    d_up = 0
                else:
                    d_up = int(205*j)
                # filling in canvas with tiles
                try:
                    self.stitch_canvas[int(j*2048)-d_up:int((j+1)*2048)-d_up,int(i*2048)-d_left:int((i+1)*2048)-d_left] = img
                except:
                    print("image damaged")
                tile += 1
        self.stitched_tif_path = os.path.join(self.root,self.animal_id,'raw',self.obj_id,self.obj_id+'_meta_1','stitched.tif')
        tiff.imwrite(self.stitched_tif_path,self.stitch_canvas)
        print('5X stitched.tif saved')

    def roi_select_tool(self,threshold=None):

        def click_event(event, x, y,flags,param):
            # when left click
            if event == cv2.EVENT_LBUTTONDOWN:
                # add current cursor position to list
                click_record.append([x,y])
                # draw a dot for visual
                cv2.circle(img_color_temp, (x,y), radius=10, color=(0, 0, 255), thickness=-1)
                cv2.imshow(window_name, img_color_temp)
                # depth 16 to 8
        if hasattr(self,"stitch_canvas"):
            if self.stitched_tif_path == os.path.join(self.root,self.animal_id,'raw',self.obj_id,self.obj_id+'_meta_1','stitched.tif'):
                pass
        else:
            self.stitch_canvas = tifffile.imread(os.path.join(self.root,self.animal_id,'raw',self.obj_id,self.obj_id+'_meta_1','stitched.tif'))
            self.stitched_tif_path = os.path.join(self.root,self.animal_id,'raw',self.obj_id,self.obj_id+'_meta_1','stitched.tif')
        # downsample by 8X
        img = cv2.resize(self.stitch_canvas,(int(self.stitch_canvas.shape[1]/8),int(self.stitch_canvas.shape[0]/8)))
        # rescale intensity @ 16 bit
        hist = np.histogram(img,bins=np.arange(65536))
        low_cutoff = 0.05
        high_cutoof = 0.95
        pct5 = np.where(np.add.accumulate(hist[0]) > np.add.accumulate(hist[0])[-1]*low_cutoff)[0][0]
        pct95 = np.where(np.add.accumulate(hist[0]) > np.add.accumulate(hist[0])[-1]*high_cutoof)[0][0]
        intensity_range = [pct5,pct95]
        img = (65535/(intensity_range[1]-intensity_range[0]) * (img-intensity_range[0])).astype(np.uint16)
        # convert to 8 bit image
        img = (img >> 8).astype(np.uint8)
        # gaussian filter
        img = cv2.GaussianBlur(img,(31,31),0)
        # OTSU threshold
        if threshold == None:
            _,img_bin = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        else:
            _,img_bin = cv2.threshold(img, threshold[0], threshold[1], cv2.THRESH_BINARY)
        # downsample further to finally 16X
        img_bin = cv2.resize(img_bin,(int(img_bin.shape[1]/2),int(img_bin.shape[0]/2)))

        img_color = cv2.cvtColor(img_bin,cv2.COLOR_GRAY2RGB)
        self.img_clean = img_color.copy()
        img_color_temp = self.img_clean.copy()
        window_name = 'Please select ROI, Click to add point, Press "n" to move to the next ROI, Press "q" to quit.'
        self.regions = []
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # resize window to smaller than FHD screen
        cv2.resizeWindow(window_name, 950, 450)
        # move window to top left of screen
        cv2.moveWindow(window_name, 0, 0)
        cv2.imshow(window_name,self.img_clean)
        # listen for input
        click_record = []
        region_num = 0
        print('define region',region_num)
        cv2.setMouseCallback(window_name, click_event)
        while True:
            k = cv2.waitKey(0)
            # press q to quit
            if k & 0xFF == ord('q'):
                print('selected regions saved')
                break
            # press n to create a new region
            elif k & 0xFF == ord('n'):
                # calculate bounding box and append to region list
                xs = np.array(click_record)[:,0]
                ys = np.array(click_record)[:,1]
                x = np.min(xs)
                y = np.min(ys)
                w = np.max(xs) - x
                h = np.max(ys) - y
                self.regions.append((x,y,w,h))
                print(x,y,w,h)
                print('----------')
                cv2.rectangle(img_color,(x,y),(x+w,y+h),(0,0,255),9)
                cv2.putText(img_color, str(region_num), (x,y+h), 
            cv2.FONT_HERSHEY_SIMPLEX, 3, (0,165,255), 9, cv2.LINE_AA)
                # clear cache
                click_record = []
                cv2.imshow(window_name,img_color)
                # clear red dots
                img_color_temp = img_color.copy()
                region_num += 1
                print('define region',region_num)
        cv2.destroyAllWindows()

        df_regions = pd.DataFrame()
        for region in range(len(self.regions)):
            df_regions['region_'+str(region)] = self.regions[region]

        df_regions.to_csv(os.path.join(self.root,self.animal_id,'raw',self.obj_id,self.obj_id+'_meta_1','stitched_regions.csv'),index=0)
        tifffile.imwrite(os.path.join(self.root,self.animal_id,'raw',self.obj_id,self.obj_id+'_meta_1','stitched_regions.tif'),img_color)
        print('Mapping to XYStage')
        x_stage,y_stage = 1222,0
        k = -1.28
        x_pixel = int(5785*8)-1024 # (25 tiles in x) lower right X pixel center
        y_pixel = int(2559*8)-1024 # (11 tiles in y) lower right Y pixel center
        b_x = x_pixel*(-k) + x_stage
        b_y = y_pixel*(-k) + y_stage
        self.xystage_df = pd.DataFrame()
        for col in df_regions.columns:
            x,y,w,h = df_regions[col].to_list()
            # rescale back to original pixel position
            x,y,w,h = int(x*16),int(y*16),int(w*16),int(h*16)
            # translate coordinates
            x = x * k + b_x
            y = y * k + b_y
            w,h = w*(-k), h*(-k)
            # save boundary xystage coordinates
            l,r,u,d = (x,y-0.5*h),(x-w,y-0.5*h),(x-0.5*w,y),(x-0.5*w,y-h)
            save_to_df_raw = [l[0],l[1],r[0],r[1],u[0],u[1],d[0],d[1]]
            save_to_df = []
            # keep inside motor range
            for coord in save_to_df_raw:
                if coord < 0:
                    save_to_df.append(0)
                else:
                    save_to_df.append(coord)
            self.xystage_df[col] = save_to_df
        # save left, right, up, down boundary xystages
        self.xystage_df.to_csv(os.path.join(self.root,self.animal_id,'raw',self.obj_id,self.obj_id+'_meta_1','stitched_boundary_xystage.csv'),index=0)
        print('ROI boundary XYStage saved')
        if self.magnif == 10:
            stepsize = 0.9 * 2048 * 0.64
        elif self.magnif == 20:
            stepsize = 0.9 * 2048 * 0.32
        # calculate position list
        self.position_list_json = {}

        for col in self.xystage_df.columns:
            # clear position list cache
            position_list_slice = []
            stage = self.xystage_df[col]
            # get number of tiles in x and y direction
            tile_xn = np.ceil(np.abs(stage[2] - stage[0])/stepsize).astype(int) + 1 # fence post problem
            tile_yn = np.ceil(np.abs(stage[7] - stage[5])/stepsize).astype(int) + 1 # fence post problem
            # direction in X-axis
            direction = 1
            hold = 1
            now_x = stage[2]
            now_y = stage[7] - stepsize
            # appending position list
            for i in range(tile_yn):
                for j in range(tile_xn):
                    if hold == 1:
                        now_y += stepsize
                        position_list_slice.append([now_x,now_y])
                        hold -= 1
                    else:
                        if direction > 0: # x increase
                            now_x += stepsize
                            position_list_slice.append([now_x,now_y])
                        else: # x decrease
                            now_x -= stepsize
                            position_list_slice.append([now_x,now_y])
                direction *= (-1)
                hold += 1
            # save to dict
            self.position_list_json[col] = position_list_slice
            # save to json
        with open(os.path.join(self.root,self.animal_id,'raw',self.obj_id,self.obj_id+'_meta_1','regions_pos.json'), 'w') as f:
            json.dump(self.position_list_json, f)
        print('XY PositionList saved')

    

    def roi_detection(self,threshold=None):
        # depth 16 to 8
        if hasattr(self,"stitch_canvas"):
            if self.stitched_tif_path == os.path.join(self.root,self.animal_id,'raw',self.obj_id,self.obj_id+'_meta_1','stitched.tif'):
                pass
            else:
                self.stitch_canvas = tifffile.imread(os.path.join(self.root,self.animal_id,'raw',self.obj_id,self.obj_id+'_meta_1','stitched.tif'))
                self.stitched_tif_path = os.path.join(self.root,self.animal_id,'raw',self.obj_id,self.obj_id+'_meta_1','stitched.tif')
        else:
            self.stitch_canvas = tifffile.imread(os.path.join(self.root,self.animal_id,'raw',self.obj_id,self.obj_id+'_meta_1','stitched.tif'))
            self.stitched_tif_path = os.path.join(self.root,self.animal_id,'raw',self.obj_id,self.obj_id+'_meta_1','stitched.tif')
        # downsample by 8X
        img = cv2.resize(self.stitch_canvas,(int(self.stitch_canvas.shape[1]/8),int(self.stitch_canvas.shape[0]/8)))
        # rescale intensity @ 16 bit
        hist = np.histogram(img,bins=np.arange(65536))
        low_cutoff = 0.05
        high_cutoof = 0.95
        pct5 = np.where(np.add.accumulate(hist[0]) > np.add.accumulate(hist[0])[-1]*low_cutoff)[0][0]
        pct95 = np.where(np.add.accumulate(hist[0]) > np.add.accumulate(hist[0])[-1]*high_cutoof)[0][0]
        intensity_range = [pct5,pct95]
        img = (65535/(intensity_range[1]-intensity_range[0]) * (img-intensity_range[0])).astype(np.uint16)
        # convert to 8 bit image
        img = (img >> 8).astype(np.uint8)
        # gaussian filter
        img = cv2.GaussianBlur(img,(31,31),0)
        if threshold is None:
            # OTSU threshold
            _,img_bin = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        else:
            _,img_bin = cv2.threshold(img, threshold[0], threshold[1], cv2.THRESH_BINARY)
        # downsample further to finally 16X
        img_bin = cv2.resize(img_bin,(int(img_bin.shape[1]/2),int(img_bin.shape[0]/2)))
        # color image for display
        img_color = cv2.cvtColor(img_bin,cv2.COLOR_GRAY2RGB)
        self.img_clean = img_color.copy()
        # find regions
        contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.regions = []
        for cnt,hi in zip(contours,hierarchy[0]):
            area = cv2.contourArea(cnt)
            # filter by region area
            if (area > self.roiarea_min)&(area < self.roiarea_max):
                x,y,w,h = cv2.boundingRect(cnt)
                # filter by region width and height
                if (w < self.roi_w_min) or (h > self.roi_h_max):
                    pass
                else:
                    # if hi[-1]==-1:
                    self.regions.append((x,y,w,h))

        # draw regions on color image
        for region in range(len(self.regions)):
            x,y,w,h = self.regions[region]
            # draw box
            cv2.rectangle(img_color,(x,y),(x+w,y+h),(255,0,0),9)
            cv2.putText(img_color, str(region), (x,y+h), 
            cv2.FONT_HERSHEY_SIMPLEX, 3, (255,165,0), 9, cv2.LINE_AA)
        print('ROI detected, combine and remove to continue.')
        plt.figure(figsize=(12,5))
        plt.imshow(img_color)
    
    def merge(self,merge_list):
        self.regions = merge_roi(self.regions,merge_list)
        # add subregions to delete later
        self.to_delete += merge_list
    
    def delete(self,delete_list):
        # add to-delete subregions to delete list
        self.to_delete += delete_list
    

    def confirm_roi(self):
        # execute delete
        self.regions = del_roi(self.regions,self.to_delete)
        # release to-delete queue
        self.to_delete = []

        # order regions by location
        self.regions = sort_region(self.regions,self.s_shaped_scan)
        # visualize again and save dataframe
        df_regions = pd.DataFrame()
        img_color = self.img_clean.copy()

        for region in range(len(self.regions)):
            df_regions['region_'+str(region)] = self.regions[region]
            x,y,w,h = self.regions[region]
            # draw box
            cv2.rectangle(img_color,(x,y),(x+w,y+h),(255,0,0),9)
            # write region ID
            cv2.putText(img_color, str(region), (x,y+h), 
            cv2.FONT_HERSHEY_SIMPLEX, 3, (255,165,0), 9, cv2.LINE_AA)

        plt.figure(figsize=(12,5))
        plt.imshow(img_color)

        df_regions.to_csv(os.path.join(self.root,self.animal_id,'raw',self.obj_id,self.obj_id+'_meta_1','stitched_regions.csv'),index=0)
        tifffile.imwrite(os.path.join(self.root,self.animal_id,'raw',self.obj_id,self.obj_id+'_meta_1','stitched_regions.tif'),img_color)
        # map region coords to xystage
        print('Mapping to XYStage')
        # no need to change if used whole5x json as position list
        x_stage,y_stage = 1222,0
        # transform from 5x stitched
        k = -1.28
        x_pixel = int(5785*8)-1024 # (25 tiles in x) lower right X pixel center
        y_pixel = int(2559*8)-1024 # (11 tiles in y) lower right Y pixel center
        b_x = x_pixel*(-k) + x_stage
        b_y = y_pixel*(-k) + y_stage
        self.xystage_df = pd.DataFrame()
        for col in df_regions.columns:
            x,y,w,h = df_regions[col].to_list()
            # rescale back to original pixel position
            x,y,w,h = int(x*16),int(y*16),int(w*16),int(h*16) 
            # translate coordinates
            x = x * k + b_x
            y = y * k + b_y
            w,h = w*(-k), h*(-k)
            # save boundary xystage coordinates
            l,r,u,d = (x,y-0.5*h),(x-w,y-0.5*h),(x-0.5*w,y),(x-0.5*w,y-h)
            save_to_df_raw = [l[0],l[1],r[0],r[1],u[0],u[1],d[0],d[1]]
            save_to_df = []
            # keep inside motor range
            for coord in save_to_df_raw:
                if coord < 0:
                    save_to_df.append(0)
                else:
                    save_to_df.append(coord)
            self.xystage_df[col] = save_to_df
        # save left, right, up, down boundary xystages
        self.xystage_df.to_csv(os.path.join(self.root,self.animal_id,'raw',self.obj_id,self.obj_id+'_meta_1','stitched_boundary_xystage.csv'),index=0)
        print('ROI boundary XYStage saved')

        # calculate motor step size
        if self.magnif == 10:
        # 10% overlap, 2048*2048 image, 0.64 pixelsize@10X objective
            stepsize = 0.9 * 2048 * 0.64
        elif self.magnif == 20:
        # 10% overlap, 2048*2048 image, 0.32 pixelsize@20X objective
            stepsize = 0.9 * 2048 * 0.32
        
        # calculate position list
        self.position_list_json = {}

        for col in self.xystage_df.columns:
            # clear position list cache
            position_list_slice = []
            stage = self.xystage_df[col]
            # get number of tiles in x and y direction
            tile_xn = np.ceil(np.abs(stage[2] - stage[0])/stepsize).astype(int) + 1 # fence post problem
            tile_yn = np.ceil(np.abs(stage[7] - stage[5])/stepsize).astype(int) + 1 # fence post problem
            # direction in X-axis
            direction = 1
            hold = 1
            now_x = stage[2]
            now_y = stage[7] - stepsize
            # appending position list
            for i in range(tile_yn):
                for j in range(tile_xn):
                    if hold == 1:
                        now_y += stepsize
                        position_list_slice.append([now_x,now_y])
                        hold -= 1
                    else:
                        if direction > 0: # x increase
                            now_x += stepsize
                            position_list_slice.append([now_x,now_y])
                        else: # x decrease
                            now_x -= stepsize
                            position_list_slice.append([now_x,now_y])
                direction *= (-1)
                hold += 1
            # save to dict
            self.position_list_json[col] = position_list_slice
            # save to json

        with open(os.path.join(self.root,self.animal_id,'raw',self.obj_id,self.obj_id+'_meta_1','regions_pos.json'), 'w') as f:
            json.dump(self.position_list_json, f)
        print('XY PositionList saved')

    def get_pl_focus(self,focus_channel):
        # save focus channel for correction later
        self.focus_channel = focus_channel
        filter_cube = self.focus_channel
        # set focus
        self.core.set_property('IL-Turret','Label',self.filter_name[filter_cube])
        if self.magnif == 10:
            self.core.set_property('ObjectiveTurret','Label','3-10x 0.4na')
        elif self.magnif == 20:
            self.core.set_property('ObjectiveTurret','Label','4-20x 0.7na')
        # call snap live manager
        slm = self.studio.get_snap_live_manager()
        if hasattr(self,'xystage_df'):
            df_boundary = self.xystage_df
        else:
            try:
                self.xystage_df = pd.read_csv(os.path.join(self.root,self.animal_id,'raw',self.obj_id,self.obj_id+'_meta_1','stitched_boundary_xystage.csv'))
                df_boundary = self.xystage_df
            except FileNotFoundError:
                print('ROIs not found, please go back to set ROIs')
        if hasattr(self,'position_list_json'):
            pass
        else:
            try:
                with open(os.path.join(self.root,self.animal_id,'raw',self.obj_id,self.obj_id+'_meta_1','regions_pos.json'), 'r') as f:
                    self.position_list_json = json.load(f)
            except FileNotFoundError:
                print('ROIs not found, please go back to set ROIs')

        if os.path.exists(os.path.join(self.root,self.animal_id,'raw',self.obj_id,self.obj_id+'_meta_1','regions_pos_withz.json')):
            print('Focus data found, loading now...')
            with open(os.path.join(self.root,self.animal_id,'raw',self.obj_id,self.obj_id+'_meta_1','regions_pos_withz.json'), 'r') as f:
                self.pos_z = json.load(f)
            print('3-point focus data restored, ready for acquisition.')
        
        else:
            df_z_info = pd.DataFrame()
            for col in df_boundary.columns:
                lx,ly,rx,ry,ux,uy,dx,dy = df_boundary[col]
                # # calculate tl,tr,bm AF coordinates
                tl = ((lx+ux)/2,(ly+uy)/2)
                tr = ((rx+ux)/2,(ry+uy)/2)
                bm = (dx,(ry+dy)/2)
                # goto top-left acq z-value
                self.core.set_xy_position('XYStage',tl[0],tl[1])
                self.core.wait_for_device('XYStage') # try to wait until motor finish
                self.core.set_position(24200)
                self.core.wait_for_device('FocusDrive') # try to wait until motor finish
                # manual focus here
                slm.set_live_mode_on(True)
                z_temp = 24200
                while slm.is_live_mode_on():
                    z_temp = self.core.get_position()
                tl_z = z_temp
                print(col,'Top-left: [',str(tl[0]),str(tl[1]),str(tl_z),']')
                # goto top-right acq z-value
                self.core.set_xy_position('XYStage',tr[0],tr[1])
                self.core.wait_for_device('XYStage') # try to wait until motor finish
                self.core.set_position(24200)
                self.core.wait_for_device('FocusDrive')
                # manual focus here
                slm.set_live_mode_on(True)
                z_temp = 24200
                while slm.is_live_mode_on():
                    z_temp = self.core.get_position()
                tr_z = z_temp
                print(col,'Top-right: [',str(tr[0]),str(tr[1]),str(tr_z),']')
                # goto bottom-mid acq z-value
                self.core.set_xy_position('XYStage',bm[0],bm[1])
                self.core.wait_for_device('XYStage') # try to wait until motor finish
                self.core.set_position(24200)
                self.core.wait_for_device('FocusDrive')
                # manual focus here
                slm.set_live_mode_on(True)
                z_temp = 24200
                while slm.is_live_mode_on():
                    z_temp = self.core.get_position()
                bm_z = z_temp
                print(col,'Bottom-mid: [',str(bm[0]),str(bm[1]),str(bm_z),']')
                # save to z-info dataframe
                df_z_info[col] = [tl[0],tl[1],tl_z,tr[0],tr[1],tr_z,bm[0],bm[1],bm_z]
                print('------')

                
            df_z_info.to_csv(os.path.join(self.root,self.animal_id,'raw',self.obj_id,self.obj_id+'_meta_1','region_focus_plane.csv'),index=0)
            print('3-point focus data acquired')

            # find z coordinates for every tile
            # read position list json
            self.pos_z = self.position_list_json

            for col in df_z_info.columns:
                # load 3 focus
                tl = np.array([df_z_info[col][0],df_z_info[col][1],df_z_info[col][2]])
                tr = np.array([df_z_info[col][3],df_z_info[col][4],df_z_info[col][5]])
                bm = np.array([df_z_info[col][6],df_z_info[col][7],df_z_info[col][8]])
                # calculate normal vector of focus plane
                vec_1 = tr-tl
                vec_2 = bm-tl
                vec_n = np.cross(vec_1,vec_2)
                for tile in self.pos_z[col]:
                    # calculate Z projection for each tile
                    z_tile = -(vec_n[0]/vec_n[2])*(tile[0]-tl[0]) - (vec_n[1]/vec_n[2])*(tile[1]-tl[1]) + tl[2]
                    # add Z coord to position list
                    this_index = self.pos_z[col].index(tile)
                    self.pos_z[col][this_index].append(z_tile)

            with open(os.path.join(self.root,self.animal_id,'raw',self.obj_id,self.obj_id+'_meta_1','regions_pos_withz.json'), 'w') as f:
                json.dump(self.pos_z,f)
            print('complete PositionList with focus saved')
            # self.generate_evt()
            # print('Acq event generated')
    
    def generate_evt(self,filter_cube):
        # get exposure time information
        with open("exposure_50um_10x.json",'r') as f:
            exp_50um_10x = json.load(f)
        try:
            exposure_t = exp_50um_10x[filter_cube]
        except KeyError: # exposure setting not given
            exposure_t = 100
        print("Exposure time: ",str(exposure_t),"ms")

        # get FocusDrive correction information
        with open("focusdrive_correction.json",'r') as f:
            focus_corr = json.load(f)
        try:
            focus_0 = focus_corr[self.focus_channel]
            focus_dest = focus_corr[filter_cube]
            focus_compensation = focus_dest - focus_0

        except KeyError: # info not given
            focus_compensation = 0
            
        print("FocusDrive correction: ",str(focus_compensation),"um")

        self.events = []

        for region in self.pos_z.keys():
            for tile in self.pos_z[region]:
                evt = {
                'axes': {'pos_x': int(tile[0]), 'pos_y': int(tile[1]), 'pos_z': int(tile[2])},
                'exposure': exposure_t, # read from exposure_50um_10x.json
                'x': tile[0],
                'y': tile[1],
                'z': tile[2] + focus_compensation,
                'keep_shutter_open': False,
                }
                self.events.append(evt)



    
    def acq(self,filter_cube):
        filter_code = self.filter_name[filter_cube]
        print('begin to acq '+filter_cube+' channel')
        self.core.set_property('IL-Turret','Label',filter_code)
        print("generating acquisition event")
        self.generate_evt(filter_cube)
        print("start acquisition ...")
        with Acquisition(directory=os.path.join(self.root,self.animal_id,'raw',self.obj_id),name=self.obj_id+"_"+filter_cube) as ac:
            ac.acquire(self.events)

        print(self.animal_id,self.obj_id,filter_cube+' channel acquired')
        print('------')



