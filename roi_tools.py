import numpy as np
import json

# each region consist of (x,y,w,h)
def merge_roi(regions,selected_regions):
    coord_record = {'x':[],'y':[],'w':[],'h':[]}
    for r in selected_regions:
        x,y,w,h = regions[r]
        coord_record['x'].append(x)
        coord_record['y'].append(y)
        coord_record['w'].append(w)
        coord_record['h'].append(h)

    x_min = np.min(coord_record['x'])
    y_min = np.min(coord_record['y'])
    x_max = np.max(np.array(coord_record['x'])+np.array(coord_record['w']))
    y_max = np.max(np.array(coord_record['y'])+np.array(coord_record['h']))
    w_new = x_max - x_min
    h_new = y_max - y_min
    regions.append((x_min,y_min,w_new,h_new))
    return regions

def del_roi(regions,selected_regions):
    for r in sorted(selected_regions, reverse = True):
        del regions[r]
    return regions


def sort_region(regions,s_shaped_scan):
    # sort into rows
    regions_np = np.array(regions)
    regions_np = regions_np[regions_np[:,1].argsort()]
    # find row break
    dy = np.diff(regions_np[:,1])
    row_break = np.where(dy>np.std(dy))[0] + 1
    regions_np_break = np.split(regions_np,row_break)
    # sort within each row
    regions_sorted = []
    if s_shaped_scan is False:
        for row in regions_np_break:
            for r in row[row[:,0].argsort()]:
                regions_sorted.append(r)
    else:
        print('S-shaped region ordering is not implemented yet.') 
    
    return regions_sorted


def position_list(tl,br,pixsize,dest):
    # bottom right coordinates need to be non-negative
    stepsize = 0.9 * 2048 * pixsize

    position_list = []
    position_list_json = {}

    tile_xn = np.ceil(np.abs(tl[0] - br[0])/stepsize).astype(int) + 1 # fence post + 1
    tile_yn = np.ceil(np.abs(tl[1] - br[1])/stepsize).astype(int) + 1
    
    # direction in X-direction
    direction = 1
    hold = 1

    now_x = br[0]
    now_y = br[1] - stepsize

    for i in range(tile_yn):
        for j in range(tile_xn):
            if hold == 1:
                now_y += stepsize
                position_list.append((now_x,now_y))
                hold -= 1
            else:
                if direction > 0: # x increase
                    now_x += stepsize
                    position_list.append((now_x,now_y))
                else: # x decrease
                    now_x -= stepsize
                    position_list.append((now_x,now_y))
        direction *= (-1)
        hold += 1

    position_list_json['pos'] = position_list
    with open(dest, 'w') as f:
        json.dump(position_list_json, f)

