#%% initiate scanner object
from device_driver import Scanner
# only run when starting VSC, when error: micromanager was not ready yet, restart VSC
scanner = Scanner()
# scanner.magnif = 20 # if not stated, 10X is default (0.64um/pixel)
#%% information about task
scanner.root = 'E:\\Username'
scanner.animal_id = '000000'
scanner.obj_id = 'obj1'
scanner.createFolder()
# scanner.focus_channel = 'cy5'
# switch to 100% camera here
#%% scan whole object slide with 5X objective
scanner.scan_5x() 
# waiting
#%% detect roi
scanner.roiarea_min = 10000
scanner.roiarea_max = 200000
scanner.roi_w_min = 50
scanner.roi_h_max = 100000

scanner.roi_detection()
 #%% modify roi here

# merge first
# scanner.merge([9,10])

# then remove
# scanner.delete([0,19])

# confirm changes
scanner.confirm_roi()
#%%
# manual select ROI
scanner.roi_select_tool([90,255])
#%% call focus helper, change focus channel here
#filter_name = {'green':'3-L5',
               # 'cy3':'5-TX2',
               # 'n3':'4-N3',
               # 'dapi':'2-A4',
               # 'cy5':'6-Y5'}

scanner.get_pl_focus('cy3') # filtercube name for focus
#%% acquire each channel
for channel in ['cy3','green']: # 'dapi', 'green','cy3','cy5', 'n3'
    scanner.acq(channel)

