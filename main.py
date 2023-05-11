"""
PICTURES/IMAGES MUST BE RESIZED TO 1920x1080
"""


import cv2 
import matplotlib.pyplot as plt 
import numpy as np 
import os

from util import Util 

for filename in os.listdir(os.getcwd() + '/images/masks'):
    # paths to images for mask and lot
    print(filename)
    
    if filename.endswith('.jpg') : # CHANGE FILE EXTENSION BASED ON MASK USED
       
        mask = './images/masks/' + filename
        print(mask)
        
        lot = './images/lot_images/RAIN_eng7.jpg' # CHANGE FILENAME
        
        # read in images in grayscale
        mask = cv2.imread(mask, 0)
        lot = cv2.imread(lot)
        utilHlper = Util()
        connected_components = cv2.connectedComponentsWithStats(mask, 16, cv2.CV_32S)
        spots = utilHlper.get_parking_spots_bboxes(connected_components)
        #@print(spots[0])

        for spot in spots:
            x1, y1, w, h = spot 
            # check to see what's in the area of interest 
            spot_crop = lot[y1:y1 + h, x1:x1 + w, :]
            # is spot empty or not
            spot_stuff, spot_obj_name = utilHlper.empty_or_not(spot_crop)
            spot_obj_name = spot_obj_name.split(":")[0]
            if spot_stuff:
                lot = cv2.rectangle(lot, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
                cv2.putText(lot, spot_obj_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 55), 1)
                #print(spot_obj_name)
            else:
                lot = cv2.rectangle(lot, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)
                cv2.putText(lot, spot_obj_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 55), 1)
                #print(spot_obj_name)

            # blue rectangle
            #lot = cv2.rectangle(lot, (x1, y1), (x1 + w, y1 + h), (255, 0, 0), 2)

        #cv2.imshow('lot_'+filename, lot)
        #cv2.imshow('mask_'+filename, mask)
        
        # display masked image of spaces overlayed on input image of parking lot
        alpha = 0.5
        gray_lot = cv2.cvtColor(lot, cv2.COLOR_BGR2GRAY)
        color_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(lot, alpha, color_mask, 1-alpha, 0)
        cv2.imshow('Overlay_'+filename, overlay)
cv2.waitKey(0)