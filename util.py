# https://github.com/computervisioneng/parking-space-counter/blob/master/util.py
#import pickle

from skimage.transform import resize
import numpy as np
import cv2
import time

EMPTY = True
NOT_EMPTY = False
min_width, min_height = 4, 4
# set model confidence level, at least 50% confident in model classification 
confidence_threshold = 0.5
nms_threshold = 0.25
# even though this is probably safe I don't feel great opening a pickle file from the web
#MODEL = pickle.load("model.p", "rb")

# Setting up yolo model instead
# Load names of classes and get random colors
classes = open('./models/coco.names').read().strip().split('\n')
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

# Give the configuration and weight files for the model and load the network.
net = cv2.dnn.readNetFromDarknet('./models/yolov3.cfg', './models/yolov3.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# determine the output layer
ln = net.getLayerNames()
print('layer: ', net.getUnconnectedOutLayers())
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]


def empty_or_not(spot_bgr):

    obj_test = '' 
  
    spot_bgr = spot_bgr.astype(np.float32)
    blob = cv2.dnn.blobFromImage(spot_bgr, 1/255.0, (416, 416), swapRB=True, crop=False)

    #r = blob[0, 0, :, :]
    net.setInput(blob)
    t0 = time.time()
    outputs = net.forward(ln)
    t = time.time()
    # print(len(outputs))
    # for out in outputs:
    #     print(out.shape)

    # TODO: fine tune and refine
    boxes = []
    confidences = []
    classIDs = []
    h, w = spot_bgr.shape[:2]
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > confidence_threshold:
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                box = [x, y, int(width), int(height)]
                boxes.append(box)
                confidences.append(float(confidence))
                classIDs.append(classID)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
    # taking this from yolo example and only looking for classes detected
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            obj_test = classes[classIDs[i]]
            text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
            print(text)

    if obj_test == 'car' or obj_test == 'truck':
        return NOT_EMPTY
    else:
        return EMPTY
    
def get_parking_spots_bboxes(connected_components):
    (totalLabels, label_ids, values, centroid) = connected_components

    slots = []
    coef = 1
    for i in range(1, totalLabels):
        # get coords
        x1 = int(values[i, cv2.CC_STAT_LEFT] * coef)
        y1 = int(values[i, cv2.CC_STAT_TOP] * coef)
        h = int(values[i, cv2.CC_STAT_HEIGHT] * coef)
        w = int(values[i, cv2.CC_STAT_WIDTH] * coef)
        # for some reason get some bounding boxes that are 1 to 4 pixels wide, check the width to skip adding them
        if w < min_width or h < min_height:
            continue       
        slots.append([x1, y1, w, h])
    return slots