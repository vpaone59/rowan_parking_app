# https://github.com/computervisioneng/parking-space-counter/blob/master/util.py
#import pickle

from skimage.transform import resize
import numpy as np
import cv2
import time

class Util:

    def __init__(self):
            self.EMPTY = True
            self.NOT_EMPTY = False

            self.min_width, self.min_height = 4, 4
            # set model confidence level, at least 50% confident in model classification 
            self.confidence_threshold = 0.5
            self.nms_threshold = 0.25
            # even though this is probably safe I don't feel great opening a pickle file from the web
            #MODEL = pickle.load("model.p", "rb")

            # Setting up yolo model instead
            # Load names of classes and get random colors
            self.classes = open('./models/coco.names').read().strip().split('\n')
            np.random.seed(42)
            self.colors = np.random.randint(0, 255, size=(len(self.classes), 3), dtype='uint8')

            # Give the configuration and weight files for the model and load the network.
            self.net = cv2.dnn.readNetFromDarknet('./models/yolov3.cfg', './models/yolov3.weights')
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            # determine the output layer
            ln = self.net.getLayerNames()
            print('layer: ', self.net.getUnconnectedOutLayers())
            self.ln = [ln[i - 1] for i in self.net.getUnconnectedOutLayers()]


    def empty_or_not(self, spot_bgr):

        obj_test = '' 
        test_title = ''
        spot_bgr = spot_bgr.astype(np.float32)
        blob = cv2.dnn.blobFromImage(spot_bgr, 1/255.0, (416, 416), swapRB=True, crop=False)

        #r = blob[0, 0, :, :]
        self.net.setInput(blob)
        t0 = time.time()
        outputs = self.net.forward(self.ln)
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
                if confidence > self.confidence_threshold:
                    box = detection[:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    box = [x, y, int(width), int(height)]
                    boxes.append(box)
                    confidences.append(float(confidence))
                    classIDs.append(classID)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        # taking this from yolo example and only looking for classes detected
        if len(indices) > 0:
            for i in indices.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                obj_test = self.classes[classIDs[i]]
                text = "{}: {:.4f}".format(self.classes[classIDs[i]], confidences[i])
                color = [int(c) for c in self.colors[classIDs[i]]]
                test_title = text
                #cv2.putText(lot_cv2_img, obj_test, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                print(text)
        # return what object prediction is to display 
        if obj_test == 'car' or obj_test == 'truck':
            return self.NOT_EMPTY, test_title
        else:
            return self.EMPTY, test_title
        
    def get_parking_spots_bboxes(self, connected_components):
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
            if w < self.min_width or h < self.min_height:
                continue       
            slots.append([x1, y1, w, h])
        return slots