import math

import cv2 as cv
import constants as const
import numpy as np
from cameraCalibration import getImagesFromVideo, showImage

class Stats(object):
    """
    Welford's algorithm computes variance and mean online
    """

    def __init__(self):
        self.count, self.M1, self.M2 = 0, 0.0, 0.0

    def add(self, val):
        self.count += 1
        self.delta = val - self.M1
        self.M1 += self.delta / self.count
        self.M2 += self.delta * (val - self.M1)

    @property
    def mean(self):
        return self.M1

    @property
    def variance(self):
        return self.M2 / self.count

    @property
    def std(self):
        return np.sqrt(self.variance)

def backgroundModel(camera, videoType):
    video = cv.VideoCapture(camera + videoType)
    c = int(video.get(cv.CAP_PROP_FRAME_WIDTH ))
    l = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    res = np.empty((l, c, 3), dtype=object)
    frames = getImagesFromVideo(camera, videoType, const.IMAGES_BACKGROUND_NB)

    for i in range(l):
        for j in range(c):
            for k in range(3):
                res[i, j, k] = Stats()

    for frame in frames:
        frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        l,c,_ = frame.shape
        for i in range(l):
            for j in range(c):
                for k in range(3):
                    res[i,j,k].add(frame[i,j,k])

    return res

def channelDist(model, val, dim):
    delta = model[dim].mean - val[dim]
    if delta < 0:
        delta = -delta
    if model[dim].std > 0.5:
        return delta/model[dim].std
    else :
        return delta * 2

def dist(model, val):
    return const.H_WEIGHT * channelDist(model,val,const.H) + const.S_WEIGHT * channelDist(model,val,const.S) + const.V_WEIGHT * channelDist(model,val,const.V)


def mask(model, val):
    if dist(model,val) > const.THRESHOLD:
        return 255
    else :
        return 0


def click_event(event, x, y, flags, params):
    if event == cv.EVENT_LBUTTONDOWN:
        for k in range(3):
            print("CHANNEL : " + str(k))
            print("MODEL Mean : " + str(m[y,x,k].mean) + " SD : " + str(m[y,x,k].std))
            print("FRAME val : " + str(f[y,x,k]))
        print("Total dist : " + str(dist(m[y,x],f[y,x])))

def substractBackground(camera, videoType, model):
    video = cv.VideoCapture(camera + videoType)
    frameCount = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    c = int(video.get(cv.CAP_PROP_FRAME_WIDTH ))
    l = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    res = np.empty(shape=[l, c, 3], dtype=np.uint8)

    # for i in range(frameCount):
    video.set(cv.CAP_PROP_POS_FRAMES, 0)
    ret, frame = video.read()
    frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    for i in range(l):
        for j in range(c):
            res[i,j] = mask(model[i, j], frame[i, j])
    showImage(const.WINDOW_NAME, res, 1000)
    global m
    global f
    m = model
    f = frame
    while True:
        #Get mouseinput
        cv.setMouseCallback(const.WINDOW_NAME, click_event)
        cv.waitKey(100)
        try:
            cv.getWindowProperty(const.WINDOW_NAME, 0)
        except:
            break

if __name__ == "__main__":
    camArray = [const.CAM1, const.CAM2, const.CAM3, const.CAM4]
    model = backgroundModel(camArray[1][0], const.VIDEO_BACKGROUND)
    substractBackground(camArray[1][0], const.VIDEO_TEST, model)