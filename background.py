import cv2 as cv
import constants as const
import numpy as np
from cameraCalibration import getImagesFromVideo

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
    randC = np.random.randint(0,c)
    randL = np.random.randint(0,l)
    array = np.empty(3, dtype=object)
    for i in range(3):
        array[i] = []
    for i in range(l):
        for j in range(c):
            for k in range(3):
                res[i, j, k] = Stats()

    for frame in frames:
        l,c,_ = frame.shape

        for i in range(l):
            for j in range(c):
                for k in range(3):
                    res[i,j,k].add(frame[i,j,k])
                    # if i == randL and j == randC:
                    #     array[k].append(frame[i,j,k])

    print((randC,randL))
    for k in range(3):
        currChannel = np.array(array[k])
        print("Exact val : Mean : "+ str(currChannel.mean()) + " SD : "+ str(currChannel.std()))
        print("Online val : Mean : "+ str(res[randL, randC, k].mean) + " SD : "+ str(res[randL, randC, k].std))
    return res

if __name__ == "__main__":
    camArray = [const.CAM1, const.CAM2, const.CAM3, const.CAM4]
    for i in range(4):
        print(i)
        backgroundModel(camArray[i][0], const.VIDEO_BACKGROUND)