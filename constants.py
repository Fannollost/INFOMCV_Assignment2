import numpy as np
#declare constants
RED    = (255,0,0)
ORANGE = (255, 102, 0)
YELLOW = (255, 255, 0)
GREEN  = (0,255,0)
LBLUE  = (102, 153, 255)
BLUE   = (0,0,255)

BOARD_SIZE = (8,6)
SQUARE_SIZE = 115
WINDOW_NAME = 'img'
WINDOW_SIZE = (60,40)

DATA_PATH   = './data/calibration.npz'
IMAGES_PATH = './pics/*.jpg'
IMAGES_PATH_YANNICK = './pics/yannick/test/*.jpg'
IMAGES_PATH_FABIEN = './pics/fabien/*.jpg'
IMAGES_PATH_DEFAULT = './pics/default/*.jpg'
IMAGES_PATH_FLOOR = './pics/floor/*.jpg'
IMAGES_PATH_TEST_MANUAL = './pics/testingSet/manual*.jpg'
IMAGES_PATH_TEST_ALL = './pics/testingSet/*.jpg'
IMAGES_PATH_TEST_SELECTION = './pics/testingSet/selection*.jpg'
IMAGES_PATH_TEST_SUB_SELECTION = './pics/testingSet/selection_sub*.jpg'


AXIS = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
CUBE_AXIS = np.float32([[0,0,0], [0,1,0], [1,1,0], [1,0,0],
                   [0,0,-1],[0,1,-1],[1,1,-1],[1,0,-1] ])

WEBCAM = False
FORCE_CALIBRATION = True
REJECT_LOW_QUALITY = False

VIDEO_PATH_CAM1 = "./data/cam1/"
VIDEO_PATH_CAM2 = "./data/cam2/"
VIDEO_PATH_CAM3 = "./data/cam3/"
VIDEO_PATH_CAM4 = "./data/cam4/"
SELECTED_CAM = VIDEO_PATH_CAM1


VIDEO_INTRINSICS = 'intrinsics.avi'
VIDEO_CHECKERBOARD = 'checkerboard.avi'
VIDEO_BACKGROUND = 'background.avi'
VIDEO_TEST = 'video.avi'
SELECTED_VIDEO = VIDEO_INTRINSICS
INTRINSICS_DATA = 'intrinsics.xml'
