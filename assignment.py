import glm
import random
import numpy as np
import xml.etree.ElementTree as ET
import constants as const
import cv2 as cv
block_size = 1.0


def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data = []
    for x in range(width):
        for z in range(depth):
            data.append([x*block_size - width/2, -block_size, z*block_size - depth/2])
    return data

def getDataFromXml(filePath, nodeName):
    tree = ET.parse(filePath)
    root = tree.getroot()
    data = ''
    for node in root.findall(nodeName):
        data = node.find("data").text
    data = data.split('\n')
    retData = []
    for i in range(len(data) - 1):
        lineData = data[i].split(' ')
        if len(lineData) > 1:
            lineRes = []
            for j in range(len(lineData)):
                lineRes.append(float(lineData[j]))
            retData.append(lineRes)
        else:
            retData.append(float(data[i]))
    return np.array(retData)

#Generate a Voxel world from camera informations
# TODO: You need to calculate proper voxel arrays instead of random ones.
def set_voxel_positions(width, height, depth):
    camArray = [const.CAM1, const.CAM2, const.CAM3, const.CAM4]

    camParams =[]
    for cam in camArray:
        camPath = cam[0]
        foreground = cv.imread(camPath + "foreground.png", cv.IMREAD_GRAYSCALE)
        rvec = getDataFromXml(camPath + 'data.xml', 'RVecs')
        tvec = getDataFromXml(camPath + 'data.xml', 'TVecs')
        cameraMatrix = getDataFromXml(camPath + 'data.xml', 'CameraMatrix')
        distCoeffs = getDataFromXml(camPath + 'data.xml', 'DistortionCoeffs')
        params  = dict(rvec = rvec, tvec = tvec, cameraMatrix = cameraMatrix, distCoeffs = distCoeffs, foreground = foreground)
        camParams.append(params)
    data = []
    for x in range(width):
        print("X : "+str(x) + " / " + str(width))
        for y in range(height):
            for z in range(depth):
                isOn = True
                for i in range(len(camArray)):
                    params = camParams[i]
                    imagepoints, _ = cv.projectPoints((x,y,z), params["rvec"], params["tvec"], params["cameraMatrix"], params["distCoeffs"])
                    imagepoints = np.reshape(imagepoints, 2)
                    (heightIm, widthIm) = params["foreground"].shape
                    if 0 <= imagepoints[0] < heightIm and 0 <= imagepoints[1] < widthIm:
                        pixVal = foreground[int(imagepoints[0]), int(imagepoints[1])]
                        if pixVal == 0:
                            isOn = False
                    else :
                        isOn = False
                if isOn:
                    data.append([x * block_size - width / 2, y * block_size, z * block_size - depth / 2])



    return data

# Generates dummy camera locations at the 4 corners of the room
# TODO: You need to input the estimated locations of the 4 cameras in the world coordinates.
def get_cam_positions():
    camArray = [const.CAM1, const.CAM2, const.CAM3, const.CAM4]
    cam_positions = []
    for i in range(len(camArray)):
        rvec = getDataFromXml(camArray[i][0] + 'data.xml', 'RVecs')
        rotationMatrix = cv.Rodrigues(np.array(rvec).astype(np.float32))[0]
        #rotationMatrix = rotationMatrix * [[1,0,0],[0,-1,0],[0,0,-1]]
        tvec = getDataFromXml(camArray[i][0] + 'data.xml', 'TVecs')
        camPos = -np.matrix(rotationMatrix).T * np.matrix(np.array(tvec).astype(np.float32)).T
        cam_positions.append([camPos[0], -camPos[2], camPos[1]])
    return cam_positions

# Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room
# TODO: You need to input the estimated camera rotation matrices (4x4) of the 4 cameras in the world coordinates.
def get_cam_rotation_matrices():
    camArray = [const.CAM1, const.CAM2, const.CAM3, const.CAM4]
    cam_rotations = []
    for i in range(len(camArray)):
        rvec = getDataFromXml(camArray[i][0] + '/data.xml', 'RVecs')
        rotationMatrix = cv.Rodrigues(np.array(rvec).astype(np.float32))[0]
        rotationMatrix = rotationMatrix.transpose()
        rotationMatrix = [rotationMatrix[0], rotationMatrix[2], rotationMatrix[1]]
        cam_rotations.append(glm.mat4(np.matrix(rotationMatrix).T))
    print("F:")
    print(cam_rotations)

    for c in range(len(cam_rotations)):
        cam_rotations[c] = glm.rotate(cam_rotations[c], -np.pi/2 , [0, 1, 0])
    return cam_rotations
    
get_cam_rotation_matrices()