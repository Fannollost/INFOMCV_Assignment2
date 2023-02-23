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


def set_voxel_positions(width, height, depth):
    # Generates random voxel locations
    # TODO: You need to calculate proper voxel arrays instead of random ones.
    data = []
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                if random.randint(0, 1000) < 5:
                    data.append([x*block_size - width/2, y*block_size, z*block_size - depth/2])
    return data


def get_cam_positions():

    # Generates dummy camera locations at the 4 corners of the room
    # TODO: You need to input the estimated locations of the 4 cameras in the world coordinates.
    camArray = [const.CAM1, const.CAM2, const.CAM3, const.CAM4]
    cam_positions = []
    for i in range(len(camArray)):
        rvec = getDataFromXml(camArray[i][0] + '/data.xml', 'RVecs')
        rotationMatrix = cv.Rodrigues(np.array(rvec).astype(np.float32))[0]
        tvec = getDataFromXml(camArray[i][0] + '/data.xml', 'TVecs')
        camPos = -np.matrix(rotationMatrix).T * np.matrix(np.array(tvec).astype(np.float32)).T
        cam_positions.append(camPos)
    return cam_positions

def getDataFromXml(filePath, nodeName):
    tree = ET.parse(filePath)
    root = tree.getroot()
    data = ''
    for node in root.findall(nodeName):
        data = node.find("data").text
    data = data.split('\n')
    retData = []
    for i in range(len(data) - 1):
        retData.append(float(data[i]))
    return retData


def get_cam_rotation_matrices():
    camArray = [const.CAM1, const.CAM2, const.CAM3, const.CAM4]
    cam_rotations = []
    for i in range(len(camArray) - 1):
        rvec = getDataFromXml(camArray[i][0] + '/data.xml', 'RVecs')
        rotationMatrix = cv.Rodrigues(np.array(rvec).astype(np.float32))[0]
        cam_rotations.append(glm.mat4(rotationMatrix))
    print(cam_rotations)
    # Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room
    # TODO: You need to input the estimated camera rotation matrices (4x4) of the 4 cameras in the world coordinates.
    cam_angles = [[0, 45, -45], [0, 135, -45], [0, 225, -45], [0, 315, -45]]
    for c in range(len(cam_rotations)):
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][0] * np.pi / 180, [1, 0, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][1] * np.pi / 180, [0, 1, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][2] * np.pi / 180, [0, 0, 1])
    return cam_rotations
    
print(get_cam_rotation_matrices())