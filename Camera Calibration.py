# Importing the necessary libraries

import cv2
import numpy as np
import os

# Creating a list of all the image paths

images_collection = ["IMG_20170209_042606.jpg", "IMG_20170209_042608.jpg", "IMG_20170209_042610.jpg", "IMG_20170209_042612.jpg", "IMG_20170209_042614.jpg", "IMG_20170209_042616.jpg", 
                     "IMG_20170209_042619.jpg", "IMG_20170209_042621.jpg", "IMG_20170209_042624.jpg", "IMG_20170209_042627.jpg", "IMG_20170209_042629.jpg", "IMG_20170209_042630.jpg",
                     "IMG_20170209_042634.jpg"]

CURRENT_DIR = os.path.dirname(__file__)
for element in images_collection:
    image_path = os.path.join(CURRENT_DIR,element)
    image = cv2.imread(image_path)

    # Resizing the imagesq

    width = int(image.shape[1] * 0.3)
    height = int(image.shape[0] * 0.3)
    new_dimensions = (width, height)
    image_resized = cv2.resize(image, new_dimensions)

    # Converting the images into grayscale

    gray = cv2.cvtColor(image_resized,cv2.COLOR_BGR2GRAY)
    boardsize = (9,6)

    # The total number of corners in the image considered for camera caliberation

    array_length = boardsize[0]*boardsize[1]

    # Calculating the world coordinates of corners based on the square size 21.5mm

    world_coordinates = np.zeros((array_length,3), np.float32)
    for x in range(54):
        world_coordinates[x][0] = (x%9)*21.5
        world_coordinates[x][1] = (int(x/9))*21.5

    # Finding the corner point coordinates using the inbuilt OpenCV function cv2.findChessboardCorners()
    # Referred to the OpenCV Camera Calibration

    ret, image_coordinates = cv2.findChessboardCorners(image_resized, boardsize)
    cv2.drawChessboardCorners(image_resized, boardsize, image_coordinates, ret)
    print(image_coordinates)
    print()

    # Creating a list of all the image and the world coordinates considered for camera caliberation

    image_coordinates_list = []
    image_coordinates_list.append(image_coordinates)

    world_coordinates_list = []
    world_coordinates_list.append(world_coordinates)

    # Geeting the intrinsic matrix, rotation and translation vectors, distortion coefficients using the inbuilt OpenCV function cv2.calibrateCamera()
    # Referred to OpenCV Camera Calibration

    ret, camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors = cv2.calibrateCamera(world_coordinates_list, image_coordinates_list, gray.shape[::-1], None, None)

    print(f"Intrinsic matrix for {element}: \n{camera_matrix}")
    print()

    # Calculating the reprojection error using intrinsic matrix, rotation and translation vectors and distortion coefficients using cv2.projectPoints() and cv2.norm()
    # Referred to OpenCV Calibration

    image_coordinates_2, _ = cv2.projectPoints(world_coordinates_list[0], rotation_vectors[0], translation_vectors[0], camera_matrix, distortion_coefficients)
    reprojection_error = cv2.norm(image_coordinates_list[0], image_coordinates_2, cv2.NORM_L2)/len(image_coordinates_2)
    print(f"Reprojection error for {element}: {reprojection_error}")
    print()

    # Displaying the images

    cv2.imshow(f"{element}", image_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
