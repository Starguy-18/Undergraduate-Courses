# Computer Vision Homework 04 Problem 01
# Author: Hunter Rouillard

import numpy as np
import sys
import os
import cv2

# It is assumed that the image is grayscale and has been blurred
# Derivatives returned as tuple (dx, dy)
def getDerivatives(image):
    k_dx, k_dy = cv2.getDerivKernels(1, 1, 3)
    return (cv2.filter2D(image, -1, np.transpose(k_dx / 2)), cv2.filter2D(image, -1, k_dy / 2))

# Outputs the elements of the structure tensor ((I_x)^2, (I_y)^2, (I_x)(I_y))
# weighted by a circular gaussian weight function using specified deviation
def getStructureTensor(dx, dy, sigma):
    s = 2 * sigma
    ksize = tuple([4 * s + 1] * 2)
    return (cv2.GaussianBlur(pow(dx, 2), ksize, s), cv2.GaussianBlur(pow(dy, 2), ksize, s), cv2.GaussianBlur(dx * dy, ksize, s))

# Given a Structure Tensor and an empirical constant "kappa", outputs the Harris Measure
def getHarrisMeasure(M, kappa):
    return (M[0] * M[1] - pow(M[2], 2)) - kappa * pow(M[0] + M[1], 2)

# Normalize array values between the range r1...r2 inclusive using linear transform
def normalizeArray(array, r1, r2):
    mi = np.amin(array)
    ma = np.amax(array)
    a = (r2 - r1) / (ma - mi)
    b = r2 - a * ma
    return a * array + b

# Applies non-maximum suppression to 
def nonMaximumSuppression(array, sigma):
    mask = cv2.dilate(array, np.ones(tuple([4 * sigma + 1] * 2), np.uint8))
    array[array < mask] = 0
    return array

# Obtains the keyPoints of an image using the Harris-Laplace keypoint detector
# Outpusts a list of [xs, ys, vals]
def getHarrisKeyPoints(R, sigma):
    # Extract peak data, (x, y, measure value), and sort by highest measure value
    point_data = sorted(np.concatenate((np.transpose(np.array(list(np.where(R > 0))[::-1])), np.transpose([R[R > 0]])), axis = 1), key = lambda x: x[2], reverse = True)
    return [cv2.KeyPoint(p[0], p[1], 4 * sigma, _response = p[2]) for p in point_data]

# Prints the top N keyPoints from a given array of keyPoint Objects
def printTopKeyPoints(array, title, N):
    print('\n' + title + ':')
    for i in range(N):
        print("{0}: ({1:.1f}, {2:.1f}) {3:.4f}".format(i, array[i].pt[0], array[i].pt[1], array[i].response))

# Determines the image distances between a specified keyPoint, and a given list of keyPoints
def keyPointDistance(pnt, points):
    return np.sqrt(pow(pnt.pt[0] - np.array([p.pt[0] for p in points]), 2) + pow(pnt.pt[1] - np.array([p.pt[1] for p in points]), 2))

# Outputs a comparison measure of image distance and rank difference between a
# specified KeyPoint the point of shortest image distance amongst a list of KeyPoints
# it is assumed that "points" is a list of KeyPoints ordered by highest response
def compareDistances(pnt, index, points):
    point_data = keyPointDistance(pnt, points)
    return [np.amin(point_data), float(abs(index - np.argmin(point_data)))]

def printComparisonStats(comp, title):
    print("\n{0}:".format(title))
    print("Median distance: {0:.1f}\nAverage distance: {1:.1f}\nMedian index difference: {2:.1f}\nAverage index difference: {3:.1f}".format(np.median(comp[:, 0]), np.mean(comp[:, 0]), np.median(comp[:, 1]), np.mean(comp[:, 1])))


if (__name__ == "__main__"):
    if (len(sys.argv) != 3):
        print("ERROR Invalid number of command line arguments")
        sys.exit()
    
    image_file = sys.argv[2]
    if (image_file.find('.') == -1):
        print("ERROR: Unable to find name in image file name")
        sys.exit()
    name = image_file.split('.')
    path = "Images/" + name[0] + '/'
    sigma = int(sys.argv[1])
    
    image = cv2.imread(path + image_file, cv2.IMREAD_GRAYSCALE)
    
    # Use Gaussian Blur on image before extracting partial derivatives
    ksize = [4 * sigma + 1] * 2
    blured_image = cv2.GaussianBlur(image.astype(np.float32), tuple(ksize), sigma)
    
    # Obtain partial derivatives
    dx, dy = getDerivatives(blured_image)
    
    # Calculate Weighted Structure Tensor
    M = getStructureTensor(dx, dy, sigma)
    
    # Calculate normalized Harris Measure between range 0...255 and then perform non-maximum suppression
    R = nonMaximumSuppression(normalizeArray(getHarrisMeasure(M, 0.004), 0, 255), sigma)
    
    # Use the Harris Measure to obtain keyPoints of an image
    H_keyPoints = getHarrisKeyPoints(R, sigma)
    
    # Print the Top 10 Harris Keypoints
    printTopKeyPoints(H_keyPoints, "Top 10 Harris KeyPoints", 10)
    
    # ORB Feature Detection
    num_features = 1000 # Maximum number of featrues
    ORB = cv2.ORB_create(num_features)
    # Calculate, order and filter ORB keypoints
    ORB_keyPoints, OBR_descriptors = ORB.detectAndCompute(image, None)
    ORB_keyPoints = list(filter(lambda x: x.size < 45, sorted(ORB_keyPoints, key = lambda x: x.response, reverse = True)))
    printTopKeyPoints(ORB_keyPoints, "Top 10 ORB KeyPoints", 10)
    
    # Calculate the distance to the closest ORB KeyPoint for each Harris KeyPoint
    # Uses the first 200 ORB KeyPoints of hightest response
    H_comp_ORB = np.array([compareDistances(H_keyPoints[i], i, ORB_keyPoints[:200]) for i in range(100)])
    printComparisonStats(H_comp_ORB, "Harris keypiont to ORB distances")
    
    # Compare ORB keypoints to Harris keypoints
    ORB_comp_H = np.array([compareDistances(ORB_keyPoints[i], i, H_keyPoints[:200]) for i in range(100)])
    printComparisonStats(ORB_comp_H, "ORB keypoint to Harris distances")
    
    # Display First 200 KeyPoints
    H_image = cv2.drawKeypoints(image, H_keyPoints[:200], None)
    ORB_image = cv2.drawKeypoints(image, ORB_keyPoints[:200], None)
    cv2.imwrite(path + name[0] + "_harris." + name[1], H_image)
    cv2.imwrite(path + name[0] + "_orb." + name[1], ORB_image)