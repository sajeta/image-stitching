import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image


MAX_FEATURES = 5000
GOOD_MATCH_PERCENT = 0.15


def detectAndDescribe(image, method=None):
    """
    Compute key points and feature descriptors using an specific method
    """

    assert method is not None, "You need to define a feature detection method. Values are: 'sift', 'surf'"

    # detect and extract features from the image
    if method == 'sift':
        descriptor = cv2.xfeatures2d.SIFT_create()
    elif method == 'surf':
        descriptor = cv2.xfeatures2d.SURF_create()
    elif method == 'brisk':
        descriptor = cv2.BRISK_create()
    elif method == 'orb':
        descriptor = cv2.ORB_create()

    # get keypoints and descriptors
    (kps, features) = descriptor.detectAndCompute(image, None)

    return kps, features


def createMatcher(method, crossCheck):
    """Create and return a Matcher Object"""

    if method == 'sift' or method == 'surf':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
    elif method == 'orb' or method == 'brisk':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
    return bf


if __name__ == '__main__':
    trainImg = np.array(Image.open("IMG_1.jpg"))
    queryImg = np.array(Image.open("IMG_2.jpg"))

    img1 = np.zeros_like(trainImg)
    kps1, features1 = detectAndDescribe(trainImg, "surf")
    cv2.drawKeypoints(trainImg, kps1, img1, color=(0, 255, 0))

    img2 = np.zeros_like(queryImg)
    kps2, features2 = detectAndDescribe(queryImg, "surf")
    cv2.drawKeypoints(queryImg, kps2, img2, color=(0, 255, 0))

    plt.figure()
    plt.imshow(img1)
    plt.show()

    plt.figure()
    plt.imshow(img2)
    plt.show()

    # # Match features.
    # matcher = createMatcher("surf", False)
    # matches = matcher.match(features1, features2)

    # # Sort matches by score
    # matches.sort(key=lambda x: x.distance, reverse=False)

    # # Remove not so good matches
    # numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    # matches = matches[:numGoodMatches]

    # # Extract location of good matches
    # points1 = np.zeros((len(matches), 2), dtype=np.float32)
    # points2 = np.zeros((len(matches), 2), dtype=np.float32)
    #
    # for i, match in enumerate(matches):
    #     points1[i, :] = kps1[match.queryIdx].pt
    #     points2[i, :] = kps2[match.trainIdx].pt
    #
    # # Find homography
    # H, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(queryImg, None)
    keypoints2, descriptors2 = orb.detectAndCompute(trainImg, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    # matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_SL2)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(queryImg, keypoints1, trainImg, keypoints2, matches, None)
    cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    H, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Apply panorama correction
    width = trainImg.shape[1] + queryImg.shape[1]
    height = trainImg.shape[0] + queryImg.shape[0]

    result = cv2.warpPerspective(queryImg, H, (width, height))
    result[0:trainImg.shape[0], 0:trainImg.shape[1]] = trainImg

    plt.figure(figsize=(20, 10))
    plt.imshow(result)

    plt.axis('off')
    plt.show()
