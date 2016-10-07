#!/usr/bin/env python
# read_sudoku.py

import cv2
import math
import numpy as np
from sklearn.externals import joblib
from skimage.feature import hog
from sklearn.cluster import KMeans
import argparse


def get_args():
    """"get arguments from command line"""
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str, help='test sudoku image')

    return parser.parse_args()


def find_sudoku_square(image):
    """"find sudoku square position and crop the image
    input : RGB image
    output : cropped RGB image around sudoku table

    method : 1. find the courners of contours
             2. pick contour points which the contour area and perimeter are large enough on

    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C
                                   ,cv2.THRESH_BINARY_INV,11,2)

    ctrs, hich = cv2.findContours(thresh.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    threshod_area = int(thresh.size/5)

    for ctr in ctrs:
        area = cv2.contourArea(ctr)
        peri = cv2.arcLength(ctr, True)

        if area > threshod_area and peri > image.shape[0] + image.shape[1]:
            rect = cv2.boundingRect(ctr)
            cropped_img = image[rect[1]: rect[1] + rect[3], rect[0]:rect[0] + rect[2]]

    return cropped_img


def angle2ptr(line):
    """convert a polar coordinate of a line
        to two (x,y) coordinate points"""

    rho = line[0]
    theta = line[1]
    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a * rho
    y0 = b * rho

    pt1 = np.array([int(x0 + 1000 * -b), int(y0 + 1000 * a)])
    pt2 = np.array([int(x0 - 1000 * -b), int(y0 - 1000 * a)])

    return pt1,pt2

def perp(a):
    """http://www.cs.mun.ca/~rod/2500/notes/numpy-arrays/numpy-arrays.html """
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b


def seg_intersect(a1,a2, b1,b2):
    """http://www.cs.mun.ca/~rod/2500/notes/numpy-arrays/numpy-arrays.html """
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot( dap, db)
    num = np.dot( dap, dp )
    return (num / denom.astype(float))*db + b1


def intersection(line_a, line_b):
    """calculate the intersection point between two lines
    input : polar coordinates of two lines
    output : (x,y) intersected point
    """
    pt1_a, pt2_a = angle2ptr(line_a)
    pt1_b, pt2_b = angle2ptr(line_b)

    inter_ptr = seg_intersect(pt1_a, pt2_a, pt1_b, pt2_b)

    return inter_ptr


def get_intersections(gray):
    """find 100 corner points in sudoku table
        in order to segment each cell
    input : gray scale image
    ouput : 100 corner points

    """
    # Hough line detector is utilised to detect the table lines
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edge = cv2.Canny(blur, threshold1=30, threshold2=90)
    lines = cv2.HoughLines(edge, rho=3, theta=cv2.cv.CV_PI / 180, threshold=300)
    lines = lines.reshape((lines.shape[1], lines.shape[2]))

    # for line in lines:
    #     pt1, pt2 = angle2ptr(line)
    #     cv2.line(image, (pt1[0], pt1[1]), (pt2[0], pt2[1]), (0, 0, 255), 3)

    # Detected lines are divided to horizontal and vertical lines by angle.
    v = []
    h = []
    for line in lines:
        if line[1] < cv2.cv.CV_PI / 20 or line[1] > cv2.cv.CV_PI - cv2.cv.CV_PI / 20:
            v.append(line)
        elif abs(line[1] - cv2.cv.CV_PI / 2) < cv2.cv.CV_PI / 20:
            h.append(line)

    # Calculate all the possible intersections
    pts = []
    for verti in v:
        for horiz in h:
            pt = intersection(verti, horiz)
            pts.append(pt)

    # Apply k-mean with 100 clusters
    kmeans = KMeans(n_clusters=100).fit(pts)

    # gap = np.min([image.shape[0], image.shape[1]])/9
    # init_pts = np.array([[(x,y) for x in range(10)] for y in range(10)]) * gap
    # kmeans = KMeans(n_clusters=100, init=init_pts.reshape((100,2))).fit(pts)

    # Cluster center points are selected to corner points on the sudoku table
    ints = kmeans.cluster_centers_.copy()
    ints = ints[ints[:, 1].argsort()]

    return ints

def main(args):

    file_name = args.image
    image = cv2.imread(file_name)

    image = find_sudoku_square(image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, im_th = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    ints = get_intersections(gray)

    for ptr in ints:
        cv2.circle(image, (int(ptr[0]), int(ptr[1])), 3, (0, 0, 255))

    # Sort intersection points so that its index directly corresponds with the point location.
    points = []
    for i in range(10):
        temp = ints[i*10:i*10+10,:].copy()
        points.append(temp[temp[:,0].argsort()])

    # load the classifier
    classifier = joblib.load('../classifier/linear_svm_hog.pkl')

    labels = []
    for y in range(9):
        for x in range(9):
            point1 = points[y][x]
            point2 = points[y+1][x+1]

            center = (point1+point2)/2
            # take target patch's width and height as 70% of that of the cell
            w, h = (point2-point1)*0.7

            left_top = (int(center[0] - w/2), int(center[1] - h/2))
            right_bottom = (int(center[0] + w/2), int(center[1] + h/2))

            # cv2.circle(image, (int(center[0]), int(center[1])), 3, (0, 255, 0))
            cv2.rectangle(image, left_top, right_bottom, (0,0,255),2 )
            crop = im_th[left_top[1]:right_bottom[1], left_top[0]:right_bottom[0]]

            patch = cv2.resize(crop, (28, 28), interpolation=cv2.INTER_AREA)

            # cv2.namedWindow('test', cv2.WINDOW_NORMAL)
            # cv2.imshow('test', patch)
            # cv2.waitKey(0)

            patch = patch.reshape(1, -1)

            # reject blank images
            if np.max(patch) < 250:
                # print '*'
                label = '*'
            else:
                # intensity normalisation
                patch = patch / 255.0 * 2 - 1

                # feature extraction
                ft = hog(patch.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1),
                         visualise=False)
                label = classifier.predict(np.array([ft], 'float64'))

                # label = classifier.predict(patch)
                label = str(int(label[0]))

                # print label

            cv2.putText(image, label, (int(center[0]), int(center[1])),cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
            labels.append(label)

    print np.array(labels).reshape(9,9)

    # cv2.namedWindow('test', cv2.WINDOW_NORMAL)
    # cv2.imshow('test', image)
    # cv2.waitKey(0)
    #
    # cv2.imwrite('result02.png', image)

if __name__ == '__main__':
    args=get_args()
    main(args)
