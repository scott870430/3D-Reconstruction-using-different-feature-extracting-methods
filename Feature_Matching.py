import os
import sys
import glob
import time

import cv2

import matplotlib.pyplot as plt

import numpy as np

Feature_methods = ['SIFT', 'SURF', 'ORB']
Tracking = ['KLT']

def Feature(imgs, name = 'SIFT', scale = 0.6):
    """FEATURE MATCHING USING SIFT/SURF"""
    bf = cv2.BFMatcher()
    if name == 'SIFT':
        process = cv2.xfeatures2d.SIFT_create()
    elif name == 'SURF':
        process = cv2.xfeatures2d.SURF_create()
    else :
        process = cv2.ORB_create(200)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    kps = []
    dess = []
    for img in imgs:
        kp,des= process.detectAndCompute(img, None)
        kps.append(kp)
        dess.append(des)

    results = []
    des = dess[0]
    for e_kp, e_des in zip(kps[0], dess[0]):
        results.append([(e_kp, e_des)])

    for i in range(1,len(imgs)):
        matches = bf.knnMatch(des, dess[i], k=2)
        goods = [m for m, n in matches if m.distance < scale * n.distance]
        new_results = []
        new_des = []
        for good in goods: 
            results[good.queryIdx].append((kps[i][good.trainIdx],dess[i][good.trainIdx]))
            new_results.append(results[good.queryIdx])
            new_des.append(des[good.queryIdx])
        results = new_results
        des = np.array(new_des)

    return results

def KLT(imgs,manual = False):
    feature_params = dict( maxCorners = 50,   # How many pts. to locate
                       qualityLevel = 0.2,  # b/w 0 & 1, min. quality below which everyone is rejected
                       minDistance = 7,   # Min eucledian distance b/w corners detected
                       blockSize = 3 )
    
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),  # size of the search window at each pyramid level
                    maxLevel = 2,   #  0, pyramids are not used (single level), if set to 1, two levels are used, and so on
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    old_gray = imgs[0]
    cv2.imshow("test",old_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    p0 = np.zeros(1)
    if manual :
        with open("manual_feature.txt") as file:
            lines = file.read().splitlines()
            #print(lines)
            for line in lines:
                l = [int(e) for e in line.split()]
                if p0.all() == np.zeros(1):
                    p0 = np.asarray(l)
                else:
                    p0 = np.concatenate((p0,l))
        p0 = p0.astype(np.float32)
    else :
        first_img = cv2.imread('./images/'+'1.jpg')
        old_gray = cv2.cvtColor(first_img ,cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    #print(len(p0))
    '''
    for e in p0:
        cv2.circle(old_gray,(int(e[0][0]), int(e[0][1])),10, (0, 0, 255), 1)

    cv2.imwrite("first_feature.jpg",old_gray)
    cv2.imshow("test",old_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(len(p0))
    '''
    print("KLT feature number before start: ",len(p0))
    for image_index in range(1,len(imgs)):

        frame_gray  = imgs[image_index-1]

        p0 = p0.reshape(-1,1,2)
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray , p0, None, **lk_params)

        good_old = p0[st==1]
        good_new = p1[st==1]

        old_gray = frame_gray .copy()
        p0 = good_new


    for image_index in range(len(imgs),0,-1):
        print("index: ",image_index," ",len(p0))
        frame_gray = imgs[image_index-1]
        #frame_gray = cv2.cvtColor(next_img,cv2.COLOR_BGR2GRAY)

        p0 = p0.reshape(-1,1,2)

        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        good_old = p0[st==1]
        good_new = p1[st==1]

        old_gray = frame_gray.copy()

        p0 = good_new

    results = []

    for image_index in range(1,len(imgs)):
        frame_gray  = imgs[image_index-1]

        p0 = p0.reshape(-1,1,2)
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        '''
        for e in good_new:
            cv2.circle(frame_gray,(int(e[0]), int(e[1])),10, (0, 0, 255), 1)
        '''
        old_gray = frame_gray.copy()
        p0 = p1

        results.append(p1.reshape(-1,2))
    '''
    for e in good_old:
        cv2.circle(old_gray,(int(e[0]), int(e[1])),10, (0, 0, 255), 1)
    '''
    #cv2.imwrite("feature.jpg",old_gray)

    return results


def process(imgs, feature = 'SIFT', scale = 0.6 ,  manual = False):

    if feature in Feature_methods:
        results_array = Feature(imgs, feature, scale)
    elif feature in Tracking:
        results_array = KLT(imgs, manual)
    else:
        sys.exit('DO NOT HAVE METHOD NAME : ' + feature)    
    
    #If there are too many images or the conditions are too harsh, the feature points may be 0

    assert len(results_array) != 0 ,'featrue point is zero!'
    dirName = feature+'_Features'

    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ")
    else:    
        print("Directory " , dirName ,  " already exists")

    if feature in Tracking:
        for idx, result in enumerate(results_array):
            with open(dirName+'/'+str(idx+1)+"image_point.txt", "w+") as fp:
                    print(len(result),file = fp)
                    for element in result:
                        print(element[0],element[1],file = fp)
    else:
        for i in range(len(results_array[0])):
            with open(dirName+'/'+str(i+1)+'image_point.txt', 'w+') as output_file:
                print(len(results_array), file=output_file)
                for arr in results_array:
                    pt = arr[i][0].pt
                    print(pt[0], pt[1], file=output_file)
        
    return results_array   

def main():
    feature = 'ORB'
    img_names = glob.glob('Photos/*.jpg')
    # load image
    imgs = [cv2.imread(name, cv2.IMREAD_GRAYSCALE) for name in img_names]  

    results_array = process(imgs[:2], feature=feature)

    if feature in Feature_methods:
        img0 = np.copy(imgs[0])
        img1 = np.copy(imgs[1])

        for arr in results_array:
            cv2.circle(img0,(int(arr[0][0].pt[0]), int(arr[0][0].pt[1])), 10, (0, 255, 255), 3)
            cv2.circle(img1,(int(arr[1][0].pt[0]), int(arr[1][0].pt[1])), 10, (0, 255, 255), 3)

        cv2.imwrite('out0.jpg',img0)
        cv2.imwrite('out1.jpg',img1)

if __name__ == '__main__':
    main()