# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 11:46:15 2019

@author: shiha
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt



def undistort(frame):
    k_matrix = np.array([[1.15422732e+03, 0.00000000e+00, 6.71627794e+02],[0.00000000e+00, 1.14818221e+03, 3.86046312e+02], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype = np.uint8)
    dist_coef = np.array([[-2.42565104e-01, -4.77893070e-02, -1.31388084e-03, -8.79107779e-05, 2.20573263e-02]], dtype = np.uint8)
    
    
    (h, w) = frame.shape[:2]
    new_cam_matrix, roi = cv2.getOptimalNewCameraMatrix(k_matrix, dist_coef, (w, h), 0)
    
    frame_undist = cv2.undistort(frame, k_matrix, dist_coef, None, new_cam_matrix)
    
    return frame_undist

    
    
def histo_eql(frame):
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_eql = cv2.equalizeHist(frame)
    frame_eql = cv2.cvtColor(frame_eql, cv2.COLOR_GRAY2BGR)

    
    return frame_eql

def homography(frame, left_shift = 0, back_project = 0):
#    pts_src = np.array([[596, 450], [204, 719], [1108, 719], [684, 450]], dtype=np.float32)
#    pts_dts = np.array([[204, 0], [204, 719], [1108, 719], [1108, 0]], dtype=np.float32)
    
    pts_src = np.array([[596/2, 450/2], [204/2, 719/2], [1108/2, 719/2], [684/2, 450/2]], dtype=np.float32)
    pts_dts = np.array([[204/2, 0], [204/2, 719/2], [1108/2, 719/2], [1108/2, 0]], dtype=np.float32)
    
    if back_project == 0:
        H = cv2.getPerspectiveTransform(pts_src, pts_dts)
    else:
        H = cv2.getPerspectiveTransform(pts_dts, pts_src)
    
    if left_shift == 0:
        return cv2.warpPerspective(frame, H, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR)
    else:
        src = pts_src.copy()
        src[:, 0] = src[:, 0] + left_shift
        
        dst = pts_dts.copy()
        dst[:, 0] = dst[:, 0] + left_shift
        
        if back_project == 0:
            H = cv2.getPerspectiveTransform(src, dst)
        else:
            H = cv2.getPerspectiveTransform(dst, src)
        
        return cv2.warpPerspective(frame, H, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR)
    
def threshold(frame):
    frame_copy = frame.copy()
    
    #frame_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_GRAY2RGB)
    frame_hsv = cv2.cvtColor(frame_copy, cv2.COLOR_RGB2HSV)
    frame_hls = cv2.cvtColor(frame_copy, cv2.COLOR_RGB2HLS)
    
    #threshold hsv
    upper = np.array([150, 255, 255], dtype = "uint8")
#    lower = np.array([30, 50, 130], dtype = "uint8")
    lower = np.array([90, 60, 130], dtype = "uint8")
        
    mask = cv2.inRange(frame_hsv, lower, upper)
    
    frame_hsv = cv2.bitwise_and(frame, frame, mask = mask)
    
    frame_hsv = cv2.cvtColor(frame_hsv, cv2.COLOR_HSV2BGR)
    frame_hsv = cv2.cvtColor(frame_hsv, cv2.COLOR_BGR2GRAY)
    
    #threshold hls
    l = frame_hls[:,:,1]
    s = frame_hls[:,:,2]
    sobel = cv2.Sobel(l, cv2.CV_64F, 1, 1)
    sobel = np.abs(sobel)
    s_sobel = np.uint8(sobel/np.max(sobel))
    
    sx_bin = np.zeros_like(s_sobel)
    sx_bin[(s_sobel >= 100) & (s_sobel <= 255)] = 1
    
    s_binary = np.zeros_like(s)
    s_binary[(s >= 100) & (s <= 255)] = 1
    
    frame_hls = np.zeros_like(sx_bin)
    frame_hls[(s_binary == 1) | (sx_bin == 1)] = 255
    
    #threshold rgb
    lower_w_rgb = np.array([200, 200, 200], dtype="uint8")
    upper_w_rgb = np.array([255, 255, 255], dtype="uint8")
    mask = cv2.inRange(frame, lower_w_rgb, upper_w_rgb)
    w_rgb = cv2.bitwise_and(frame, frame, mask=mask)
    w_rgb = cv2.cvtColor(w_rgb, cv2.COLOR_RGB2GRAY)
    # y_rgb = bina(y_rgb, threshold)

    lower_y_rgb = np.array([0, 190, 190], dtype="uint8")
    upper_y_rgb = np.array([255, 255, 255], dtype="uint8")
    mask = cv2.inRange(frame, lower_y_rgb, upper_y_rgb)
    y_rgb = cv2.bitwise_and(frame, frame, mask=mask)
    y_rgb = cv2.cvtColor(y_rgb, cv2.COLOR_RGB2GRAY)

    mask_i = cv2.bitwise_or(w_rgb, y_rgb)
    i_rgb = cv2.bitwise_and(frame, frame, mask=mask_i)
    i_rgb = cv2.cvtColor(i_rgb, cv2.COLOR_BGR2GRAY)
    
    #combine
    mask_comb = cv2.bitwise_or(frame_hsv, frame_hls)
    mask_comb = cv2.bitwise_or(mask_comb, i_rgb)
    frame_out = cv2.bitwise_and(frame, frame, mask = mask_comb)
    
    frame_out = cv2.cvtColor(frame_out, cv2.COLOR_BGR2GRAY)
    
    return frame_out

def thresholding(frame, redThrsd=(200, 255), satThrsd=(90, 255), absThrsd=(20, 255), orient='x'):
    #thresholding HLS-satuarate channel
    img_hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    
    s_channel = img_hls[:,:, 2]
    
    bin_hls = np.zeros_like(s_channel)
    bin_hls[(s_channel > satThrsd[0]) & (s_channel <= satThrsd[1])] = 1
    
    #thresholding RGB-red channel
    red_channel = frame[:,:, 2]
    
    bin_rgb = np.zeros_like(red_channel)
    bin_rgb[(red_channel > redThrsd[0]) & (red_channel <= redThrsd[1])] = 1
    
    #thresholding gradient-absolute
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sobel = cv2.Sobel(grey, cv2.CV_64F, 1, 0, 3)
    sobel = np.abs(sobel)
    s_sobel = np.uint8(255 * sobel/np.max(sobel))
    
    sx_bin = np.zeros_like(s_sobel)
    sx_bin[(s_sobel > absThrsd[0]) & (s_sobel < absThrsd[1])] = 1
     
    #combined thresholding pipeline
    combined = np.zeros_like(bin_rgb)
    #test = np.ones_like(combined)

    combined[(bin_rgb == 1)&(bin_hls == 1) | sx_bin == 1] = 255
    #print(np.count_nonzero(combined == 0))
    
    #finalize
    padding = np.zeros((combined.shape[0], 170), dtype = np.uint8)
    combined = np.hstack((padding, combined, padding))
    
    return combined
    


    