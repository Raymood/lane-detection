# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 13:51:18 2019

@author: shiha
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

store_l_x = []
store_l_y = []
store_r_x = []
store_r_y = []
left_fit_a = []
left_fit_b = []
left_fit_c = []
right_fit_a = []
right_fit_b = []
right_fit_c = []

def find_line(top_down):
    # 直方图
    histogram = np.sum(top_down[top_down.shape[0]//2:,:],axis=0)
    #print(histogram)
    top_down = np.dstack((top_down, top_down, top_down))
    # 车道中点
    midpoint = np.int(histogram.shape[0]/2)

    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint: 800]) + midpoint
    #print("leftx_base:", leftx_base)
    #print("rightx_base:", rightx_base)
    #plt.plot(np.linspace(0, histogram.shape[0], histogram.shape[0]), histogram)

    nwindows = 30

    window_height = np.int(top_down.shape[0]/nwindows)

    nonzero = top_down.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base

    width = 50
    min_pix = 60

    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = top_down.shape[0] - (window+1) * window_height
        #print(top_down.shape)
        win_y_high = top_down.shape[0] - window * window_height
        win_xleft_low = leftx_current - width
        win_xleft_high = leftx_current + width
        win_xright_low = rightx_current - width
        win_xright_high = rightx_current + width
        
        cv2.rectangle(top_down,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 1) 
        cv2.rectangle(top_down,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 1)

        valid_l_pxs = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        valid_r_pxs = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(valid_l_pxs)
        right_lane_inds.append(valid_r_pxs)

        if len(valid_l_pxs) > min_pix:
            leftx_current = np.int(np.mean(nonzerox[valid_l_pxs]))
        if len(valid_r_pxs) > min_pix:
            rightx_current = np.int(np.mean(nonzerox[valid_r_pxs]))
            

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    if len(leftx) != 0:
        store_l_x.append(leftx)
    lefty = nonzeroy[left_lane_inds]
    if len(lefty) != 0:
        store_l_y.append(lefty)
    rightx = nonzerox[right_lane_inds]
    if len(rightx) != 0:
        store_r_x.append(rightx)
    righty = nonzeroy[right_lane_inds]
    if len(righty) != 0:
        store_r_y.append(righty)
    
    if len(leftx) != 0 or len(lefty) != 0 or len(rightx) != 0 or len(righty) != 0:
        leftx = store_l_x[-1]
        lefty = store_l_y[-1]
        rightx = store_r_x[-1]
        righty = store_r_y[-1]
    


    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    left_fit_a.append(left_fit[0])
    left_fit_b.append(left_fit[1])
    left_fit_c.append(left_fit[2])
    right_fit_a.append(right_fit[0])
    right_fit_b.append(right_fit[1])
    right_fit_c.append(right_fit[2])
    
    left_fit[0] = np.mean(left_fit_a[-20:])
    left_fit[1] = np.mean(left_fit_b[-20:])
    left_fit[2] = np.mean(left_fit_c[-20:])
    
    right_fit[0] = np.mean(right_fit_a[-20:])
    right_fit[1] = np.mean(right_fit_b[-20:])
    right_fit[2] = np.mean(right_fit_c[-20:])
    
    
    top_down[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    top_down[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    return left_fit, right_fit, top_down