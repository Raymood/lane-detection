# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 19:11:52 2019

@author: shiha
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import img_processing as ip
from sliding_windows import*

def draw_area(top_down, top_down_rgb, left_fit, right_fit):
    # Generate x and y values for plotting
    ploty = np.linspace(0, top_down.shape[0] - 1, top_down.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(top_down).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    #cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    img_poly = cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    # H, status = cv2.findHomography(pts_src, pts_dst)
    # img_dts_new = cv2.warpPerspective(img_threshold, H, (1280, 420))
    # Combine the result with the original image
    #result = cv2.addWeighted(top_down_rgb, 0.5, img_poly, 0.5, 0)
    return img_poly

def curv_position(top_down,left_fit,right_fit):
    ploty = np.linspace(0, top_down.shape[0] - 1, top_down.shape[0])
    leftx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    rightx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    y_eval = np.max(ploty)

    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)

    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute( 2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])

    curvature = ((left_curverad - right_curverad) / 2)

    lane_width = np.absolute(leftx[top_down.shape[0] - 1] - rightx[top_down.shape[0] - 1])
    lane_xm_per_pix = 3.7 / lane_width
    vehicle = (((leftx[top_down.shape[0] - 1] + rightx[top_down.shape[0] - 1] - 170) * lane_xm_per_pix) / 2.)
    center = ((top_down.shape[1] * lane_xm_per_pix) / 2.)
    off_set = vehicle - center
    return curvature, off_set


cap = cv2.VideoCapture('project_video.mp4')

ret, frame = cap.read()
frame = cv2.resize(frame, (0,0), fx = 0.5, fy = 0.5)

writer = cv2.VideoWriter('lane_detection_pure.mp4', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame.shape[1],frame.shape[0]))

while 1:
    ret, frame = cap.read()
    
    if ret is not True:
        break
    #pre-process of each frame    
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (0,0), fx = 0.5, fy = 0.5)

    frame_undist = ip.undistort(frame)
    frame_eql = ip.histo_eql(frame_undist)
    
    #thresholding
    frame_thresh = ip.thresholding(frame_undist)
    
    frame_warpped = ip.homography(frame_thresh, left_shift = 170)
    
    frame_warpped_rgb = ip.homography(frame_undist)
    padding = np.zeros((frame_warpped_rgb.shape[0], 170, 3), dtype = np.uint8)
    frame_warpped_rgb = np.hstack((padding, frame_warpped_rgb, padding))
    
    left_fit, right_fit, frame_line = find_line(frame_warpped)
    
    curvature, offset = curv_position(frame_warpped, left_fit, right_fit)
    
    frame_poly = draw_area(frame_warpped, frame_warpped_rgb, left_fit, right_fit)
    
    project_back = ip.homography(frame_poly, left_shift = 170, back_project = 1)[:, 170:170+frame_undist.shape[1], :]
    
    frame_final = cv2.addWeighted(frame_undist, 1.0, project_back, 0.7, 0.0)
    
    #frame_final = cv2.resize(frame_final, (0,0), fx = 2.0, fy = 2.0)
    #writer.write(frame_final)
    
    if curvature < 0 and curvature >= -100:
        cv2.putText((frame_final), 'left turn', (100, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    elif curvature > 0 and curvature <= 200:
        cv2.putText((frame_final), 'right turn', (100, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText((frame_final), 'straight', (100, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.putText(frame_final, 'curvature: {:.0f}m'.format(curvature), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    cv2.putText(frame_final, 'offset: {:.2f}m'.format(offset), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)



    #cv2.imwrite('frame0.jpg', frame)
    cv2.imshow('ori', frame_undist)
    cv2.imshow('', frame_warpped)
    cv2.imshow('threshold', frame_thresh)
    cv2.imshow('sliding_window', frame_line)
    cv2.imshow('img_poly', frame_poly)
    cv2.imshow('img_final', frame_final)
    
    if cv2.waitKey(1)&0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()