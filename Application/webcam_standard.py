#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 22:40:11 2023

@author: pmmedina
"""


# import the opencv library 
import cv2 

import tensorflow as tf

import numpy as np

import time 

import datetime

import os

import pickle

os.environ["KERAS_BACKEND"] = "tensorflow"

def main():

    # used to record the time when we processed last frame 
    prev_frame_time = 0
      
    # used to record the time at which we processed current frame 
    new_frame_time = 0
    
    with open("er_index.pickle", "rb") as handle:
        er_index = pickle.load(handle)
    
    
    def imageToLabel(imgArr, new_model):
        imgArrNew = imgArr.reshape(1, 250, 250, 3)
        prediction = new_model.predict(imgArrNew, verbose=0)
        if prediction[0][0] >= er_index:
            label = 1
        else:
            label = 0
        probability = prediction[0][0]
        return label, probability
      
    # define a video capture object 
    vid = cv2.VideoCapture(0) 
    
    xc_model_custom = tf.keras.models.load_model('model_checkpoint_xc_direct_unfrozen.h5')
    
    #result = cv2.VideoWriter('filename.avi',  
     #                       cv2.VideoWriter_fourcc(*'MJPG'), 
      #                      10, size) 
    
    while(True): 
          
        # Capture the video frame 
        # by frame 
        ret, frame = vid.read() 
        
        resizedImg = cv2.resize(frame, (250, 250))
        
        label, prob = imageToLabel(resizedImg, xc_model_custom)
        
        now_date = datetime.datetime.now().strftime("%m-%d-%Y %a %X")
        
        title = "Smokeception Cam"
      
        # font which we will be using to display FPS 
        font = cv2.FONT_HERSHEY_SIMPLEX 
        
        font_2 = cv2.FONT_HERSHEY_DUPLEX
        # time when we finish processing for this frame 
        new_frame_time = time.time() 
      
        # Calculating the fps 
      
        # fps will be number of frame processed in given time frame 
        # since their will be most of time error of 0.001 second 
        # we will be subtracting it to get more accurate result 
        fps = 1/(new_frame_time-prev_frame_time) 
        prev_frame_time = new_frame_time 
      
        # converting the fps into integer 
        fps = int(fps) 
      
        # converting the fps to string so that we can display it on frame 
        # by using putText function 
        fps = "FPS: " + str(fps) 
      
        # putting the FPS count on the frame 
        
        cv2.putText(frame, fps, (20, 70), font, 2, (255, 255, 255), 3, cv2.LINE_AA) 
        
        cv2.rectangle(frame, (985, 20), (1900, 80), (0, 0, 0, 0.5), -1)
        
        cv2.putText(frame, now_date, (1000, 70), font, 2, (255, 255, 255), 3, cv2.LINE_AA) 
        
        cv2.rectangle(frame, (10, 1000), (640, 1070), (0, 0, 0, 0.5), -1)
        
        cv2.putText(frame, title, (20, 1050), font, 2, (255, 255, 255), 3, cv2.LINE_AA) 
        
        cv2.rectangle(frame, (820, 950), (1920, 1080), (0, 0, 0, 0.2), -1)
        
        if label == 1:
            prob = np.round(prob * 100, 2)
            text_smoke = f"Smoking ({prob:.2f}%)"
            cv2.putText(frame, text_smoke, (910, 1050), font_2, 3, (0, 0, 255), 3, cv2.LINE_AA)
        
        else:
            prob = np.round(prob * 100, 2)
            text_smoke = f"No Smoking ({prob:.2f}%)"
            cv2.putText(frame, text_smoke, (840, 1050), font_2, 3, (85, 214, 31), 3, cv2.LINE_AA)
        
        # Display the resulting frame 
        cv2.imshow('frame', frame) 
          
        #result.write(frame) 
        
        # the 'q' button is set as the 
        # quitting button you may use any 
        # desired button of your choice 
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
      
    # After the loop release the cap object 
    vid.release() 
    #result.release() 
    # Destroy all the windows 
    cv2.destroyAllWindows() 
    
if __name__ == '__main__':
    main()