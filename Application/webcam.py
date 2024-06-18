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

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras

# Display
import matplotlib as mpl

import pickle

def main():

    # used to record the time when we processed last frame 
    prev_frame_time = 0
        
    # used to record the time at which we processed current frame 
    new_frame_time = 0
    
    with open("er_index.pickle", "rb") as handle:
        er_index = pickle.load(handle)
    
    def imageToLabel(imgArr, new_model):
        """Predict class from image array.
    
        Parameters
        ----------
        
        imgArr : array-like
            Array of image pixel values
            
        new_model : Keras model
            Model used for classification
            
        Returns
        ----------
        label : int
            Classification or prediction
            
        probability : float
            Probability produced from predicting the image array
        """
        imgArrNew = imgArr.reshape(1, 250, 250, 3)
        prediction = new_model.predict(imgArrNew, verbose=0)
        if prediction[0][0] >= er_index:
            label = 1
        else:
            label = 0
        probability = prediction[0][0]
        return label, probability
    
    def get_img_array(img_path, size):
        """Return image array from image path.
    
        Parameters
        ----------
        
        img_path : string
            Path of image
            
        size : tuple
            Target dimensions of image
            
        Returns
        ----------
        array-like
            Array containing pixel values of image
        """
        img_path = cv2.resize(img_path, size)
        img_path = np.expand_dims(img_path, axis=0)
        return img_path
    
    def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
        """Create Gradcam heatmap from the model and the image.
        
        Parameters
        ----------
        
        img_array : array-like
            Array containing pixel values of image
            
        model : Keras model
            Keras CNN model
    
        last_conv_layer_name : string
            Name of last convolutional layer of CNN model
    
        pred_index : int
            Prediction value which can be 0 or 1
            
        Returns
        ----------
        array-like
            Array containing pixel values of heatmap
        """
        # First, we create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions
        grad_model = keras.models.Model(
            model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
        )
    
        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]
    
        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)
    
        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
    
        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    
    def save_and_display_gradcam(img, heatmap, alpha=0.9):
        """Display heatmap superimposed on image.
    
        Parameters
        ----------
        
        img_path : string
            Path of image
            
        heatmap : array-like
            Array containing pixel values of heatmap
        
        alpha : float
            Alpha value for the heatmap
            
        Returns
        ----------
        
        superimposed_img : array-like
            Array of superimposed heatmap and image
        
        """
    
        # Rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)
    
        # Use jet colormap to colorize heatmap
        jet = mpl.colormaps["jet_r"]
    
        # Use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
    
        # Create an image with RGB colorized heatmap
        jet_heatmap = keras.utils.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = keras.utils.img_to_array(jet_heatmap)
    
        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap * alpha + img
    
        return superimposed_img
        
    # define a video capture object 
    vid = cv2.VideoCapture(0) 
    
    xc_model_custom = keras.models.load_model("model_checkpoint_xc_direct_unfrozen.h5")
    
    
    #result = cv2.VideoWriter('filename.avi',  
        #                       cv2.VideoWriter_fourcc(*'MJPG'), 
        #                      10, size) 
        
    preprocess_input = keras.applications.xception.preprocess_input
    
    last_conv_layer_name = "block14_sepconv2_act"
        
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
        
        
        heatmap = make_gradcam_heatmap(preprocess_input(get_img_array(frame, (250, 250))), xc_model_custom.get_layer('xception'), last_conv_layer_name, pred_index=label)
        
        frame_colored = save_and_display_gradcam(frame, heatmap)
        
        cv2.putText(frame_colored, fps, (20, 70), font, 2, (255, 255, 255), 3, cv2.LINE_AA) 
        
        cv2.rectangle(frame_colored, (985, 20), (1900, 80), (0, 0, 0, 0.5), -1)
        
        cv2.putText(frame_colored, now_date, (1000, 70), font, 2, (255, 255, 255), 3, cv2.LINE_AA) 
        
        cv2.rectangle(frame_colored, (10, 1000), (640, 1070), (0, 0, 0, 0.5), -1)
        
        cv2.putText(frame_colored, title, (20, 1050), font, 2, (255, 255, 255), 3, cv2.LINE_AA) 
        
        cv2.rectangle(frame_colored, (820, 950), (1920, 1080), (0, 0, 0, 0.2), -1)
        
        if label == 1:
            prob = np.round(prob * 100, 2)
            text_smoke = f"Smoking ({prob:.2f}%)"
            cv2.putText(frame_colored, text_smoke, (910, 1050), font_2, 3, (0, 0, 255), 3, cv2.LINE_AA)
        
        else:
            prob = np.round(prob * 100, 2)
            text_smoke = f"No Smoking ({prob:.2f}%)"
            cv2.putText(frame_colored, text_smoke, (840, 1050), font_2, 3, (85, 214, 31), 3, cv2.LINE_AA)
        
        # Display the resulting frame 
        img_normalized = cv2.normalize(frame_colored, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        cv2.imshow('frame', img_normalized) 
            
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