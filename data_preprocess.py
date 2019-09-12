 # -*- coding: utf-8 -*-
"""
Created on Wed Jul 8 15:45:51 2019

@author: niuzhengnan
"""

import wave as we
import numpy as np
import matplotlib.pyplot as plt
#import tensorflow as tf 

def wave_as_array(wav_path):
    wavfile = we.open(wav_path,'rb')
    params = wavfile.getparams()
    #number of channels, 1 or 2
    #Sampling accuracy (byte width per frame)
    #sample rate（Hz）
    #number of audio frames
    channels, sampwidth, framesra, frameswav = params[0], params[1], params[2], params[3]
    datawav = wavfile.readframes(frameswav)
    wavfile.close()
    datause = np.fromstring(datawav, dtype = np.short)
    datause.shape = -1,channels
    datause = datause.T
    return datause, channels, sampwidth, framesra, frameswav

# Time in s
def array2input(datause,channels,framesra, time):
    nframes = time * framesra
    if channels != 1 and channels != 2:
        return None
    elif channels == 2:        
        channel1 = datause[0][:nframes]
        channel2 = datause[1][:nframes]
    else:        
        channel1 = datause[:nframes]
        channel2 = channel1
    res = np.append(channel1, channel2)
    if len(res)<nframes*2:
        print("r u there?")
        diff = nframes*2-len(res)
        for k in range(diff):
            res = np.append(res,0)
    if len(res)>nframes*2:
        print("r u there??")
        res = res[:nframes*2]
    return res

#if max_label = 10, then label should be 0 - 9 (totally 10)
def add_into_Trainset(res, label, path_x_Train = "x_train.npy" , path_y_Train = "y_train.npy"):
    try:
        x_train = np.load(path_x_Train)
        np.save(path_x_Train,np.concatenate([res[np.newaxis,:],x_train]))
    except FileNotFoundError:
        np.save(path_x_Train, res[np.newaxis,:])
        
    #if not 0<=label<=max_label:
    #    print("label error!")
    #else:
    #    label_array = np.zeros(max_label,)
    #    label_array[label] = 1    
    
    try:
        y_train = np.load(path_y_Train).tolist()
        y_train[0].append(label)
        np.save(path_y_Train, y_train)
    except FileNotFoundError:
        np.save(path_y_Train, [[label]])
        
    return None

def add_into_Testset(res, label, path_x_Test = "x_test.npy" , path_y_Test = "y_test.npy"):
    try:
        x_test = np.load(path_x_Test)
        np.save(path_x_Test,np.concatenate([res[np.newaxis,:],x_test]))
    except FileNotFoundError:
        np.save(path_x_Test, res[np.newaxis,:])
        
    #if not 0<=label<=max_label:
    #    print("label error!")
    #else:
    #    label_array = np.zeros(max_label,)
    #    label_array[label] = 1    
    
    try:
        y_test = np.load(path_y_Test).tolist()
        y_test[0].append(label)
        np.save(path_y_Test, y_test)
    except FileNotFoundError:
        np.save(path_y_Test, [[label]])
        
    return None

#if __name__ == "__main__": 
# label 0 : Bike  110/18
#    for i in range(1,110):
#        datause, channels, sampwidth, framesra, frameswav = wave_as_array('D:\\training_data\\Bike\Bike_%d.wav'%i)
#        res = array2input(datause,channels,framesra, 10)
#        add_into_Trainset(res, 0, path_x_Train = "x_train.npy" , path_y_Train = "y_train.npy")
#    for i in range(111,128):
#        datause, channels, sampwidth, framesra, frameswav = wave_as_array('D:\\training_data\\Bike\Bike_%d.wav'%i)
#        res = array2input(datause,channels,framesra, 10)
#        add_into_Testset(res, 0, path_x_Test = "x_test.npy" , path_y_Test = "y_test.npy") 
# label 1 : Car 125/20    
#    for i in range(1,125):
#        datause, channels, sampwidth, framesra, frameswav = wave_as_array('D:\\training_data\\Car\Car_%d.wav'%i)
#        res = array2input(datause,channels,framesra, 10)
#        add_into_Trainset(res, 1, path_x_Train = "x_train.npy" , path_y_Train = "y_train.npy")
#    for i in range(125,145):
#        datause, channels, sampwidth, framesra, frameswav = wave_as_array('D:\\training_data\\Car\Car_%d.wav'%i)
#        res = array2input(datause,channels,framesra, 10)
#        add_into_Testset(res, 1, path_x_Test = "x_test.npy" , path_y_Test = "y_test.npy")
# label 2: Emergency vehicle  (107/13)
    
#    for i in range(1,108):
#        datause, channels, sampwidth, framesra, frameswav = wave_as_array('D:\\training_data\\Emergencyvehicle\Emergencyvehicle_%d.wav'%i)
#        res = array2input(datause,channels,framesra, 10)
#        add_into_Trainset(res, 2, path_x_Train = "x_train.npy" , path_y_Train = "y_train.npy")
#    for i in range(108,121):
#        datause, channels, sampwidth, framesra, frameswav = wave_as_array('D:\\training_data\\Emergencyvehicle\Emergencyvehicle_%d.wav'%i)
#        res = array2input(datause,channels,framesra, 10)
#        add_into_Testset(res, 2, path_x_Test = "x_test.npy" , path_y_Test = "y_test.npy")
    
 # Label 3:Horn 141/30
    
#    for i in range(1,142):
#        datause, channels, sampwidth, framesra, frameswav = wave_as_array('D:\\training_data\\Horn\Horn_%d.wav'%i)
#        res = array2input(datause,channels,framesra, 10)
#        add_into_Trainset(res, 3, path_x_Train = "x_train.npy" , path_y_Train = "y_train.npy")
#    for i in range(142,172):
#        datause, channels, sampwidth, framesra, frameswav = wave_as_array('D:\\training_data\\Horn\Horn_%d.wav'%i)
#        res = array2input(datause,channels,framesra, 10)
#        add_into_Testset(res, 3, path_x_Test = "x_test.npy" , path_y_Test = "y_test.npy")
        
        
 # Label 4 : Motorcycle 107/13
    
#    for i in range(1,108):
#        datause, channels, sampwidth, framesra, frameswav = wave_as_array('D:\\training_data\\Motorcycle\Motorcycle_%d.wav'%i)
#        res = array2input(datause,channels,framesra, 10)
#        add_into_Trainset(res, 4, path_x_Train = "x_train.npy" , path_y_Train = "y_train.npy")
#    for i in range(142,121):
#        datause, channels, sampwidth, framesra, frameswav = wave_as_array('D:\\training_data\\Motorcycle\Motorcycle_%d.wav'%i)
#        res = array2input(datause,channels,framesra, 10)
#        add_into_Testset(res, 4, path_x_Test = "x_test.npy" , path_y_Test = "y_test.npy")    
#        
# # Label 5 : Noise 147/30
#    
#    for i in range(1,148):
#        datause, channels, sampwidth, framesra, frameswav = wave_as_array('D:\\training_data\\Noise\\Noise_%d.wav'%i)
#        res = array2input(datause,channels,framesra, 10)
#        add_into_Trainset(res, 5, path_x_Train = "x_train.npy" , path_y_Train = "y_train.npy")
#    for i in range(148,178):
#        datause, channels, sampwidth, framesra, frameswav = wave_as_array('D:\\training_data\\Noise\\Noise_%d.wav'%i)
#        res = array2input(datause,channels,framesra, 10)
#        add_into_Testset(res, 5, path_x_Test = "x_test.npy" , path_y_Test = "y_test.npy")     
#        
# # Label 6 : rail 110/13
#    
#    for i in range(1,111):
#        datause, channels, sampwidth, framesra, frameswav = wave_as_array('D:\\training_data\\Rail\\rail_%d.wav'%i)
#        res = array2input(datause,channels,framesra, 10)
#        add_into_Trainset(res, 6, path_x_Train = "x_train.npy" , path_y_Train = "y_train.npy")
#    for i in range(111,124):
#        datause, channels, sampwidth, framesra, frameswav = wave_as_array('D:\\training_data\\Rail\\rail_%d.wav'%i)
#        res = array2input(datause,channels,framesra, 10)
 #       add_into_Testset(res, 6, path_x_Test = "x_test.npy" , path_y_Test = "y_test.npy")  
#    for i in range(1,8):
#        datause, channels, sampwidth, framesra, frameswav = wave_as_array('D:\\training_data\\Car\Car_%d.wav'%i)
#        res = array2input(datause,channels,framesra, 10)
#        add_into_Trainset(res, 1, path_x_Train = "x_train.npy" , path_y_Train = "y_train.npy")

#    for i in range(9,101):
#        datause, channels, sampwidth, framesra, frameswav = wave_as_array('D:\\training_data\\Car\Car_%d.wav'%i)
#        res = array2input(datause,channels,framesra, 10)
#        add_into_Trainset(res, 1, path_x_Train = "x_train.npy" , path_y_Train = "y_train.npy")
     
    
    
    
    
    
    
    
    
    
#datause, channels, sampwidth, framesra, frameswav = wave_as_array('dzq.wav')
#time = np.arange(0,frameswav) * (1.0/framesra)
#plt.title("frames")
#plt.subplot(211)
#plt.plot(time,datause[0],color='green')
#plt.subplot(212)
#plt.plot(time,datause[1],color='red')    
#plt.show()