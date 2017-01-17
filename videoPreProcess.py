import numpy as np
import cv2
from functools import reduce
import tensorflow as tf
import time
import random

def evenSample(m,n):
    o = range(0,n)
    o =  [i*int(m/n) for i in o]   
    return o


def videoRead(fileName):
    cap = cv2.VideoCapture(fileName)
    firstFrame = True
    ret,frame = cap.read()
    while(ret):
        frame_4d = np.reshape(frame,((1,)+frame.shape))
        if firstFrame:
            video = frame_4d
            firstFrame = False
        else:
            video = np.append(video,frame_4d,axis=0)
        ret,frame = cap.read()
    cap.release()
    if firstFrame is True:
        print('read video file ' + fileName + ' failed!!')
        return None
    else:
        frames = video[xrange(0,video.shape[0],2)]
        return frames 

def videoPlay(video):
    cv2.namedWindow('Video Player',cv2.WINDOW_AUTOSIZE)
    for i in range(video.shape[0]):
        img_show = video[i].copy()
        cv2.putText(img_show, str(i), (20,20), cv2.FONT_HERSHEY_COMPLEX, .5, (255,0,0), 1, cv2.LINE_AA)
        cv2.imshow('Video Player',img_show)
        if cv2.waitKey(40) == 27:
            break
    cv2.destroyAllWindows()

def downSampling(video,n = 16):
    frameN = video.shape[0]
    if (frameN > n):
        #sample = np.sort(np.random.randint(0,frameN,64))
        sample = np.array(evenSample(frameN,n))
    else:
        sample = np.sort(np.random.randint(0,frameN,int(frameN/16)*16))
    return video[sample]

def videoSave(video,fileName):
    frmSize = tuple(reversed(video.shape[2:4]))
    if cv.__version__ is '3.2.0':
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(fileName, fourcc, 20.0,frmSize)
    else:
        out = cv2.VideoWriter(fileName, cv2.cv.CV_FOURCC('X','V','I','D'),20,frmSize)
        
    for frame in video:
        out.write(frame)
    out.release()

def videoSimplify(videoIn):
    frames = videoIn
    firstFrame = True
    frameAfterSimpilfied = 0
    for i in range(frames.shape[0]):
        img = frames[i] 
        if i > 0:
            # calculate the difference between two adjacent frames
            img_diff = cv2.absdiff(img,img_pre)
            sum_diff = np.sum(img_diff)/1000
            
            # save current frame if it's abs_diff larger than a threshold 
            if sum_diff > 400:
                img_save_4d = np.reshape(img,((1,) + img.shape))
                if firstFrame:
                    videoOut = img_save_4d
                    firstFrame = False 
                    frameAfterSimpilfied = 1
                else:
                    videoOut = np.append(videoOut,img_save_4d,axis=0)
                    frameAfterSimpilfied = frameAfterSimpilfied + 1
        img_pre = img
    
    if frameAfterSimpilfied < 32:
        videoOut = videoIn
    return videoOut

def batchFormat(videoIn):
    return np.reshape(videoIn,((int(videoIn.shape[0]/16),) + (16,) + videoIn.shape[1:4]))

def videoFormat(batchIn):
    return np.reshape(batchIn,((batchIn.shape[0] * batchIn.shape[1]),) + batchIn.shape[2:5])

def videoRezise(videoIn,frmSize):
    videoOut = np.empty((0,) + frmSize + (3,),dtype=np.uint8)
    for image in videoIn:
        resizedImg = cv2.resize(image,tuple(reversed(frmSize)),interpolation=cv2.INTER_AREA)
        videoOut = np.append(videoOut,np.reshape(resizedImg,(1,)+resizedImg.shape),axis=0)
    return videoOut