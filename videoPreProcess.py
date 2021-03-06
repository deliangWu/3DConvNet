import numpy as np
import cv2
from functools import reduce
import tensorflow as tf
import time

def videoRead(fileName,grayMode=True,downSample = 1):
    cap = cv2.VideoCapture(fileName)
    firstFrame = True
    ret,frame = cap.read()
    frmNo = 0
    while(ret):
        if ret:
            if grayMode == True:
                gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                frame_4d = np.reshape(gray,((1,)+gray.shape+(1,)))
            else:
                frame_4d = np.reshape(frame,((1,)+frame.shape))
            if firstFrame:
                video = frame_4d
                firstFrame = False
            else:
                video = np.append(video,frame_4d,axis=0)
        ret,frame = cap.read()
        frmNo += 1
    if firstFrame is True:
        print('read video file ' + fileName + ' failed!!')
        return None
    else:
        frames = video[range(0,video.shape[0],downSample)]
        return frames 
    
def videoNorm(videoIn):
    vmax = np.amax(videoIn)
    vmin = np.amin(videoIn)
    vo = (videoIn.astype(np.float32) - vmin)/(vmax-vmin) * 255
    vo = vo.astype(np.uint8)
    return vo

def videoPlay(video,fps = 25):
    cv2.namedWindow('Video Player',cv2.WINDOW_AUTOSIZE)
    for i in range(video.shape[0]):
        img_show = video[i].copy()
        cv2.putText(img_show, str(i), (20,20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
        cv2.imshow('Video Player',img_show)
        if cv2.waitKey(int(1000/fps)) == 27:
            break
    cv2.destroyAllWindows()

def videoToImages(video):
    cv2.namedWindow('Video Player',cv2.WINDOW_AUTOSIZE)
    video_copy = video.copy()
    print(video_copy.shape)
    images = video_copy[0]
    for i in range(1,video_copy.shape[0]):
        images = np.append(images,video_copy[i],axis = 1)
    while (True):
        cv2.imshow('Video Player',images)
        if cv2.waitKey(40) == 27:
            break
    cv2.destroyAllWindows()
    
def videofliplr(videoIn):
    videoOut = np.array([np.fliplr(img) for img in videoIn])
    return videoOut

def downSampling(video,n=8):
    frameN = video.shape[0]
    #if (frameN > n):
    #    sample = np.sort(np.random.randint(0,frameN,n))
    #else:
    #    sample = np.sort(np.random.randint(0,frameN,int(frameN/16)*16))
    sample = range(int(frameN/n)*n)
    return video[sample]

def videoSave(video,fileName):
    frmSize = (video.shape[2],)+(video.shape[1],)
    if cv2.__version__  == '3.2.0':
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(fileName, fourcc, 20.0,frmSize)
    else:
        out = cv2.VideoWriter(fileName, cv2.cv.CV_FOURCC('X','V','I','D'),20,frmSize)
        
    for i in range(video.shape[0]):
        out.write(video[i])
        cv2.waitKey(33)
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
            if sum_diff > 200:
                img_save_4d = np.reshape(img,((1,) + img.shape))
                if firstFrame:
                    videoOut = img_save_4d
                    firstFrame = False 
                else:
                    videoOut = np.append(videoOut,img_save_4d,axis=0)
                frameAfterSimpilfied += 1
        img_pre = img
    
    if frameAfterSimpilfied < 64:
        videoOut = videoIn
    return videoOut

def batchFormat(videoIn):
    videoBatch = np.empty((0,16) + videoIn.shape[1:4],dtype = np.uint8)
    i = 0
    while(True):
        seq = np.arange(i*8,i*8 + 16)
        videoBatch = np.append(videoBatch,np.reshape(videoIn[seq],(1,) + videoIn[seq].shape),axis = 0)
        if i*8 + 16 >= videoIn.shape[0]:
            break
        i += 1
    return videoBatch

def videoFormat(batchIn):
    return np.reshape(batchIn,((batchIn.shape[0] * batchIn.shape[1]),) + batchIn.shape[2:5])

def videoRezise(videoIn,frmSize):
    def imgResize(img,frmSize):
        imgOut = np.reshape(np.array([119,136,153] * frmSize[0] * frmSize[1],dtype=np.uint8), frmSize)
        whRatio = img.shape[1] / img.shape[0]
        refRatio = frmSize[1] / frmSize[0]
        if whRatio < refRatio * 0.8:
            imgResized = cv2.resize(img,(int(img.shape[1] * frmSize[0] / (img.shape[0] * 2)) * 2 ,frmSize[0]), interpolation= cv2.INTER_AREA)
            imgOut[:, int((frmSize[1] - imgResized.shape[1])/2) : int((frmSize[1] + imgResized.shape[1])/2)] = imgResized
        elif whRatio > refRatio * 1.2:
            cropWidth = img.shape[0] * 1.2 * refRatio
            imgCrop = img[:,int((img.shape[1] - cropWidth)/2):int((img.shape[1] + cropWidth)/2)]
            imgOut = cv2.resize(imgCrop,(frmSize[1], frmSize[0]), interpolation= cv2.INTER_AREA)
        else:
            imgOut = cv2.resize(img,(frmSize[1], frmSize[0]), interpolation= cv2.INTER_AREA)
        return imgOut
    videoOut = np.array([imgResize(image,frmSize) for image in videoIn])
    return videoOut


def videoProcess(fileName,frmSize,RLFlipEn = True, batchMode = True):
    vIn = videoRead(fileName,grayMode=frmSize[2] == 1,downSample=2)
    print(vIn.shape)
    if vIn is not None:
        vRS = videoRezise(vIn,frmSize)
        vSimp = videoSimplify(vRS)
        vNorm = videoNorm(vSimp)
        vDS = downSampling(vNorm,8)
        vDS_Flipped = videofliplr(vDS)
        if RLFlipEn is True:
            vBatch = np.append(batchFormat(vDS),batchFormat(vDS_Flipped),axis=0)
        else:
            vBatch = batchFormat(vDS)
        
        if batchMode is True:
            return vBatch
        else:
            return vDS
    else:
        return None

def int2OneHot(din,range):
    code = np.zeros(range,dtype=np.float32)
    code[din] = 1
    return code
