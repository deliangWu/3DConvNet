from os import listdir
from os.path import isfile, join
import numpy as np
import re
import videoPreProcess


def sequence(fileName):
    return int(fileName[fileName.index("_") + 1 : fileName.rindex("_")])

def Label(fileName):
    return int(fileName[fileName.index(".") - 1 : fileName.index(".")])

    
def tenFold(filesSet):
    for i in range(1,11):
        trainingSet,testingSet = splitTrainingTesting(filesSet,i)
        print('***************************************************')
        


class ut_interaction:
    def __init__(self,path,frmSize):
        self._path = path
        self._frmSize = frmSize
        self._trainingSet = []
        self._testingSet= []
        files = np.array([f for f in listdir(path) if isfile(join(path,f)) and re.search('.avi',f) is not None])
        self._filesSet = np.array([[sequence(fileName), fileName, Label(fileName)] for fileName in files])
        self._videos = np.empty((0,16) + self._frmSize + (3,),dtype = np.uint8)
        self._labels = np.empty((0,6),dtype=np.float32)
        for file in self._filesSet:
            video = videoPreProcess.videoProcess(join(path,file[1]),self._frmSize)
            self._videos = np.append(self._videos,video,axis=0)
            labelCode = videoPreProcess.int2OneHot(int(file[2]),6)
            label = np.reshape(labelCode,(1,6))
            self._labels =  np.append(self._labels,label,axis=0)
            print(label)
        
    def splitTrainingTesting(self,n):
        testingIndex = [i for i,j in enumerate(self._filesSet[:,0]) if int(j) is n]
        trainingIndex = [i for i,j in enumerate(self._filesSet[:,0]) if int(j) is not n]
        self._trainingSet = [self._videos[trainingIndex], self._labels[trainingIndex]]
        self._testingSet = [self._videos[testingIndex], self._labels[testingIndex]]
        self._trainingIndex = np.array(range(self._trainingSet[0].shape[0]))
        return None
    
    def loadTraining(self):
        np.random.shuffle(self._trainingIndex)
        return(self._trainingSet[0][self._trainingIndex],self._trainingSet[1][self._trainingIndex])
    
    def loadTesting(self):
        return(self._testingSet)
        
        
                
class ut_interaction_set1(ut_interaction):
    def __init__(self,frmSize):
        path = "D:/Course/Final_Thesis_Project/project/datasets/UT_Interaction/ut-interaction_segmented_set1/segmented_set1"
        ut_interaction.__init__(self,path,frmSize)

class ut_interaction_set2(ut_interaction):
    def __init__(self,frmSize):
        path = "D:/Course/Final_Thesis_Project/project/datasets/UT_Interaction/ut-interaction_segmented_set2/segmented_set2"
        ut_interaction.__init__(self,path,frmSize)