
# coding: utf-8

# In[ ]:


from keras.models import model_from_json
import numpy as np
from scipy import misc
import os


# In[1]:


def one_hot(Y):
    C = len(list(set(Y)))
    Y_OH = np.eye(C)[Y.reshape(-1)]

    return Y_OH


# In[2]:


def evaluate_model(model, X_train, Y_train, X_test, Y_test, maxItem = 100):
    #test model on training data set and testing data set
    nTrain = min([maxItem, X_train.shape[0]])
    print('Performance on Training data set (%d)' % nTrain)
    preds = model.evaluate(X_train[0:nTrain], Y_train[0:nTrain])
    print ("Loss = " + str(preds[0]))
    print ("Train Accuracy = " + str(preds[1]))

    nTest = min([maxItem, X_test.shape[0]])
    print('Performance on Testing data set (%d)' % nTest)
    preds = model.evaluate(X_test[0:nTest], Y_test[0:nTest])
    print ("Loss = " + str(preds[0]))
    print ("Test Accuracy = " + str(preds[1]))


# In[2]:


def load_QRIFaceData(datapath = './QRI_faces_clean/'):
    
    FaceData = []
    labels = []
    name_list = []
    facefiles = os.listdir(datapath)
    for i,facefile in enumerate(facefiles):
        im = misc.imread(datapath+facefile)
        im = misc.imresize(im,(96,96,3))
        im = np.around(np.transpose(im, (2,0,1))/255.0, decimals=12)
        #im = np.transpose(im, (2,0,1))
        #im = np.transpose(im,[2,0,1])
        FaceData.append(im)
        labels.append(i)
        FaceData.append(im[:,:,::-1])
        labels.append(i)
        name_list.append(facefile[0:-4])

    labels = np.array(labels)
    FaceData = np.array(FaceData)
    labels_OH = one_hot(labels)
    return FaceData, labels,labels_OH, name_list


# In[3]:


def load_FaceData(nFace, datapath = './../Dataset/CaltechFaceSet/processed/'):
    FaceData = []
    for iFace in range(nFace):
        im = misc.imread(datapath+'cf%04d.png' % iFace)
        im = misc.imresize(im,(96,96,3))
        im = np.around(np.transpose(im, (2,0,1))/255.0, decimals=12)
        #im = np.transpose(im,[2,0,1])
        FaceData.append(im)
    FaceData = np.array(FaceData)
    labels = np.load(datapath+'Labels.npy')
    labels_OH = one_hot(labels)
    return FaceData, labels,labels_OH


# In[4]:


def face_dist(f1,f2,w=1):
    if w==1:
        w = np.ones(f1.shape)
    if len(f1.shape)==len(f2.shape)==1:
        dist = np.linalg.norm((f1-f2)*w)
    else:
        dist = np.linalg.norm((f1-f2)*w, axis=1)
    return dist


# In[5]:


def distance_based_prediction(FaceEmbedding, Labels, f):
    
    Dists = face_dist(FaceEmbedding,f)
    
    i = np.argmin(Dists)
    
    return Labels[i]


# In[1]:


def load_base_model():
    model = model_from_json(open('./FaceNetModel/FaceNet.json').read())
    model.load_weights('./FaceNetModel/FaceNet_weights.h5')
    return model

