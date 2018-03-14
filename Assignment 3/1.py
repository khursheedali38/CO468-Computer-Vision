
# coding: utf-8

# Test for following :
# 1 . Cluster number k
# 3 . training, testing set split

# In[135]:


import cv2
import itertools
import numpy as np
from glob import glob
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


classDict = {}
class_labelsTrain = np.array([])
descriptors = []


# In[136]:


# load train dataset for feature extraction and training
datasetTrain = {}
countTrain = 0
for x in glob("Dataset/new/train/*"):
    key = x.split("\\")[-1]
    datasetTrain[key] = []
    for image in glob("Dataset/new/train/" + key + "/*"):
        im = cv2.imread(image, 0)
        datasetTrain[key].append(im)
        countTrain+=1


# In[163]:


labels = {}
i = 0
for keys in datasetTrain.keys():
    labels[keys] = i
    i+=1
    


# In[162]:


# feature extraction using sift for all images
class_labels = 0
for key, imlist in datasetTrain.items():
    classDict[str(class_labels)] = key
    print(key)
    for image in imlist:
        class_labelsTrain = np.append(class_labelsTrain, class_labels)
        sift = cv2.xfeatures2d.SIFT_create()
        kp, descp = sift.detectAndCompute(image, None)
        descriptors.append(descp)
    class_labels+=1


# In[138]:


# Reshape SIFT features extracted for clustering
featuresTrain = np.array(descriptors[0])
for a in descriptors[1:]:
    featuresTrain = np.vstack((featuresTrain, a))


# In[140]:


# Cluster SIFT features detected
kmeansF = KMeans(n_clusters=20, random_state=0).fit(featuresTrain)
kmeans = kmeansF.predict(featuresTrain)


# In[142]:


# Creating a BOW representation of features, with frequency of each word in images
count = 0
n_clusters = 20
hist = np.array([np.zeros(n_clusters) for i in range(countTrain)])

for i in range(countTrain):
    for j in range(len(descriptors[i])):
        hist[i][kmeans[count + j]] += 1
    count+=1        


# In[143]:


# normalizing the BOW features before training
scaler = StandardScaler()
hist = scaler.fit_transform(hist)


# In[144]:


# Training the model
gnb = GaussianNB()
gnb.fit(hist, class_labelsTrain)


# In[145]:


# Testing the model
datasetTest = {}
countTest = 0
for x in glob("Dataset/new/test/*"):
    key = x.split("\\")[-1]
    datasetTest[key] = []
    for image in glob("Dataset/new/test/" + key + "/*"):
        im = cv2.imread(image, 0)
        datasetTest[key].append(im)
        countTest+=1


# In[146]:


# feature extraction using sift for all testimages
predictions = []

for word, imlist in datasetTest.items():
    for image in imlist:
        sift = cv2.xfeatures2d.SIFT_create()
        kp, descp = sift.detectAndCompute(image, None)
        histTest = np.array([0 for i in range(n_clusters)])
        for a in kmeansF.predict(descp):
            histTest[a]+=1
        histTest = scaler.fit_transform(histTest)
        labelPredictedTest = gnb.predict(histTest)
        predictions.append({
                    'image':image,
                    'class':labelPredictedTest,
                    'object_name':str(int(labelPredictedTest[0]))
                    })


# In[147]:


for each in predictions:
     plt.imshow(cv2.cvtColor(each['image'], cv2.COLOR_GRAY2RGB))
     plt.title(each['object_name'])
     plt.show()


# In[148]:


y_pred = np.vstack(ypred)
y_test = np.vstack(ytest)


# In[149]:


def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
   
    print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
class_names = []
class_names.append(classDict['0'])
class_names.append(classDict['1'])
class_names.append(classDict['2'])
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

plt.show()

