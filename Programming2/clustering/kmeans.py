import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import copy
import sys

PLUSINF = sys.float_info.max
MINUSINF = sys.float_info.min
np.set_printoptions(precision=3)

projdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def allPairDistances(A,B,squared=False):
    numpyA = np.array(A,dtype=float)
    numpyB = np.array(B,dtype=float)

    M = numpyA.shape[0]
    N = numpyB.shape[0]

    assert numpyA.shape[1] == numpyB.shape[1], f"The number of components for vectors in A \
        {numpyA.shape[1]} does not match that of B {numpyB.shape[1]}!"

    A_dots = (numpyA*numpyA).sum(axis=1).reshape((M,1))*np.ones(shape=(1,N))
    B_dots = (numpyB*numpyB).sum(axis=1)*np.ones(shape=(M,1))
    D_squared =  A_dots + B_dots -2*numpyA.dot(numpyB.T)

    if squared == False:
        zero_mask = np.less(D_squared, 0.0)
        D_squared[zero_mask] = 0.0
        return np.sqrt(D_squared)

    return D_squared

class KMeans():
    def __init__(self,inputs,clusters,maxIterations=100,threshold = 0.01):
        self.inputs = inputs
        self.clusters = clusters
        self.maxIterations = maxIterations
        self.threshold = threshold

    def train(self):
        inputs = self.inputs
        clusters = self.clusters
        kIndices = np.random.choice(range(len(inputs)), clusters, replace=False)
        centroids = inputs[kIndices,:]

        distances = allPairDistances(inputs,centroids)
        labels = np.array([np.argmin(i) for i in distances],dtype=int)

        for iter in range(self.maxIterations):
            old_centroids = copy.deepcopy(centroids)
            centroids = []
            for cluster in range(self.clusters):
                clusterCentroid = inputs[labels==cluster].mean(axis=0)
                centroids.append(clusterCentroid)
            self.centroids = np.vstack(centroids)

            distances = allPairDistances(inputs,centroids)
            self.labels = np.array([np.argmin(i) for i in distances])
            
            centroidDistances = allPairDistances(old_centroids,centroids)
            centroidShift = 0.0
            for i in range(clusters):
                centroidShift += centroidDistances[i,i]
            if iter == 0:
                old_centroidShift = centroidShift
            elif iter > 1 and centroidShift < self.threshold * old_centroidShift:
                break
            else:
                old_centroidShift = min(centroidShift,old_centroidShift)
        
            
        self.centroids = centroids
        

    def predict(self,inputs):
        distances = allPairDistances(inputs,self.centroids)
        return np.array([np.argmin(i) for i in distances])
    
    def error(self,inputs):
        distances = allPairDistances(inputs,self.centroids)
        labels = self.predict(inputs)
        totalError = np.zeros(shape=len(inputs))
        for cluster in range(self.clusters):
            totalError = distances[labels==cluster,cluster]
        return totalError.mean()

    
def show_image(image,title):
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title(title)
    plt.show()

if __name__ == '__main__':
    ipath = projdir.replace('\\', '/') + '/data/510_cluster_dataset.txt'
    inputs = np.array(pd.read_csv(ipath,header=None,delimiter=r"\s+"))
    r = 10
    scaleRatio = 0.30
    maxIterations = 100

    for k in [2,3,4]:
        models = []
        errors = []
        for _ in range(r):
            model = KMeans(inputs,k,maxIterations)
            model.train()
            models.append(model)
            errors.append(model.error(inputs))

        plt.plot(range(len(errors)),errors)
        plt.xlabel('Run Number')
        plt.ylabel('Mean Square Error')
        plt.title("Gaussian data:KMeans Mean Square Error")
        plt.ylim([0.75,4.75])
        plt.show()

        rmin = np.argmin(np.array(errors))
        labels = models[rmin].predict(inputs)
        for i in range(k):
            plt.scatter(inputs[labels == i , 0] , inputs[labels == i , 1] , label = i)
        plt.legend()
        plt.title("Gaussian data:KMeans(Min Error @ run = " + str(rmin) + ") ---> Clusters = " + str(k))
        plt.show()

        rmax = np.argmax(np.array(errors))
        labels = models[rmax].predict(inputs)
        for i in range(k):
            plt.scatter(inputs[labels == i , 0] , inputs[labels == i , 1] , label = i)
        plt.legend()
        plt.title("Gaussian data:KMeans(Max Error @ run = " + str(rmax) + ") ---> Clusters = " + str(k))
        plt.show()

    for fname in ["Kmean_img1.jpg","Kmean_img2.jpg"]:
        iname = fname.split('_img')[0] + fname.split('_img')[1].split('.')[0]
        ipath = projdir.replace('\\', '/') + '/data/' + fname

        src_image = cv2.imread(ipath)
        src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
        src_inputs = src_image.reshape((-1, 3))

        dsize = (int(src_image.shape[0] * scaleRatio), int(src_image.shape[1] * scaleRatio))
        image = cv2.resize(src_image,dsize)

        show_image(src_image,iname + ":orginal")

        inputs = image.reshape((-1, 3))

        for k in [5,10]:
            model = KMeans(inputs,k,maxIterations)
            model.train()
            centroids = np.uint8(model.centroids)

            labels = model.predict(src_inputs)
            labels = labels.flatten()

            segmented_image = centroids[labels.flatten()]
            segmented_image = segmented_image.reshape(src_image.shape)
            show_image(segmented_image,iname + ":KMeans ---> Clusters = " + str(k))




