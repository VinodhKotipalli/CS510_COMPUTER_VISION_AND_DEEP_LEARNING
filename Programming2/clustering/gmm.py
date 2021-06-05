import os
import sys
import math

filedir = os.path.dirname(os.path.abspath(__file__))
filedir = os.path.dirname(filedir).replace('\\', '/').replace('C:', '')
sys.path.insert(0, filedir)

from clustering.kmeans import *

from sklearn.mixture import GaussianMixture as skGMM

from scipy.stats import multivariate_normal as mvnorm


class GMM():
    def __init__(self,inputs,clusters,maxIterations=100,kmeansIterations=10,threshold=0.001,thresholdWindow=10):
        self.inputs = inputs
        self.clusters = clusters
        self.maxIterations = maxIterations
        self.kmeansIterations = kmeansIterations
        self.threshold = threshold
        self.thresholdWindow = thresholdWindow

    
    def train(self):
        inputs = self.inputs
        clusters = self.clusters
        self.initialize()

        probability = self.probability
        means = self.means
        weights = self.weights
        covariance = self.covariance
        nll = []
        labels = []


        for iter in range(self.maxIterations):
            labels.append(self.predict(inputs))
            iterNLL = self.calcNLL(inputs,means,covariance,weights)

            probability = self.getProbability(inputs,means,covariance,weights)
            means = self.getMeans(inputs,probability)
            weights = self.getWeights(probability)
            covariance = self.getCovariance(inputs,probability,means)
            nll.append(iterNLL)

            ###logic for early termination
            if len(nll) > self.thresholdWindow:
                window = self.thresholdWindow
            else:
                window = len(nll)
            if window > 1:
                windowSum = 0.0
                for i in range(len(nll)-1,len(nll) - window,-1):
                    windowSum += abs(nll[i]-nll[i-1])
                windowAverage = windowSum / window
                #print("iteration = %4d , NLL = %.4f, Average NLL shift = %.4f" %(iter,iterNLL,windowAverage))
                if windowAverage < self.threshold:
                    break
            #else:
                #print("iteration = %4d , NLL = %.4f" %(iter,iterNLL))
        
        labels.append(self.predict(inputs))
        iterNLL = self.calcNLL(inputs,means,covariance,weights)

        self.probability = probability
        self.means = means
        self.weights = weights
        self.covariance = covariance
        self.nll = np.array(nll)
        self.labelsPerIteration = np.array(labels)
    
    def predict(self,inputs,hardLabel=True):
        probability = self.getProbability(inputs,self.means,self.covariance,self.weights)
        if hardLabel == True:
            return np.array([np.argmin(i) for i in probability])
        else:
            return probability

    def getProbability(self,inputs,means,covariance,weights):
        clusters = self.clusters
        result = np.zeros(shape=(len(inputs),clusters))

        """
        for i in range(len(inputs)):
            for cluster in range(clusters):
                iProb = self.Gaussian(inputs[i], means[cluster], covariance[cluster])
                result[i,cluster] = weights[cluster]*iProb
        """

        for cluster in range(clusters):
            result[:,cluster] = weights[cluster]*mvnorm.pdf(inputs,means[cluster], covariance[cluster])

        for i in range(len(inputs)):
            result[i] /= np.sum(result[i])

        """
        for cluster in range(clusters):
            result[:,cluster] = result[:,cluster] / np.sum(result, axis=1)
        """
        return result

    def Gaussian(self,X,mean,sigma):
        cols = X.shape
        distance = X - mean
        sigma_inv = np.linalg.inv(sigma)
        result = np.exp(-1*0.5*np.matmul(np.matmul(distance,sigma_inv),distance.T))
        result = result * (1.0 / np.sqrt(np.power(2*np.pi,cols)*np.linalg.det(sigma)))
        return result


    def calcNLL(self,inputs,means,covariance,weights):
        clusters = self.clusters
        pdf = np.zeros(shape=(len(inputs),clusters))

        result = 0.0

        """
        for i in range(len(inputs)):
            for cluster in range(clusters):
                iProb = self.Gaussian(inputs[i], means[cluster], covariance[cluster])
                pdf[i,cluster] = weights[cluster]*iProb
            result -= math.log(np.sum(pdf[i,:]))
        """

        for cluster in range(clusters):
            pdf[:,cluster] = weights[cluster]*mvnorm.pdf(inputs,means[cluster], covariance[cluster])
        
        for i in range(len(inputs)):
            result -= math.log(np.sum(pdf[i,:]))

        return result

    def initialize(self):
        inputs = self.inputs
        clusters = self.clusters
        kmeans = KMeans(inputs,clusters,self.kmeansIterations)
        kmeans.train()
        labels = kmeans.predict(inputs)
        probability = np.zeros(shape=(len(inputs),clusters))
        for cluster in range(clusters):
            probability[labels==cluster,cluster] = 1.0
        
        means = self.getMeans(inputs,probability)
        weights = self.getWeights(probability)
        covariance = self.getCovariance(inputs,probability,means)

        self.probability = probability
        self.means = means
        self.weights = weights
        self.covariance = covariance


    def getMeans(self,inputs,probability):
        clusters = self.clusters
        result = np.matmul(probability.T,inputs)
        for cluster in range(clusters):
            result[cluster] = result[cluster]/sum(probability[:,cluster])
        return result
    
    def getWeights(self,probability):
        clusters = self.clusters
        result = np.zeros(shape=clusters)
        for cluster in range(clusters):
            result[cluster] = sum(probability[:,cluster])/np.sum(probability)
        return result

    def getCovariance(self,inputs,probability,means):
        clusters = self.clusters
        rows,cols = inputs.shape
        result = np.zeros((clusters, cols, cols))
        for cluster in range(clusters):
            for i in range(rows):
                iMean = np.reshape(inputs[i] - means[cluster],(cols,1))
                result[cluster] += probability[i,cluster]*np.dot(iMean,iMean.T)
            result[cluster] /= sum(probability[:,cluster])

        return result


if __name__ == '__main__':
    ipath = projdir.replace('\\', '/') + '/data/510_cluster_dataset.txt'
    inputs = np.array(pd.read_csv(ipath,header=None,delimiter=r"\s+"))
    r = 10
    scaleRatio = 0.50
    maxIterations = 500
    kmeansIterations = 10
    threshold = 0.001
    for k in [2,3,4]:
        models = []
        finalNLLs = []
        labelsPerIteration = []
        for _ in range(r):
            model = GMM(inputs,k,maxIterations,kmeansIterations,threshold)
            model.train()
            models.append(model)
            finalNLLs.append(model.nll[-1])
            labelsPerIteration.append(model.labelsPerIteration)

        plt.plot(range(len(finalNLLs)),finalNLLs)
        plt.xlabel('Run Number')
        plt.ylabel('Final Negative Log Likelihood')
        plt.title("Gaussian data:GMM(clusters = " + str(k) +") Final Negative Log Likelihood")
        plt.show()

        rmin = np.argmin(np.array(finalNLLs))
        model = models[rmin]        
        plt.plot(range(len(model.nll)),model.nll,'b', label=' Min NLL @ run = ' + str(rmin))

        print("=======================================================")
        print("Params(Clusters = %d): Min NLL @ run = %d" % (k,rmin))
        print("\t========================")
        print("\tmeans = \n\t%s" % str(model.means))
        print("\t========================")
        print("\tweights = \n\t%s" % str(model.weights))
        print("\t========================")
        print("\tcovariance  = \n\t%s" % str(model.covariance))
        print("=======================================================")

        
        rmax = np.argmax(np.array(finalNLLs))
        model = models[rmax]
        plt.plot(range(len(model.nll)),model.nll,'r', label=' Max NLL @ run = ' + str(rmax))

        print("=======================================================")
        print("Params(Clusters = %d): Max NLL @ run = %d" % (k,rmax))
        print("\t========================")
        print("\tmeans = \n\t%s" % str(model.means))
        print("\t========================")
        print("\tweights = \n\t%s" % str(model.weights))
        print("\t========================")
        print("\tcovariance  = \n\t%s" % str(model.covariance))
        print("=======================================================")

        plt.xlabel('Iteration Number')
        plt.ylabel('Negative Log Likelihood')
        plt.title("Gaussian data:GMM(clusters = " + str(k) +") Negative Log Likelihood")
        plt.legend(loc='upper right')
        plt.show()

        model = models[rmin]    
        labels = model.predict(inputs)
        for i in range(k):
            plt.scatter(inputs[labels == i , 0] , inputs[labels == i , 1] , label = i)
        plt.legend()
        plt.title("Gaussian data 2D Scatter:GMM(clusters = " + str(k) +") --> Min NLL @ run = " + str(rmin) + ")")
        plt.show()

        model = models[rmax]    
        labels = model.predict(inputs)
        for i in range(k):
            plt.scatter(inputs[labels == i , 0] , inputs[labels == i , 1] , label = i)
        plt.legend()
        plt.title("Gaussian data 2D Scatter:GMM(clusters = " + str(k) +") --> Max NLL @ run = " + str(rmax) + ")")
        plt.show()

    imageThresholds = [1,1,1]
    for fname in ["GMM_test1.jpg","GMM_test2.jpg"]:
        iname = fname.split('_test')[0] + fname.split('_test')[1].split('.')[0]
        ipath = projdir.replace('\\', '/') + '/data/' + fname

        src_image = cv2.imread(ipath)
        src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
        src_inputs = src_image.reshape((-1, 3))

        dsize = (int(src_image.shape[0] * scaleRatio), int(src_image.shape[1] * scaleRatio))
        image = cv2.resize(src_image,dsize)

        show_image(src_image,iname + ":orginal")

        inputs = image.reshape((-1, 3))

        i = 0
        for k in [3,5,10]:
            #threshold = imageThresholds[i]
            model = GMM(inputs,k,maxIterations,kmeansIterations,threshold)
            model.train()
            centroids=np.uint8(model.means)

            labels = model.predict(src_inputs)
            labels = labels.flatten()

            segmented_image = centroids[labels.flatten()]
            segmented_image = segmented_image.reshape(src_image.shape)
            show_image(segmented_image,iname + ":GMM(Custom Implementation) ---> Clusters = " + str(k))

            model = skGMM(n_components=k,covariance_type='tied').fit(inputs)
            centroids = np.uint8(model.means_)

            labels = model.predict(src_inputs)
            labels = labels.flatten()

            segmented_image = centroids[labels.flatten()]
            segmented_image = segmented_image.reshape(src_image.shape)
            show_image(segmented_image,iname + ":GMM(SciKit-Learn) ---> Clusters = " + str(k))


            i += 1


