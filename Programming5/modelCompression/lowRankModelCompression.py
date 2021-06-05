import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
import numpy as np
import pandas as pd
import seaborn as sns

import os
import sys
import copy
import matplotlib.pyplot as plt
import math

filedir = os.path.dirname(os.path.abspath(__file__))
filedir = os.path.dirname(filedir).replace('\\', '/').replace('C:', '')
sys.path.insert(0, filedir)

projdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def writeModelSummary(model,outpath,title):
    outfile = open(outpath, 'w')
    outfile.write("%s Model Summary" %(title))
    model.summary(print_fn=lambda x: outfile.write(x + '\n'))
    outfile.close()

def show_model_metrics(model,inputs_test, targets_test,inputs_train, targets_train,title=None,classes=range(10),batch_size=256):
    predicted_labels = np.argmax(model.predict(inputs_test),axis=1)
    target_labels = np.argmax(targets_test,axis=1)

    loss_test,accuracy_test = model.evaluate(inputs_test,targets_test,batch_size=batch_size,verbose=0)
    loss_train,accuracy_train = model.evaluate(inputs_train,targets_train,batch_size=batch_size,verbose=0)


    confusion_matrix = tf.math.confusion_matrix(labels=target_labels,predictions=predicted_labels)
    confusion_matrix_numpy = np.array(confusion_matrix)
    confusion_matrix_normalized = np.around(confusion_matrix_numpy.astype('float') / confusion_matrix_numpy.sum(axis=1)[:,np.newaxis],decimals=3)
    confusion_matrix_test = pd.DataFrame(confusion_matrix_normalized,index=classes,columns=classes)

    fig =  plt.figure(figsize=(6, 6))
    sns.heatmap(confusion_matrix_test, annot=True,cmap=plt.cm.Blues)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if title and title != "":
        fig.suptitle("%s \n Model Accuracy Percentage: Training = %.2f Test = %.2f \n Model Loss: Training = %.2f Test = %.2f \n\n Model Confusion Matrix" % (title,accuracy_train*100.0,accuracy_test*100.0,loss_train,loss_test), fontsize=16)
    else:
        print("%s: Model Accuracy Percentage: Training = %.2f Test = %.2f \n Model Loss: Training = %.2f Test = %.2f" % (title,accuracy_train*100.0,accuracy_test*100.0,loss_train,loss_test))
        fig.suptitle("Model Confusion Matrix", fontsize=16)

    plt.show()

class TestCallback(Callback):
    def __init__(self, inputs_test, targets_test,inputs_train, targets_train,outdata,validation_freq=10,batch_size=256):
        self.inputs_test = inputs_test
        self.targets_test = targets_test
        self.inputs_train = inputs_train
        self.targets_train = targets_train
        self.outdata = outdata
        self.validation_freq = validation_freq
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs={}):
        if epoch < self.validation_freq or epoch % self.validation_freq == 0:  

            loss_test,accuracy_test = self.model.evaluate(self.inputs_test,self.targets_test,batch_size=batch_size,verbose=0)
            loss_train,accuracy_train = self.model.evaluate(self.inputs_train,self.targets_train,batch_size=batch_size,verbose=0)

            print("Epoch(%d): Test Accuracy = %.3f, Training Accuracy = %.3f, Test Loss = %.3f, Training Loss = %.3f" % (epoch+1,accuracy_test*100.0,accuracy_train*100.0,loss_test,loss_train))
            self.outdata["test-accuracy"][epoch+1] = accuracy_test * 100.0
            self.outdata["train-accuracy"][epoch+1] = accuracy_train * 100.0
            self.outdata["test-loss"][epoch+1] = loss_test
            self.outdata["train-loss"][epoch+1] = loss_train
def plot_per_epoch_data(data,title):
    x1 = data["test-accuracy"].keys()
    y1 = data["test-accuracy"].values()

    x2 = data["train-accuracy"].keys()
    y2 = data["train-accuracy"].values()

    plt.plot(x1, y1, 'b', label='Test Accuracy')
    plt.plot(x2, y2, 'r', label='Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')
    plt.title(title + " :Accuracy")
    plt.legend(loc='lower right')
    plt.show()

    x1 = data["test-loss"].keys()
    y1 = data["test-loss"].values()

    x2 = data["train-loss"].keys()
    y2 = data["train-loss"].values()

    plt.plot(x1, y1, 'b', label='Test Loss')
    plt.plot(x2, y2, 'r', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.title(title + " :Loss")
    plt.legend(loc='upper center')
    plt.show()

if __name__ == '__main__':
    epochs = [100,100,100,100]
    batch_size = 256
    validation_freq = 1
    earlyStopPatience = 10
    steps_per_execution = 1
    fit_verbose = 0

    model_labels = ['Light Weight Dense Model','Low-Rank(2X Compression) Model','Low-Rank(4X Compression) Model','Low-Rank(8X Compression) Model']
    factors = [1,2,4,8]

    models = dict()
    (trainX,trainY), (testX,testY) = mnist.load_data()

    num_features = trainX.shape[1]*trainX.shape[2]
    num_classes = 10 

    trainX = trainX.reshape(trainX.shape[0], num_features)
    testX = testX.reshape(testX.shape[0], num_features)

    trainX = trainX.astype(float) / 255.0
    testX = testX.astype(float) / 255.0

    trainY = to_categorical(trainY, num_classes)
    testY = to_categorical(testY, num_classes)

    num_hidden_0 = 100
    num_hidden_1 = 50
    
    input_shape = (num_features,)

    models[model_labels[0]] = Sequential()
    models[model_labels[0]].add(Dense(num_hidden_0, input_shape=input_shape, activation='relu'))
    models[model_labels[0]].add(Dense(num_hidden_1, activation='relu'))
    models[model_labels[0]].add(Dense(num_classes, activation='softmax'))

    title = "Step1: %s" % model_labels[0]
    writeModelSummary(models[model_labels[0]],projdir.replace('\\', '/') + '/results/step1_dense_model_summary.txt',title)

    models[model_labels[0]].compile(optimizer=keras.optimizers.RMSprop(),loss=keras.losses.CategoricalCrossentropy(), metrics=[keras.metrics.CategoricalAccuracy()],steps_per_execution=steps_per_execution)

    data = dict()
    for label in model_labels:
        data[label] = dict()
        data[label]["test-accuracy"] = dict()
        data[label]["train-accuracy"] = dict()
        data[label]["test-loss"] = dict()
        data[label]["train-loss"] = dict()
    earlyStopCallBack = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=earlyStopPatience,restore_best_weights=True)
    testAccuracyCallBack = TestCallback(testX,testY,trainX,trainY,data[model_labels[0]],validation_freq,batch_size)
    print("\n=============Started: Training %s ==============" % model_labels[0])
    models[model_labels[0]].fit(trainX, trainY, epochs=epochs[0], batch_size=batch_size,  callbacks=[testAccuracyCallBack],verbose=fit_verbose,validation_data=(testX, testY),validation_freq=1)
    print("============Finished: Training %s ==============" % model_labels[0])

    print("\n========Started: Evaluating Trained %s =========" % model_labels[0])
    title = "Step1: %s" % model_labels[0]
    show_model_metrics(models[model_labels[0]],testX,testY,trainX,trainY,title,range(num_classes),batch_size)
    print("========Finished: Evaluating Trained %s ========" % model_labels[0])

    plot_per_epoch_data(data[model_labels[0]],model_labels[0])

    weights = dict()
    for label in model_labels:
        weights[label] = dict()
    i = 0
    for layer in models[model_labels[0]].layers:
        weights[model_labels[0]][i] = np.array(layer.weights[0])
        w = weights[model_labels[0]][i]

        
        u,s,v = np.linalg.svd(w)
        for idx in range(1,len(model_labels)):
            k = int(w.shape[1] / factors[idx])
            if i == len(models[model_labels[0]].layers) - 1:
                k = w.shape[1]
            weights[model_labels[idx]][2*i] = np.dot(u[:,:k], np.diag(s[:k]))
            weights[model_labels[idx]][(2*i) + 1]= v[:k,:]


        #print("i = %d" % i)
        #print("\t%s = %s" %(model_labels[0],weights[model_labels[0]][i].shape))
        #for idx in range(1,len(model_labels)):
        #    print("\t%s = [%s,%s]" %(model_labels[idx],weights[model_labels[idx]][2*i].shape,weights[model_labels[idx]][(2*i) + 1].shape))
   
        i += 1
    
    for idx in range(1,len(model_labels)):
        models[model_labels[idx]] = Sequential()
        models[model_labels[idx]].add(Dense(int(num_hidden_0/factors[idx]), input_shape=input_shape, activation='relu'))
        models[model_labels[idx]].add(Dense(num_hidden_0, activation='relu'))
        models[model_labels[idx]].add(Dense(int(num_hidden_1/factors[idx]), activation='relu'))
        models[model_labels[idx]].add(Dense(num_hidden_1, activation='relu'))
        models[model_labels[idx]].add(Dense(num_classes, activation='relu'))
        models[model_labels[idx]].add(Dense(num_classes, activation='softmax'))
        
        i = 0 
        for _ in models[model_labels[idx]].layers:
            iweights = weights[model_labels[idx]][i]
            ibias = np.reshape(np.array(tf.random.normal([iweights.shape[1]])),(iweights.shape[1],))

            models[model_labels[idx]].layers[i].set_weights([iweights,ibias])
            i += 1
    
        title = "Step2: %s" % model_labels[idx]
        writeModelSummary(models[model_labels[idx]],projdir.replace('\\', '/') + '/results/step2_' + str(factors[idx]) + 'X_low_rank_model_summary.txt',title)

        models[model_labels[idx]].compile(optimizer=keras.optimizers.RMSprop(),loss=keras.losses.CategoricalCrossentropy(), metrics=[keras.metrics.CategoricalAccuracy()],steps_per_execution=steps_per_execution)

        earlyStopCallBack = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=earlyStopPatience,restore_best_weights=True)
        testAccuracyCallBack = TestCallback(testX,testY,trainX,trainY,data[model_labels[idx]],validation_freq,batch_size)
        print("\n=============Started: Training %s ==============" % model_labels[idx])
        models[model_labels[idx]].fit(trainX, trainY, epochs=epochs[idx], batch_size=batch_size,  callbacks=[testAccuracyCallBack,earlyStopCallBack],verbose=fit_verbose,validation_data=(testX, testY),validation_freq=1)
        print("============Finished: Training %s ==============" % model_labels[idx])

        print("\n========Started: Evaluating Trained %s =========" % model_labels[idx])
        title = "Step2: %s" % model_labels[idx]
        show_model_metrics(models[model_labels[idx]],testX,testY,trainX,trainY,title,range(num_classes),batch_size)
        print("========Finished: Evaluating Trained %s ========" % model_labels[idx])

        plot_per_epoch_data(data[model_labels[idx]],model_labels[idx])