import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras import layers,models
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import Callback

import os
import sys
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

filedir = os.path.dirname(os.path.abspath(__file__))
filedir = os.path.dirname(filedir).replace('\\', '/').replace('C:', '')
sys.path.insert(0, filedir)

projdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def writeFiltersSummary(model,outpath,title):
    outfile = open(outpath, 'w')
    outfile.write("%s: 2D-Convolution  Layer Summary" %(title))
    for layer in model.layers:
        if 'conv2d' in layer.name:
            filters = np.array(layer.get_weights())[0]
            outfile.write("layerName = %s, filtersShape = %s\n" %(layer.name,filters.shape))
    outfile.close()

def visualizeFilters(model,title,layerName,rows = 3, cols = 6):
    layer = model.get_layer(name=layerName)
    filters = np.array(layer.get_weights())[0]
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    index = 1
    fig = plt.figure()
    fig.suptitle("%s 2D-Convolution Layer Filters: %s" %(title,layerName), fontsize=16)
    for r in range(rows):
        for c in range(cols):
            ax = plt.subplot(rows,cols,index)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title("filter(" + str(r) + "," + str(c) + ")")
            plt.imshow(filters[:,:,r,c], cmap='gray')
            index += 1

    plt.show()

def writeModelSummary(model,outpath,title):
    outfile = open(outpath, 'w')
    outfile.write("%s Model Summary" %(title))
    model.summary(print_fn=lambda x: outfile.write(x + '\n'))
    outfile.close()
    
def show_image(image,title=None):
    plt.figure(figsize=(6, 6))
    image_numpy = np.array(image,dtype=float)
    i_min, i_max = image_numpy.min(), image_numpy.max()
    image_norm = (image_numpy - i_min) / (i_max - i_min)
    plt.imshow(image_norm)
    if title and title != "":
        plt.title(title)
    plt.show()

def show_model_metrics(model,inputs,targets,title=None,batch_size=32):
    predicted = (model.predict(inputs) > 0.5).astype("int32")[:,0]   
    hits = len(np.where((predicted - targets) == 0)[0])
    accuracy = (hits/inputs.shape[0]) * 100

    #score=model.evaluate(inputs,targets,batch_size=batch_size)
    #accuracy = score[1] * 100.0

    confusion_matrix = tf.math.confusion_matrix(labels=targets,predictions=predicted)
    confusion_matrix_numpy = np.array(confusion_matrix)
    confusion_matrix_normalized = np.around(confusion_matrix_numpy.astype('float') / confusion_matrix_numpy.sum(axis=1)[:,np.newaxis],decimals=3)
    confusion_matrix_dataframe = pd.DataFrame(confusion_matrix_normalized,index=classes,columns=classes)
   
    fig =  plt.figure(figsize=(6, 6))
    sns.heatmap(confusion_matrix_dataframe, annot=True,cmap=plt.cm.Blues)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if title and title != "":
        fig.suptitle("%s \n Model Accuracy Percentage = %.2f \n\n Model Confusion Matrix" % (title,accuracy), fontsize=16)
    else:
        print("%s: Model Accuracy Percentage = %.2f " % (title,accuracy))
        fig.suptitle("Model Confusion Matrix", fontsize=16)

    plt.show()
    return accuracy

class TestCallback(Callback):
    def __init__(self, inputs, targets,outdata,validation_freq=10,batch_size=256):
        self.inputs = inputs
        self.targets = targets
        self.outdata = outdata
        self.validation_freq = validation_freq
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs={}):
        if epoch < self.validation_freq or epoch % self.validation_freq == 0:            
            predicted = (model.predict(self.inputs) > 0.5).astype("int32")[:,0]   
            hits = len(np.where((predicted - self.targets) == 0)[0])
            accuracy = (hits/self.inputs.shape[0]) * 100
            #loss, acc = self.model.evaluate(self.inputs, self.targets,batch_size=self.batch_size)
            #accuracy = acc * 100
            print("Epoch(%d): Test Accuracy = %.3f" % (epoch+1,accuracy))
            self.outdata[epoch+1] = accuracy

if __name__ == '__main__':
    epochs = 1000
    batch_size = 256
    validation_freq = 10
    earlyStopPatience = 3
    steps_per_execution = 1
    subNetworkFraction = 0.10
    fit_verbose = 1

    pre_model = InceptionResNetV2(weights="imagenet",include_top=False,input_shape=(150,150,3))
    title = "Step1 Pre-Trained Base CNN(InceptionResNetV2)"
    writeModelSummary(pre_model,projdir.replace('\\', '/') + '/results/step1_pre_model_summary.txt',title)
    writeFiltersSummary(pre_model,projdir.replace('\\', '/') + '/results/step1_pre_model_filters.txt',title)
    visualizeFilters(pre_model,title,"conv2d")
    visualizeFilters(pre_model,title,"conv2d_1")
    visualizeFilters(pre_model,title,"conv2d_7")

    print("\n==========================Started: Pre-Processing Training Dataset===========================")
    dirpath_train = projdir.replace('\\', '/') + '/data/dataset/training_set'
    dataset_train = image_dataset_from_directory(dirpath_train,color_mode="rgb",image_size=(150, 150),label_mode = "binary",batch_size=1,shuffle=False)
    inputs_train = np.concatenate([x for x, y in dataset_train], axis=0)
    i_min, i_max = inputs_train.min(), inputs_train.max()
    inputs_train = (inputs_train - i_min) / (i_max - i_min)
    targets_train = np.concatenate([y for x, y in dataset_train], axis=0).astype("int32")[:,0]
    print("==========================Finished: Pre-Processing Training Dataset==========================")

    print("\n============================Started: Pre-Processing Test Dataset=============================")
    dirpath_test = projdir.replace('\\', '/') + '/data/dataset/test_set'
    dataset_test = image_dataset_from_directory(dirpath_test,color_mode="rgb",image_size=(150, 150),label_mode = "binary",batch_size=1,shuffle=False)
    inputs_test = np.concatenate([x for x, y in dataset_test], axis=0)
    i_min, i_max = inputs_test.min(), inputs_test.max()
    inputs_test = (inputs_test - i_min) / (i_max - i_min)
    targets_test = np.concatenate([y for x, y in dataset_test], axis=0).astype("int32")[:,0]
    print("============================Finished: Pre-Processing Test Dataset============================")

    classes = dataset_test.class_names

    model = models.Sequential()
    model.add(pre_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(256,activation='relu'))
    #model.add(layers.Dense(64,activation='relu'))
    #model.add(layers.Dense(16,activation='relu'))
    model.add(layers.Dense(1,activation='sigmoid'))

    pre_model.trainable=False

    title = "Step3 Transfer Model CNN(Base:InceptionResNetV2 + Custom-Classifier)"
    writeModelSummary(model,projdir.replace('\\', '/') + '/results/step3_transfer_model_summary.txt',title)

    model.compile(optimizer=keras.optimizers.RMSprop(),loss=keras.losses.BinaryCrossentropy(),metrics=[keras.metrics.BinaryAccuracy()],steps_per_execution=steps_per_execution)

    print("\n======Started: Evaluating Pre-Trained Transfer Model CNN(Base:InceptionResNetV2 + Custom-Classifier)======")
    title = "Step4(i) Pre-Trained Transfer Model CNN(Base:InceptionResNetV2 + Custom-Classifier)"
    pretrain_accuracy = show_model_metrics(model,inputs_test,targets_test,title,batch_size)
    print("======Finished: Evaluating Pre-Trained Transfer Model CNN(Base:InceptionResNetV2 + Custom-Classifier)======")
    
    accuracy_full = dict()
    accuracy_full[0] = pretrain_accuracy
    earlyStopCallBack = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=earlyStopPatience,restore_best_weights=True)
    testAccuracyCallBack = TestCallback(inputs_test,targets_test,accuracy_full,validation_freq,batch_size)
    print("\n=============Started: Training Transfer Model CNN(Base:InceptionResNetV2 + Custom-Classifier)==============")
    model.fit(inputs_train, targets_train, epochs = epochs,batch_size=batch_size, callbacks=[testAccuracyCallBack,earlyStopCallBack],verbose=fit_verbose)
    print("============Finished: Training Transfer Model CNN(Base:InceptionResNetV2 + Custom-Classifier)==============")

    print("\n========Started: Evaluating Trained Transfer Model CNN(Base:InceptionResNetV2 + Custom-Classifier)=========")
    title = "Step4(ii) Trained Transfer Model CNN(Base:InceptionResNetV2 + Custom-Classifier)"
    show_model_metrics(model,inputs_test,targets_test,title,batch_size)
    print("========Finished: Evaluating Trained Transfer Model CNN(Base:InceptionResNetV2 + Custom-Classifier)========")
    

    x1 = accuracy_full.keys()
    y1 = accuracy_full.values()

    plt.plot(x1, y1, 'b', label='Transfer Model CNN(Base:InceptionResNetV2 + Custom-Classifier)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')
    plt.title('Model Prediction Accuracy on Test Dataset(Cat/Dog)')
    plt.legend(loc='lower right')
    plt.show()

    k = int(len(pre_model.layers)*subNetworkFraction)
    model = models.Sequential()
    model.add(pre_model)
    model.add(layers.Flatten())
    i = 0
    for layer in model.layers:
        if i >= k:
            model.pop(layer)
            print("layer(%d) = %s" %(i,layer.name))
        i += 1
    model.add(layers.Dense(256,activation='relu'))
    #model.add(layers.Dense(64,activation='relu'))
    #model.add(layers.Dense(16,activation='relu'))
    model.add(layers.Dense(1,activation='sigmoid'))

    pre_model.trainable=False

    model.compile(optimizer=keras.optimizers.RMSprop(),loss=keras.losses.BinaryCrossentropy(),metrics=[keras.metrics.BinaryAccuracy()],steps_per_execution=steps_per_execution)

    print("\n======Started: Evaluating Pre-Trained Transfer(subnetwork) Model CNN(Base:InceptionResNetV2[0:%d] + Custom-Classifier)======" % (k-1))
    title = "Step4(iii) Pre-Trained Transfer(subnetwork) Model CNN(Base:InceptionResNetV2[0:" + str(k) +"] + Custom-Classifier)"
    pretrain_accuracy = show_model_metrics(model,inputs_test,targets_test,title,batch_size)
    print("======Finished: Evaluating Pre-Trained Transfer(subnetwork) Model CNN(Base:InceptionResNetV2[0:%d] + Custom-Classifier)======" % (k-1))
    
    accuracy_sub = dict()
    accuracy_sub[0] = pretrain_accuracy
    earlyStopCallBack = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=earlyStopPatience,restore_best_weights=True)
    testAccuracyCallBack = TestCallback(inputs_test,targets_test,accuracy_sub,validation_freq,batch_size)
    print("\n=============Started: Training Transfer(subnetwork) Model CNN(Base:InceptionResNetV2[0:%d] + Custom-Classifier)==============" % (k-1))
    model.fit(inputs_train, targets_train, epochs = epochs,batch_size=batch_size, callbacks=[testAccuracyCallBack,earlyStopCallBack],verbose=fit_verbose)
    print("============Finished: Training Transfer(subnetwork) Model CNN(Base:InceptionResNetV2[0:%d] + Custom-Classifier)=============="  % (k-1))

    print("\n========Started: Evaluating Trained Transfer(subnetwork) Model CNN(Base:InceptionResNetV2[0:%d] + Custom-Classifier)========="  % (k-1))
    title = "Step4(iii) Trained Transfer(subnetwork) Model CNN(Base:InceptionResNetV2 + Custom-Classifier)"
    show_model_metrics(model,inputs_test,targets_test,title,batch_size)
    print("========Finished: Evaluating Trained Transfer(subnetwork) Model CNN(Base:InceptionResNetV2[0:%d] + Custom-Classifier)========" % (k-1))
    

    x1 = accuracy_full.keys()
    y1 = accuracy_full.values()

    x2 = accuracy_sub.keys()
    y2 = accuracy_sub.values()    

    plt.plot(x1, y1, 'b', label='Transfer Model CNN(Base:InceptionResNetV2 + Custom-Classifier)')
    plt.plot(x2, y2, 'r', label='Transfer(subnetwork) Model CNN(Base:InceptionResNetV2[0:' + str(k) +'] + Custom-Classifier)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')
    plt.title('Model Prediction Accuracy on Test Dataset(Cat/Dog)')
    plt.legend(loc='lower right')
    plt.show()
