import tensorflow as tf 
from tensorflow import keras
import numpy as np # linear algebra
from tensorflow.keras.applications import ResNet50, DenseNet121, Xception,InceptionResNetV2,VGG16
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras import optimizers,losses,metrics
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, Callback
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
import pandas as pd
import seaborn as sns

import os
import sys
import math
import matplotlib.pyplot as plt

filedir = os.path.dirname(os.path.abspath(__file__))
filedir = os.path.dirname(filedir).replace('\\', '/').replace('C:', '')
sys.path.insert(0, filedir)

from covid19detection.common import *

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

def show_model_metrics(model,inputs_test, targets_test,inputs_train, targets_train,title=None,classes=['Negative','Typical','Indeterminate','Atypical'],batch_size=256,threshold_wh=1e-6,generator_train=None,generator_test=None):
    loss_function = tf.keras.losses.MeanSquaredError()  
    if inputs_test is not None:
        predicted = model.predict(inputs_test)
        actual = targets_test
        #loss_test_eval,accuracy_test_eval = model.evaluate(inputs_test,targets_test,batch_size=batch_size,verbose=0)
    else:
        predicted = model.predict(generator_test)
        actual = np.transpose(np.array(generator_train.labels))
        #loss_test_eval,accuracy_test_eval = model.evaluate(self.generator_test,batch_size=batch_size,verbose=0)

    predicted_study = np.argmax(predicted[:,:4],axis=1)
    actual_study = np.argmax(actual[:,:4],axis=1)

    hits_study = len(np.where((actual_study - predicted_study) == 0)[0])
    accuracy_test_study = (hits_study/actual.shape[0])
    #loss_test_study = loss_function(actual_study,predicted_study).numpy() / 4

    confusion_matrix = tf.math.confusion_matrix(labels=actual_study,predictions=predicted_study)
    confusion_matrix_numpy = np.array(confusion_matrix)
    confusion_matrix_normalized = np.around(confusion_matrix_numpy.astype('float') / confusion_matrix_numpy.sum(axis=1)[:,np.newaxis],decimals=3)
    confusion_matrix_test = pd.DataFrame(confusion_matrix_normalized,index=classes,columns=classes)

    if inputs_train is not None:
        predicted = model.predict(inputs_train)
        actual = targets_train
        #loss_train_eval,accuracy_train_eval = model.evaluate(inputs_train,targets_train,batch_size=batch_size,verbose=0)
    else:
        predicted = model.predict(generator_train)
        actual = np.transpose(np.array(generator_train.labels))
        #loss_train_eval,accuracy_train_eval = model.evaluate(self.generator_train,batch_size=batch_size,verbose=0)

    predicted_study = np.argmax(predicted[:,:4],axis=1)
    actual_study = np.argmax(actual[:,:4],axis=1)

    hits_study = len(np.where((actual_study - predicted_study) == 0)[0])
    accuracy_train_study = (hits_study/actual.shape[0])
    #loss_train_study = loss_function(actual_study,predicted_study).numpy() / 4

    fig =  plt.figure(figsize=(6, 6))
    sns.heatmap(confusion_matrix_test, annot=True,cmap=plt.cm.Blues)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if title and title != "":
        fig.suptitle("%s \n Model Accuracy Percentage: Test = %.2f Training = %.2f \n\n Model Confusion Matrix" % (title,accuracy_test_study*100.0,accuracy_train_study*100.0), fontsize=16)
    else:
        print("%s: Model Accuracy Percentage: Test = %.2f Training = %.2f \n\n Model Confusion Matrix" % (title,accuracy_test_study*100.0,accuracy_train_study*100.0))
        fig.suptitle("Model Confusion Matrix", fontsize=16)

    plt.show()

def plot_per_epoch_data(data,title,tag=["study","eval"], plot_loss=False):
    x1 = data["test-accuracy:" + tag[0]].keys()
    y1 = data["test-accuracy:" + tag[0]].values()

    x2 = data["train-accuracy:" + tag[0]].keys()
    y2 = data["train-accuracy:" + tag[0]].values()

    plt.plot(x1, y1, 'b', label='Test Accuracy')
    plt.plot(x2, y2, 'r', label='Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')
    plt.title(title + " :Accuracy")
    plt.legend(loc='lower right')
    plt.show()

    if plot_loss:
        x1 = data["test-loss:" + tag[1]].keys()
        y1 = data["test-loss:" + tag[1]].values()

        x2 = data["train-loss:" + tag[1]].keys()
        y2 = data["train-loss:" + tag[1]].values()

        plt.plot(x1, y1, 'b', label='Test Loss')
        plt.plot(x2, y2, 'r', label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('loss')
        plt.title(title + " :Loss")
        plt.legend(loc='upper center')
        plt.show()


class TestCallback(Callback):
    def __init__(self,inputs_test, targets_test,inputs_train, targets_train,outdata,validation_freq=10,batch_size=256,threshold_wh=1e-6,generator_train=None,generator_test=None):
        self.inputs_test = inputs_test
        self.targets_test = targets_test
        self.inputs_train = inputs_train
        self.targets_train = targets_train
        self.threshold_wh = threshold_wh
        self.outdata = outdata
        self.validation_freq = validation_freq
        self.batch_size = batch_size
        self.generator_train = generator_train
        self.generator_test = generator_test

    def on_epoch_end(self, epoch, logs={}):
        if epoch < self.validation_freq or epoch % self.validation_freq == 0:
            loss_function = tf.keras.losses.MeanSquaredError()  
            if self.inputs_train is not None:
                predicted = self.model.predict(self.inputs_train)
                actual = np.array(self.targets_train)
                #loss_train_eval,accuracy_train_eval = self.model.evaluate(self.inputs_train,self.targets_train,verbose=0)

            else:
                predicted = self.model.predict(self.generator_train)
                actual = np.transpose(np.array(self.generator_train))
                #loss_train_eval,accuracy_train_eval = self.model.evaluate(self.generator_train,verbose=0)

            predicted_study = np.argmax(predicted[:,:4],axis=1)
            actual_study = np.argmax(actual[:,:4],axis=1)
            hits_study = len(np.where((actual_study - predicted_study) == 0)[0])
            accuracy_train_study = (hits_study/actual.shape[0])
            #loss_train_study = loss_function(actual_study,predicted_study).numpy()

            if self.inputs_test is not None:
                predicted = self.model.predict(self.inputs_test)
                actual = self.targets_test
                #loss_test_eval,accuracy_test_eval = self.model.evaluate(self.inputs_test,self.targets_test,batch_size=batch_size,verbose=0)

            else:
                predicted = self.model.predict(self.generator_test)
                actual = np.transpose(np.array(self.generator_test))
                #loss_test_eval,accuracy_test_eval = self.model.evaluate(self.generator_test,batch_size=batch_size,verbose=0)

            predicted_study = np.argmax(predicted[:,:4],axis=1)
            actual_study = np.argmax(actual[:,:4],axis=1)
            hits_study = len(np.where((predicted_study - actual_study) == 0)[0])
            accuracy_test_study = (hits_study/actual.shape[0])
            #loss_test_study = loss_function(actual_study,predicted_study).numpy()

            print("================================== Epoch(%d) Custom Metrics: Start ==================================" % (epoch+1))
            print("\t\tTest Accuracy(study) = %.3f, Training Accuracy(study) = %.3f" % (accuracy_test_study*100.0,accuracy_train_study*100.0))
            #print("\t\tTest Loss(study) = %.3f, Training Loss(study) = %.3f" % (loss_test_study,loss_train_study))
            #print("\t\tTest Accuracy(eval) = %.3f, Training Accuracy(eval) = %.3f" % (accuracy_test_eval*100.0,accuracy_train_eval*100.0))
            #print("\t\tTest Loss(eval) = %.3f, Training Loss(eval) = %.3f" % (loss_test_eval,loss_train_eval))

            self.outdata["test-accuracy:study"][epoch+1] = accuracy_test_study * 100.0
            self.outdata["train-accuracy:study"][epoch+1] = accuracy_train_study * 100.0
            #self.outdata["test-loss:study"][epoch+1] = loss_test_study
            #self.outdata["train-loss:study"][epoch+1] = loss_train_study

            #self.outdata["test-accuracy:eval"][epoch+1] = accuracy_test_eval * 100.0
            #self.outdata["train-accuracy:eval"][epoch+1] = accuracy_train_eval * 100.0
            #self.outdata["test-loss:eval"][epoch+1] = loss_test_eval
            #self.outdata["train-loss:eval"][epoch+1] = loss_train_eval
            print("=================================== Epoch(%d) Custom Metrics: End ===================================" % (epoch+1))

if __name__ == '__main__':
    study_csv_path_training = inputdir+'/train_study_level.csv'
    image_csv_path_training = inputdir+'/train_image_level.csv'
    size_csv_path_training = imagedir+'/size.csv'
    study_columns = ['id','Negative','Typical','Indeterminate','Atypical']
    
    padding_value_xy = 0.0
    padding_value_wh = 0.0
    
    epochs = 15
    validation_split = 0.2
    steps_per_execution = 1
    validation_freq = 1
    fit_verbose = 1

    threshold_wh=1e-6

    include_box_outputs = True
    num_hidden = [128,64,32]   
    
    img_x = 64
    img_y = 64
    batch_size = 16
    image_format = 'jpg'

    class_labels = ['Negative','Typical','Indeterminate','Atypical']
    pre_model_fn = VGG16
    pre_model_name = 'VGG16_with_study_labels_only'

    #model_labels = [pre_model_name + ' transfer model with hidden layers = 1',pre_model_name + ' transfer model with hidden layers = 2',pre_model_name + ' transfer model with hidden layers = 3']
    model_labels = [pre_model_name + ' transfer model with hidden layers = 1']

    print("\n=============Started: Reading CSV files ==============")
    df_train = read_dataset_csv(study_csv_path_training,image_csv_path_training,size_csv_path_training,study_columns,padding_value_xy,padding_value_wh,image_format)
    print("============Finished: Reading CSV files ==============")

    max_boxes = int(df_train['ImageBoxCount'].max())

    y_col = study_columns[1:]
    loss_weights = np.ones(len(y_col))

    if include_box_outputs:
        for i in range(max_boxes):
            y_col.append('X'+str(i+1))
            y_col.append('Y'+str(i+1))
            y_col.append('W'+str(i+1))
            y_col.append('H'+str(i+1))

        for i in range(max_boxes):
            idx = 4*(i+1)
            if i > 0:
                loss_weights[idx:idx+4] = 1/(i+1)
            else:
                loss_weights[idx:idx+4] = 1

    num_outputs = len(y_col)

    base_model = pre_model_fn(weights="imagenet",include_top=False,input_shape=(img_x,img_y,3))
    base_model.trainable=False

    title = "Pre-Trained Base CNN(%s)" % pre_model_name
    writeModelSummary(base_model,projdir.replace('\\', '/') + '/results/' + pre_model_name + '_pre_model_summary.txt',title)

    models = dict()
    for idx in range(len(model_labels)):
        print("\n=============Started: Building %s ==============" % model_labels[idx])
        pre_model = pre_model_fn(weights="imagenet",include_top=False,input_shape=(img_x,img_y,3))
        pre_model.trainable=False
        models[model_labels[idx]] = Sequential()
        models[model_labels[idx]].add(pre_model)
        models[model_labels[idx]].add(Flatten())
        models[model_labels[idx]].add(Dense(16,activation='relu'))
        for h in range(idx+1):
            models[model_labels[idx]].add(Dense(num_hidden[h],activation='relu'))
        models[model_labels[idx]].add(Dense(num_outputs))

        title = pre_model_name + "_with_" + str(idx+1) + "_hidden_layers"
        writeModelSummary(models[model_labels[idx]],projdir.replace('\\', '/') + '/results/' + title + '_summary.txt',title)
        models[model_labels[idx]].compile(optimizer=optimizers.RMSprop(),loss=losses.MeanSquaredError(),loss_weights=loss_weights)
        #models[model_labels[idx]].compile(optimizer=optimizers.RMSprop(),loss=losses.CategoricalCrossentropy(), metrics=[metrics.CategoricalAccuracy()],steps_per_execution=steps_per_execution)
        print("============Finished: Building %s ==============" % model_labels[idx])

    print("\n=============Started: Reading training images ==============")
    generator_train = get_image_generator(df_train,imagedir + '/train','fname',y_col,img_x,img_y,batch_size,validation_split,'training')
    trainX,trainY = iterator_to_numpy(df_train,imagedir + '/train','fname',y_col,img_x,img_y,batch_size,validation_split,'training',generator_train)
    print("============Finished: Reading training images ==============")

    print("\n=============Started: Reading test images ==============")
    generator_test = get_image_generator(df_train,imagedir + '/train','fname',y_col,img_x,img_y,batch_size,validation_split,'validation')
    testX,testY = iterator_to_numpy(df_train,imagedir + '/train','fname',y_col,img_x,img_y,batch_size,validation_split,'validation',generator_test)
    print("============Finished: Reading test images ==============")

    data = dict()
    for label in model_labels:
        data[label] = dict()
        data[label]["test-accuracy:study"] = dict()
        data[label]["train-accuracy:study"] = dict()
        data[label]["test-loss:study"] = dict()
        data[label]["train-loss:study"] = dict()

        data[label]["test-accuracy:opacity"] = dict()
        data[label]["train-accuracy:opacity"] = dict()
        data[label]["test-loss:opacity"] = dict()
        data[label]["train-loss:opacity"] = dict()
    
        data[label]["test-accuracy:bbox"] = dict()
        data[label]["train-accuracy:bbox"] = dict()
        data[label]["test-loss:bbox"] = dict()
        data[label]["train-loss:bbox"] = dict()

        data[label]["test-accuracy:eval"] = dict()
        data[label]["train-accuracy:eval"] = dict()
        data[label]["test-loss:eval"] = dict()
        data[label]["train-loss:eval"] = dict()

    for idx in range(len(model_labels)):
        rlr = ReduceLROnPlateau(monitor = 'loss', factor = 0.1, patience = 2, verbose = 1,min_delta = 1e-4, min_lr = 1e-6, mode = 'min')
        es = EarlyStopping(monitor = 'val_loss', min_delta = 1e-4, patience = 5, mode = 'min',restore_best_weights = True, verbose = 1)
        ckp = ModelCheckpoint('model.h5',monitor = 'val_loss',verbose = 0, save_best_only = True, mode = 'min')
        metricsCallBack = TestCallback(testX,testY,trainX,trainY,data[model_labels[idx]],validation_freq,batch_size,threshold_wh)
        #metricsCallBack = TestCallback(None,None,None,None,data[model_labels[idx]],validation_freq,batch_size,threshold_wh,generator_train,generator_test)

        print("\n=============Started: Training %s ==============" % model_labels[idx])
        models[model_labels[idx]].fit(trainX, trainY, epochs=epochs,batch_size=batch_size,callbacks=[rlr,metricsCallBack],verbose=fit_verbose,shuffle=True)
        #models[model_labels[idx]].fit(generator_train, epochs=epochs,batch_size=batch_size,callbacks=[rlr,metricsCallBack],verbose=fit_verbose,shuffle=True)
        print("============Finished: Training %s ==============" % model_labels[idx])

        print("\n========Started: Evaluating Trained %s =========" % model_labels[idx])
        title = model_labels[idx]
        show_model_metrics(models[model_labels[idx]],testX,testY,trainX,trainY,title,study_columns[1:],batch_size)
        plot_per_epoch_data(data[model_labels[idx]],model_labels[idx])
        print("========Finished: Evaluating Trained %s ========" % model_labels[idx])



    
