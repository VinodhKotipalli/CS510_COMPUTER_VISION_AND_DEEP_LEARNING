import tensorflow as tf 
import numpy as np # linear algebra
from tensorflow.keras.applications import ResNet50, DenseNet121, Xception,InceptionResNetV2,VGG16
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D
from tensorflow.keras import optimizers,losses,metrics
from tensorflow.keras import models,layers
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, Callback
import tensorflow.keras.backend as K
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
            if self.inputs_train is not None:
                predicted = model.predict(self.inputs_train)
                actual = self.targets_train
            else:
                 predicted = model.predict(self.generator_train)
                 actual = np.transpose(np.array(self.generator_train))
            print("=======Training Data========")
            print(predicted.shape)
            print(actual.shape)
            predicted_study = np.argmax(predicted[:,:4],axis=1)
            actual_study = np.argmax(actual[:,:4],axis=1)
            hits_study = len(np.where((predicted_study - actual_study) == 0)[0])
            accuracy_train_study = (hits_study/actual.shape[0])

            if self.inputs_test is not None:
                predicted = model.predict(self.inputs_test)
                actual = self.targets_test
            else:
                 predicted = model.predict(self.generator_test)
                 actual = np.transpose(np.array(self.generator_test))
            print("=======Test Data========")
            print(predicted.shape)
            print(actual.shape)
            predicted_study = np.argmax(predicted[:,:4],axis=1)
            actual_study = np.argmax(actual[:,:4],axis=1)
            hits_study = len(np.where((predicted_study - actual_study) == 0)[0])
            accuracy_test_study = (hits_study/actual.shape[0])

            print("Epoch(%d): Test Accuracy = %.3f, Training Accuracy = %.3f" % (epoch+1,accuracy_test_study*100.0,accuracy_train_study*100.0))

if __name__ == '__main__':
    study_csv_path_training = inputdir+'/train_study_level.csv'
    image_csv_path_training = inputdir+'/train_image_level.csv'
    size_csv_path_training = imagedir+'/size.csv'
    study_columns = ['id','Negative','Typical','Indeterminate','Atypical']
    
    padding_value_xy = 0.0
    padding_value_wh = 0.0
    
    epochs = 10
    validation_split = 0.2
    steps_per_execution = 1
    num_hidden_0 = 256
    num_hidden_1 = 64
    num_hidden_2 = 16    
    validation_freq = 1
    fit_verbose = 1

    threshold_wh=1e-6
    
    img_x = 256
    img_y = 256
    batch_size = 16
    image_format = 'jpg'

    model_labels = ['CNN with dense layers = 1','CNN with dense layers = 2','CNN with dense layers = 3']
    titles = ['CNN_with_1_dense_layer','CNN_with_3_dense_layer','CNN_with_3_dense_layer']
    print("\n=============Started: Reading CSV files ==============")
    df_train = read_dataset_csv(study_csv_path_training,image_csv_path_training,size_csv_path_training,study_columns,padding_value_xy,padding_value_wh,image_format)
    print("============Finished: Reading CSV files ==============")

    max_boxes = int(df_train['ImageBoxCount'].max())

    y_col = study_columns[1:]
    #for i in range(max_boxes):
    #    y_col.append('X'+str(i+1))
    #    y_col.append('Y'+str(i+1))
    #    y_col.append('W'+str(i+1))
    #    y_col.append('H'+str(i+1))
    
    loss_weights = np.ones(len(y_col))
    for i in range(max_boxes):
        idx = 4*(i+1)
        loss_weights[idx:idx+4] = 1/(i+1)

    print("\n=============Started: Reading training images ==============")
    generator_train = get_image_generator(df_train,imagedir + '/train','fname',y_col,img_x,img_y,batch_size,validation_split,'training')
    trainX,trainY = iterator_to_numpy(df_train,imagedir + '/train','fname',y_col,img_x,img_y,batch_size,validation_split,'training')
    print("============Finished: Reading training images ==============")

    print("\n=============Started: Reading test images ==============")
    generator_test = get_image_generator(df_train,imagedir + '/train','fname',y_col,img_x,img_y,batch_size,validation_split,'validation')
    testX,testY = iterator_to_numpy(df_train,imagedir + '/train','fname',y_col,img_x,img_y,batch_size,validation_split,'validation')
    print("============Finished: Reading test images ==============")



    num_outputs = len(y_col)

    model = models.Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu',padding='same',input_shape=(img_x, img_y, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
    model.add(Flatten())
    model.add(Dense(num_hidden_0,activation='relu'))
    model.add(Dense(num_hidden_1,activation='relu'))
    model.add(Dense(num_hidden_2,activation='relu'))
    model.add(Dense(num_outputs,activation='softmax'))

    title = titles[0]
    writeModelSummary(model,projdir.replace('\\', '/') + '/results/' + title + '_summary.txt',title)

    #model.compile(optimizer=optimizers.RMSprop(),loss=losses.MeanSquaredError(),steps_per_execution=steps_per_execution,loss_weights=loss_weights)
    model.compile(optimizer=optimizers.RMSprop(),loss=losses.CategoricalCrossentropy(), metrics=[metrics.CategoricalAccuracy()],steps_per_execution=steps_per_execution)

    data = dict()
    data["test-accuracy"] = dict()
    data["train-accuracy"] = dict()
    data["test-loss"] = dict()
    data["train-loss"] = dict()

    rlr = ReduceLROnPlateau(monitor = 'loss', factor = 0.1, patience = 2, verbose = 1,min_delta = 1e-4, min_lr = 1e-6, mode = 'min')
    es = EarlyStopping(monitor = 'val_loss', min_delta = 1e-4, patience = 5, mode = 'min',restore_best_weights = True, verbose = 1)
    ckp = ModelCheckpoint('model.h5',monitor = 'val_loss',verbose = 0, save_best_only = True, mode = 'min')
    metricsCallBack = TestCallback(testX,testY,trainX,trainY,data,validation_freq,batch_size,threshold_wh)
    #metricsCallBack = TestCallback(None,None,None,None,data,validation_freq,batch_size,threshold_wh,generator_train,generator_test)


    print("\n=============Started: Training %s ==============" % model_labels[0])
    #history = model.fit(trainX, trainY, epochs=epochs,batch_size=batch_size,validation_data=(testX, testY),callbacks=[rlr,metricsCallBack],verbose=fit_verbose,validation_freq=validation_freq,shuffle=True)
    history = model.fit(generator_train, epochs=epochs,batch_size=batch_size,validation_data=generator_test,callbacks=[metricsCallBack],verbose=fit_verbose,validation_freq=validation_freq,shuffle=True)
    print("============Finished: Training %s ==============" % model_labels[0])




    
