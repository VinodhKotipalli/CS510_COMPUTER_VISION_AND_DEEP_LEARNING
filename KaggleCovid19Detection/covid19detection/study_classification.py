import tensorflow as tf 
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
    study_csv_path_training = inputdir+'/train_study_level.csv'
    image_csv_path_training = inputdir+'/train_image_level.csv'
    size_csv_path_training = imagedir+'/size.csv'
    study_columns = ['id','Negative','Typical','Indeterminate','Atypical']
    
    padding_value_xy = 0.0
    padding_value_wh = 0.0
    
    epochs = 2
    validation_split = 0.2
    steps_per_execution = 1
    num_hidden_0 = 256
    num_hidden_1 = 64
    num_hidden_2 = 16    
    
    
    img_x = 224
    img_y = 224
    batch_size = 16
    image_format = 'jpg'
    
    df_train = read_dataset_csv(study_csv_path_training,image_csv_path_training,size_csv_path_training,study_columns,padding_value_xy,padding_value_wh,image_format)
    
    max_boxes = int(df_train['ImageBoxCount'].max())

    y_col = study_columns[1:]
    for i in range(max_boxes):
        y_col.append('X'+str(i+1))
        y_col.append('Y'+str(i+1))
        y_col.append('W'+str(i+1))
        y_col.append('H'+str(i+1))

    generator_train = get_image_generator(df_train,imagedir + '/train','fname',y_col,img_x,img_y,batch_size,validation_split,'training')
    generator_val = get_image_generator(df_train,imagedir + '/train','fname',y_col,img_x,img_y,batch_size,validation_split,'validation')

    


    num_outputs = len(y_col)
    
    pre_model = VGG16(weights='imagenet',include_top = False,input_shape=(img_x,img_y, 3))
    pre_model.trainable=False
    
    model = models.Sequential()
    model.add(pre_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(num_hidden_0,activation='relu'))
    #model.add(layers.Dense(num_hidden_1,activation='relu'))
    #model.add(layers.Dense(num_hidden_2,activation='relu'))
    model.add(layers.Dense(num_outputs,activation='relu'))

    model.compile(optimizer=optimizers.RMSprop(),loss=losses.MeanSquaredError(),metrics=[metrics.MeanSquaredError(),metrics.RootMeanSquaredError()],steps_per_execution=steps_per_execution)

    rlr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 2, verbose = 1,min_delta = 1e-4, min_lr = 1e-6, mode = 'min')
    es = EarlyStopping(monitor = 'val_loss', min_delta = 1e-4, patience = 5, mode = 'min',restore_best_weights = True, verbose = 1)
    ckp = ModelCheckpoint('model.h5',monitor = 'val_loss',verbose = 0, save_best_only = True, mode = 'min')

    data = dict()
    data["test-accuracy"] = dict()
    data["train-accuracy"] = dict()
    data["test-loss"] = dict()
    data["train-loss"] = dict()

    print(df_train.shape)
    print(df_train.columns)

    #history = model.fit(generator_train, epochs=epochs,validation_data=generator_val,callbacks=[es, rlr, ckp],verbose=1)
    predictions = model.predict(generator_val)
    print(predictions.shape)
    print(predictions[:4,:5])
    print(np.argmax(predictions[:4,:5],axis=0))
    print(predictions[:5,4:8])

    actual = np.array(generator_val.labels)
    print(actual.shape)
    print(y_col)
    print(actual[:4,:5])
    print(np.argmax(actual[:4,:5],axis=0))
    print(actual[4:8,:5])

    filenames = generator_val.filenames
    print(len(filenames))
    print(filenames[:5])


    
