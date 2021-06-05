import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras import backend as K
from tensorflow.keras import models
import numpy as np

import os
import sys
import copy
import matplotlib.pyplot as plt

PLUSINF = sys.float_info.max
MINUSINF = sys.float_info.min

filedir = os.path.dirname(os.path.abspath(__file__))
filedir = os.path.dirname(filedir).replace('\\', '/').replace('C:', '')
sys.path.insert(0, filedir)

projdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#tf.compat.v1.disable_eager_execution()

def show_image(image1,title=None,image2=None,alpha=0.5):
    plt.figure(figsize=(6, 6))
    image1_numpy = np.nan_to_num(np.array(image1,dtype=float))
    i_min, i_max = image1_numpy.min(), image1_numpy.max()
    if i_max - i_min > 0.0:
        image1_norm = np.nan_to_num((image1_numpy - i_min) / (i_max - i_min))
    else:
        image1_norm = image1_numpy
    plt.imshow(image1_norm)
    if image2 is None:
        image_numpy = image1_norm
    else:
        image2_numpy = np.nan_to_num(np.array(image2,dtype=float))
        i_min, i_max = image2_numpy.min(), image2_numpy.max()
        if i_max - i_min > 0.0:
            image2_norm = np.nan_to_num((image2_numpy - i_min) / (i_max - i_min))
        else:
            image2_norm = image2_numpy
        plt.imshow(image2_norm, alpha=alpha)
    if title and title != "":
        plt.title(title)
    plt.show()

def get_heatmap(importance_weight,feature_maps):
    rows,cols,filters = feature_maps.shape
    result = np.zeros(shape=(rows,cols))
    print(feature_maps)
    for i in range(rows):
        for j in range(cols):
            for k in range(filters):
                result[i,j] = importance_weight[k] * feature_maps[i,j,k]
    return result

def get_topindex(inputs,skip_count=0):
    inputs_copy = copy.deepcopy(np.array(inputs))
    result = -1
    for i in range(skip_count+1):
        if result > 0:
            inputs_copy[result] = MINUSINF
        result = np.argmax(inputs_copy)
    return result

if __name__ == '__main__':
    dataset_size = 5
    max_classes = 3
    alpha = 0.5
    model = VGG16(weights='imagenet')
    last_layer=model.get_layer('block5_conv3')
    heatmap_model = models.Model([model.inputs],[last_layer.output,model.output])
    for i in range(1,dataset_size+1):
        fname = "gc%d.jpg" %i
        iname = "gc%d"%i
        ipath = projdir.replace('\\', '/') + '/data/' + fname

        print("==================Image = %s==================" % iname)
        src_image = image.load_img(ipath)
        title = "%s:orginal" % iname
        show_image(src_image,title)
        src_image_shape = src_image.size

        rsz_image = image.load_img(ipath,target_size=(224,224))
        title = "%s:resized" % iname
        #show_image(rsz_image,title)
        rsz_image_shape = rsz_image.size
        
        x = image.img_to_array(rsz_image)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        prediction = model.predict(x)
        top_predictions = decode_predictions(prediction,top=max_classes)[0]
        #print(top_predictions)
        top_classes = []
        for top_prediction in top_predictions:
            top_classes.append(top_prediction[1])
        print("Top-%d class predictions for image(%s) are :  %s" %(max_classes,iname,top_classes))
        for c in range(max_classes):
            cname = top_classes[c]
            with tf.GradientTape() as gtape:
                conv_output,predictions = heatmap_model(x)
                loss = predictions[:,get_topindex(predictions[0],c)]
                gradient = gtape.gradient(loss,conv_output)
                importance_weight = tf.reduce_mean(gradient, axis=(0, 1, 2))
            heatmap = tf.reduce_sum(tf.multiply(importance_weight,conv_output),axis=-1)[0]
            heatmap=np.maximum(heatmap,0.0)
            title = "%s:heatmap(%s)" % (iname,cname)
            #show_image(heatmap,title)

            src_heatmap = np.expand_dims(heatmap, axis=2)
            for _ in range(2):
                src_heatmap = np.append(src_heatmap, np.expand_dims(heatmap, axis=2),axis=2)
            src_heatmap = image.smart_resize(src_heatmap, (src_image_shape[1],src_image_shape[0]))
            src_heatmap = src_heatmap[:,:,0]
            title = "%s:heatmap-resized(%s)" % (iname,cname)
            show_image(src_heatmap,title)
            title = "%s:super-imposed(%s)" % (iname,cname)
            show_image(src_image,title,src_heatmap,alpha)
