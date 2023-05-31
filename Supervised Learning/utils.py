import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import PIL
from sklearn.metrics import confusion_matrix, classification_report
import os
import gradcam
import tensorflow as tf
from tensorflow import keras

def get_labels_from_tfdataset(tfdataset, batched=False):
    #implementation by https://stackoverflow.com/questions/62436302/extract-target-from-tensorflow-prefetchdataset
    
    labels = list(map(lambda x: x[1], tfdataset)) # Get labels 

    if not batched:
        return tf.concat(labels, axis=0) # concat the list of batched labels

    return labels


def top_(pred, class_set):
    preds = []
    count = 0
    for class_ in class_set:
        preds.append((class_, round(pred[count]*100,2)))
        count += 1
    
    preds.sort(key=lambda x:x[1], reverse=True)
    
    return preds
    

def plot_gradcam(img_path, image_size, model, preprocess_input, last_conv_layer_name, class_set, cam_path = "gradcam.jpg"):
    img_array = preprocess_input(gradcam.get_img_array(img_path, image_size))

    preds = model.predict(img_array)
    preds = top_(preds[0], class_set)
    
    # Remove last layer
    model.layers[-1].activation = None

    # get GradCam image
    heatmap = gradcam.make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    gradcam_img = gradcam.save_and_display_gradcam(img_path, heatmap, cam_path=cam_path)
    return gradcam_img, preds  


def plot_training_images(image_path, rows=3, columns=5, figsize=(15,8)):
    fig, axs = plt.subplots(rows, columns, figsize=figsize)
    
    images = os.listdir(image_path)
    
    count = 0
    for row in axs:
        for col in row:
            img = PIL.Image.open(image_path+images[count])
            col.imshow(img)
            col.axes.get_xaxis().set_ticks([])
            col.axes.get_yaxis().set_ticks([])
            count +=1


def plot_cm(ds, model, class_set):
    y_pred = model.predict(ds)
    predicted_categories = tf.argmax(y_pred, axis=1)

    true_categories = get_labels_from_tfdataset(ds)

    array = confusion_matrix(true_categories, predicted_categories)
    cm = pd.DataFrame(array, class_set, class_set)
    plt.figure(figsize = (10,7))
    ax = sns.heatmap(cm, annot=True, cmap="mako", fmt='g')
    ax.set(xlabel=' Predicted Label ', ylabel=' True Label ')
    plt.show()


def get_classification_report(ds, model):
    y_pred = model.predict(ds)
    predicted_categories = tf.argmax(y_pred, axis=1)

    true_categories = get_labels_from_tfdataset(ds)

    print(classification_report(true_categories, predicted_categories))