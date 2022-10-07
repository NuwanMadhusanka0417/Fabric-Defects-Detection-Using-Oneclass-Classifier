
# -*- coding: utf-8 -*-
# @Time    : 2021-10-10 08.31
# @Author  : Nuwan Madhusanka
# @Email   : nuwan@xdoto.io
# @File    : Train_svm_sep_network.py
# @Software: PyCharm
import getopt
import math
import os
import pickle
import sys

import cv2
import pywt
from datetime import datetime
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import RandomFourierFeatures
# from tensorflow.keras.layers.experimental import
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from keras.models import Model
import joblib
import dill
import weakref

# from tf_cwt import Wavelet1D, Scaler, RGBStack
from wavetf import WaveTFFactory
import tensorflow_wavelets.Layers.DWT as DWT
import tensorflow_wavelets.Layers.DTCWT as DTCWT
import tensorflow_wavelets.Layers.DMWT as DMWT

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
# from sklearn.externals import

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

dataset = ''
gradient = 0.0
ramp = 0.0

def history_(history):
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

# # LOSS FUNCTION 1
# def my_loss_fn(y_true, y_pred):
#     squared_difference = tf.square(y_true - y_pred)
#     return tf.reduce_mean(squared_difference, axis=-1)


# # LOSS FUNCTION 2
# def wrapper(param1):
#     def custom_mse_nuwan(y_true, y_pred):
#         tf.print("y_true : ", y_true, "    y_pred : ", y_pred)
#         squared_difference = tf.square(y_true - y_pred)
#         # tf.print(tf.reduce_mean(squared_difference, axis=-1))
#         return tf.reduce_mean(squared_difference, axis=-1)
#
#     tf.print(param1)
#     return custom_mse_nuwan


#

# LOSS FUNCTION 1 - Hinge Loss
def svm_loss(layer):
    weights = layer.weights[0]
    weights_tf = tf.convert_to_tensor(weights)

    def categorical_hinge_loss(y_true, y_pred):
        pos = K.sum(y_true * y_pred, axis=-1)
        neg = K.max((1.0 - y_true) * y_pred, axis=-1)
        hinge_loss = K.mean(K.maximum(0.0, neg - pos + 1), axis=-1)
        regularization_loss = 0.00001 * (tf.reduce_sum(tf.square(weights_tf)))
        return regularization_loss + hinge_loss

    return categorical_hinge_loss

# LOSS FUNCTION 2 - Ramp Loss
def svm_ramp_loss(layer):
    weights = layer.weights[0]
    weights_tf = tf.convert_to_tensor(weights)

    def categorical_hinge_loss(y_true, y_pred):
        pos = K.sum(y_true * y_pred, axis=-1)
        neg = K.max((1.0 - y_true) * y_pred, axis=-1)
        hinge_loss = K.mean(K.maximum(0.0, neg - pos + 1), axis=-1)
        # Do the ramp process using maximum
        hinge_loss_ramp = K.minimum(0.05,hinge_loss)

        regularization_loss = 0.00001 * (tf.reduce_sum(tf.square(weights_tf)))
        return regularization_loss +  hinge_loss_ramp

    return categorical_hinge_loss

# LOSS FUNCTION 3 - Robust Loss
def svm_robust_loss(layer):
    weights = layer.weights[0]
    weights_tf = tf.convert_to_tensor(weights)

    def categorical_hinge_loss(y_true, y_pred):
        pos = K.sum(y_true * y_pred, axis=-1)
        neg = K.max((1.0 - y_true) * y_pred, axis=-1)
        hinge_loss = K.mean(K.maximum(0.0, neg - pos + 1), axis=-1)
        # Do the ramp process using maximum
        eta = 0.05
        beta = 1/(1 - K.exp(-1 * eta))

        hinge_loss_robust = beta*(1 - K.exp(-1*eta*hinge_loss))



        regularization_loss = 0.00001 * (tf.reduce_sum(tf.square(weights_tf)))
        return regularization_loss +  hinge_loss_robust

    return categorical_hinge_loss

# LOSS FUNCTION 4 -  Robust + Ramp Loss
def svm_robust_with_ramp_loss(layer):
    weights = layer.weights[0]
    weights_tf = tf.convert_to_tensor(weights)

    def categorical_hinge_loss(y_true, y_pred):
        pos = K.sum(y_true * y_pred, axis=-1)
        neg = K.max((1.0 - y_true) * y_pred, axis=-1)
        hinge_loss = K.mean(K.maximum(0.0, neg - pos + 1), axis=-1)
        # Do the ramp process using maximum
        eta = 0.005
        ramp_limit = 8.0
        beta = 1/(1 - K.exp(-1 * eta))

        hinge_loss_robust = beta*(1 - K.exp(-1*eta*hinge_loss))

        hinge_loss_robust_with_ramp = K.minimum(ramp_limit, hinge_loss_robust)
        regularization_loss = 0.00001 * (tf.reduce_sum(tf.square(weights_tf)))
        return regularization_loss +  hinge_loss_robust_with_ramp

    return categorical_hinge_loss

# LOSS FUNCTION 5 - Upper Ramp Loss
def svm_upper_ramp_loss(layer):
    weights = layer.weights[0]
    weights_tf = tf.convert_to_tensor(weights)

    def categorical_hinge_loss(y_true, y_pred):
        pos = K.sum(y_true * y_pred, axis=-1)
        neg = K.max((1.0 - y_true) * y_pred, axis=-1)
        hinge_loss = K.mean(K.maximum(0.0, neg - pos + 1), axis=-1)
        # Do the ramp process using maximum
        hinge_loss_ramp = K.minimum(0.05,hinge_loss)
        if hinge_loss_ramp >0:
            hinge_loss_ramp = hinge_loss_ramp + 0.005
        regularization_loss = 0.00001 * (tf.reduce_sum(tf.square(weights_tf)))
        return regularization_loss +  hinge_loss_ramp

    return categorical_hinge_loss

# LOSS FUNCTION 6 - gradient Ramp Loss
def svm_gradient_ramp_loss(layer):
    weights = layer.weights[0]
    weights_tf = tf.convert_to_tensor(weights)

    def categorical_hinge_loss(y_true, y_pred):
        pos = K.sum(y_true * y_pred, axis=-1)
        neg = K.max((1.0 - y_true) * y_pred, axis=-1)
        grad_ratio = gradient #0.02
        print("grad ratio",grad_ratio)
        print("ramp",ramp)
        hinge_loss = grad_ratio * K.mean(K.maximum(0.0, neg - pos + 1), axis=-1)
        # Do the ramp process using maximum
        hinge_loss_ramp = K.minimum(ramp,hinge_loss)

        regularization_loss = 0.00001 * (tf.reduce_sum(tf.square(weights_tf)))
        return regularization_loss +   hinge_loss_ramp

    return categorical_hinge_loss

def wavelet_activation(x):
    cwtmatr, freqs = pywt.dwt(x,'db1')
    return cwtmatr


# create SVM model
def model_():
    model = keras.Sequential(
        [
            keras.Input(shape=(1,1,4096)),

            # RandomFourierFeatures(
            #     output_dim=4096, scale=10.0, kernel_initializer="gaussian"
            # ),
            # WaveTFFactory.build('db2', 'smooth'),

            # Wavelet1D(batch_size=2,trainable=True),

            # DWT.DWT(name='coif',input_dim=1,concat=0), # ,concat=1

            # layers.Dense(4096,activation = wavelet_activation),

            # DWT.DWT('db1', 'smooth'),
            DWT.DWT(name = 'haar',concat=1),  # ,concat=1
            layers.Flatten(),
            layers.Dense(units=4096), # new update
            layers.Dense(units=8192), # new update
            layers.Dense(units=8192),
            layers.BatchNormalization(),
            layers.Dense(units=4096),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(units=2048),
            layers.Dense(units=512),
            layers.Dense(units=128),

            layers.Dense(units=32),
            layers.Dense(units=2, name='svm'),
        ]
    )
    print(model.summary())
    optimizer = keras.optimizers.Adam(learning_rate=1e-5)  # tf.keras.optimizers.RMSprop(lr=2e-3, decay=1e-5)
    # optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-5, decay=1e-5)
    # loss = my_loss_fn, wrapper(model.layers[0].get_weights()),#keras.losses.hinge,

    # svm_robust_loss
    # svm_ramp_loss
    # svm_loss
    # svm_robust_with_ramp_loss
    # svm_upper_ramp_loss
    # svm_gradient_ramp_loss

    loss = svm_gradient_ramp_loss(model.get_layer('svm')) #
    print("svm_gradient_ramp_loss")

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )

    return model


# Extract features from VGG16
def feature_extract(images_processed_):
    feature_list = []
    # load model
    model = VGG16()
    # remove the output layer
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

    # get extracted features
    for image_prop in images_processed_:
        features = model.predict(image_prop)
        # print(features)
        # print("feature shape : ", features.shape)
        # break
        feature_list.append(np.array(features[0]).reshape(1,1,4096))
    # save to file
    # dump(features, open('dog.pkl', 'wb'))
    return np.array(feature_list)


# pre process image for VGG16
def preprocess_(image):
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    return image


# Read images from train and test directory
def get_data(data_dir,labels):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                # img_arr = cv2.imread(os.path.join(path, img))# [...,::-1] #convert BGR to RGB format
                # resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                data.append([load_img(os.path.join(path, img), target_size=(224, 224)), class_num])
            except Exception as e:
                print(e)
    return data


# get x_train, x_test, y_train, y_test data and preprocess image for VGG16
def load_data(train_path,test_path,labels):
    train = get_data(train_path,labels)
    test = get_data(test_path,labels)

    x_train_ = []
    y_train_ = []
    x_test_ = []
    y_test_ = []
    for feature, label in train:
        x_train_.append(preprocess_(feature))
        y_train_.append(label)

    for feature, label in test:
        x_test_.append(preprocess_(feature))
        y_test_.append(label)

    y_train_ = np.array(y_train_)
    y_test_ = np.array(y_test_)

    num_classes = 2
    y_test_ = tf.keras.utils.to_categorical(y_test_, num_classes)
    y_train_ = tf.keras.utils.to_categorical(y_train_, num_classes)

    return x_train_, x_test_, y_train_, y_test_




def main(argv):
    global dataset,gradient,ramp
    # dataset = ''
    # gradient = ''
    # ramp = ''
    try:
        opts, args = getopt.getopt(argv, "hd:g:r:", ["dataset=", "gradient=", "ramp="])
    except getopt.GetoptError:
        print("Error")
        print('test.py -d <dataset> -g <gradient> -r <ramp limit>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -d <dataset> -g <gradient> -r <ramp limit>')
            sys.exit()
        elif opt in ("-d", "--dataset"):
            dataset = arg
        elif opt in ("-g", "--gradient"):
            gradient = float(arg)
        elif opt in ("-r", "--ramp"):
            ramp = float(arg)
    print('Input file is "', float(gradient))
    print('Output file is "', float(ramp))
    print('Output file is "', dataset)

    print('Input file is "', type(float(gradient)))
    print('Output file is "', type(float(ramp)))
    print('Output file is "', type(dataset))
    # exit()
    if dataset == "Carpet":
        # carpet
        train_path_ = '/home/nuwan/Applications/AI_cource/research/data/carpet/data/train'
        test_path_ = '/home/nuwan/Applications/AI_cource/research/data/carpet/data/test'
        labels = ['defect', 'nondefect']
        print("carpet")
    elif dataset == "UMDAA":
        # # UMDAA
        train_path_ = '/home/nuwan/Applications/AI_cource/research/data/umdaa02-fd-20211017T053249Z-001/new_data/train'
        test_path_ = '/home/nuwan/Applications/AI_cource/research/data/umdaa02-fd-20211017T053249Z-001/new_data/test'
        labels = ['face', 'nonface']
        print("UMDAA")
    elif dataset == "Capsulle":
        # capsulle
        train_path_ = '/home/nuwan/Applications/AI_cource/research/data/capsule/data/train'
        test_path_ = '/home/nuwan/Applications/AI_cource/research/data/capsule/data/test'
        labels = ['defect', 'nondefect']
        print("capsulle")
    elif dataset == "Cable":
        # cable
        train_path_ = '/home/nuwan/Applications/AI_cource/research/data/cable/data/train'
        test_path_ = '/home/nuwan/Applications/AI_cource/research/data/cable/data/test'
        labels = ['defect', 'nondefect']
        print("cable")
    elif dataset == "Hazelnut":
        # hazelnut
        train_path_ = '/home/nuwan/Applications/AI_cource/research/data/hazelnut/data/train'
        test_path_ = '/home/nuwan/Applications/AI_cource/research/data/hazelnut/data/test'
        labels = ['defect', 'nondefect']
        print("hazelnut")
    elif dataset == "Leather":
        # leather
        train_path_ = '/home/nuwan/Applications/AI_cource/research/data/leather/data/train'
        test_path_ = '/home/nuwan/Applications/AI_cource/research/data/leather/data/test'
        labels = ['defect', 'nondefect']
        print("leather")
    elif dataset == "Tile":
        # tile
        train_path_ = '/home/nuwan/Applications/AI_cource/research/data/tile/data/train'
        test_path_ = '/home/nuwan/Applications/AI_cource/research/data/tile/data/test'
        labels = ['defect', 'nondefect']
        print("tile")

    elif dataset == "Transistor":
        # transistor
        train_path_ = '/home/nuwan/Applications/AI_cource/research/data/transistor/data/train'
        test_path_ = '/home/nuwan/Applications/AI_cource/research/data/transistor/data/test'
        labels = ['defect', 'nondefect']
        print("transistor")
    else:
        print("ERROR - datset name input is not correct")
        exit()

    x_train, x_test, y_train, y_test = load_data(train_path_,test_path_,labels)

    x_train = feature_extract(x_train)
    x_test = feature_extract(x_test)

    print(x_train.shape)
    print(y_train.shape)

    model = model_()
    history = model.fit(x_train, y_train, epochs=20, batch_size=20,validation_data=(x_test,y_test)) # , validation_split=0.2

    date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
    filename = f'models/Trained_model_'+dataset+'.sav'
    # pickle.dump(model, open('sdsdsdsd.pkl', 'wb'))
    # with open('model_pkl', 'wb') as files:
    #     pickle.dump(model, files)

    # model.save("models/model.h5")
    # history_(history)
    # with open('models/trainHistoryDict_'+dataset, 'wb') as file_pi:
    #     pickle.dump(history.history, file_pi)

    # dill.dump(model,open('sdsdsdsd.pkl', 'wb'))
    # weakref.d
    # joblib.dump(model,"model.pkl")
    # tf.keras.models.save_model(model, "ffff.h5")
    model.save_weights("ckpt")

    yhat_probs = model.predict(x_test, verbose=0)
    yhat_classes = np.argmax(yhat_probs, axis=1)
    # print("yhat_classes : ", yhat_classes)
    # accuracy: (tp + tn) / (p + n)
    # print("y_test", y_test)
    y_test = y_test[:, 1]
    # print("y_test", y_test)
    accuracy = accuracy_score(y_test, yhat_classes)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(y_test, yhat_classes)
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(y_test, yhat_classes)
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_test, yhat_classes)
    print('F1 score: %f' % f1)
    # kappa
    kappa = cohen_kappa_score(y_test, yhat_classes)
    print('Cohens kappa: %f' % kappa)
    matrix = confusion_matrix(y_test, yhat_classes)
    print(matrix)

if __name__ == "__main__":
    main(sys.argv[1:])