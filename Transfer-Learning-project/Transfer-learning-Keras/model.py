import os
import cv2
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import sys
from keras.models import load_model

dFlag = 0
def debug_print(debug_flag = 1, *arg):
    if(debug_flag):
        print(arg)

class cfg(object):
    """
    configuration class to set/pass different parameter arguments
    """
    def __init__(self):
        self.batch_size = 16
        self.load_file = None
        self.save_file = None
        self.epochs = 5
        self.flip_augment_flag = 0
        self.flipCode = 0
        self.rotate_augment_flag = 0
        self.lateral_shift_augment_flag = 0
        self.shift = 0
        self.val_img = None
        self.train_img = None
        self.img_path = ''
        self.input_shape = (0,0,3)
        self.yuv_flag = 0

    def set_batch_size(self, batch_size = 16):
        self.batch_size = batch_size

    def set_load_file_name(self, load_file_name):
        self.load_file = load_file_name

    def set_save_file_name(self, save_file_name):
        self.save_file = save_file_name

    def set_epochs(self, epochs=5):
        self.epochs = epochs

    def set_flip(self, flip_flag = 0, flip_mode = 0):
        self.flip_augment_flag = flip_flag
        if(flip_flag) :
            self.flipCode = flip_mode

    def set_shift(self, shift_flag = 0, shift_pixel = 0):
        self.lateral_shift_augment_flag = shift_flag
        if(shift_flag) :
            self.shift = shift_pixel
    def print_model_cfg(self):
        print("batch_size:{}".format(batch_size))


def parse_train_csv(path):
    """
    function to parse csv file

    input : path of the CSV file
    return: tuple of train, val image and image path(not needed)
    """

    #read and open csv file
    curr_path = os.getcwd()
    data_path = curr_path + "/data/data/"
    data_path=path
    debug_print(curr_path)
    debug_print(data_path)
    img_data = []
    with open(data_path + "/driving_log.csv", "rt") as f:
        data = csv.reader(f, delimiter=",", skipinitialspace=True)
        count = 0
        for row in data:
            if(count):
                img_data.append(row)
            count +=1
    debug_print(0, img_data)
    #debug_print(img_data.shape)
    # split into train and validation
    img_data = shuffle(img_data)
    train_img_data, val_img_data = train_test_split(img_data, test_size=0.20)
    debug_print(dFlag, len(val_img_data), len(train_img_data), len(img_data))
    debug_print(dFlag, val_img_data[1], train_img_data[1])
    img_path = data_path
    return train_img_data, val_img_data, img_path

#def

import matplotlib.pyplot as plt

#def generate_train_batch(train_img_data, batch_size, img_path, cfg=None):
def generate_train_batch(cfg = None):

    """
    Train data generator which provides batch of train set examples
    """
    assert (cfg != None)

    train_img_data = cfg.train_img
    batch_size     = cfg.batch_size
    img_path       = cfg.img_path

    assert (train_img_data != None)
    assert (batch_size != 0)
    assert (img_path != None)

    print(batch_size, img_path)
    steer_correction = 0.2

    rgb2yuv = cfg.yuv_flag
    flip_flag = cfg.flip_augment_flag
    shift_flag = cfg.lateral_shift_augment_flag

    train_size = len(train_img_data)
    assert train_size != 0

    if 0: #for testing few train images
        train_size = 128
        train_img_data = train_img_data[:128]
    #print(train_size, len(train_img_data))

    while 1:
        #shuffle the data
        shuffle(train_img_data)
        for row in  range(0, int(train_size/batch_size)*batch_size,  batch_size):
            end_idx = row + batch_size
            local_batch_size = batch_size
            if 0:
                if(end_idx >= train_size):
                    end_idx = train_size-1
                    local_batch_size = end_idx - row

            train_batch = train_img_data[row:end_idx]
            train_image_db = []
            train_steer_db = []
            mu, sigma = 0, 6
            shift1 = (np.random.normal(mu,sigma, batch_size))

            # print("LBS:", row, end_idx,local_batch_size)
            for img_index in range(local_batch_size):
                center_img = cv2.imread(img_path + train_batch[img_index][0])
                left_img =   cv2.imread(img_path + train_batch[img_index][1])
                right_img =   cv2.imread(img_path + train_batch[img_index][2])
                center_steer = float(train_batch[img_index][3])

                if rgb2yuv != 1:
                    train_image_db.append(center_img)
                    train_image_db.append(left_img)
                    train_image_db.append(right_img)
                else:
                    center_yuv = cv2.cvtColor(center_img, cv2.COLOR_BGR2YUV)
                    left_yuv = cv2.cvtColor(left_img, cv2.COLOR_BGR2YUV)
                    right_yuv = cv2.cvtColor(right_img, cv2.COLOR_BGR2YUV)
                    train_image_db.append(center_yuv)
                    train_image_db.append(left_yuv)
                    train_image_db.append(right_yuv)
                    center_img = center_yuv
                    left_img = left_yuv
                    right_img = right_yuv



                train_steer_db.append(center_steer)
                train_steer_db.append(center_steer + steer_correction)
                train_steer_db.append(center_steer - steer_correction)
                #mirror image

                if flip_flag == True:
                    #print("Flipping the image ", flip_flag)

                    center_flip = cv2.flip(center_img, flipCode=1)
                    train_image_db.append(center_flip)
                    train_steer_db.append(-center_steer)

                    left_flip = cv2.flip(left_img, flipCode=1)
                    train_image_db.append(left_flip)
                    train_steer_db.append(-(center_steer + steer_correction))

                    right_flip = cv2.flip(right_img, flipCode=1)
                    train_image_db.append(right_flip)
                    train_steer_db.append(-(center_steer - steer_correction))


                if shift_flag == True:
                    #print("Shift the image ", shift_flag)
                    #randomize the shift
                    #mu, sigma = 0, 8
                    #shift = np.random.normal(mu,sigma)
                    #shift = 8
                    #shift = int(shift)
                    shift = int(shift1[img_index])
                    left_M = np.float32([[1,0,-shift],[0,1,0]])
                    right_M = np.float32([[1,0,shift],[0,1,0]])
                    correction_factor = 1.00 + float(abs(shift) / 100)
                    if(shift < 0 ):
                    #print("!!!!!!!",temp_left.shape, temp_right.shape, (left_img.shape[1]-64), (right_img.shape[1]+64))
                        left_shift_img = cv2.warpAffine(left_img,left_M,(left_img.shape[1],left_img.shape[0]))
                        temp_left = left_shift_img[:,:(left_img.shape[1]-shift),:]

                        left_shift_img = cv2.resize(temp_left, (left_img.shape[1],left_img.shape[0]))
                        left_shift_img_steer = center_steer + correction_factor * steer_correction
                        train_image_db.append(left_shift_img)
                        train_steer_db.append(left_shift_img_steer)
                    else:
                        right_shift_img = cv2.warpAffine(right_img,right_M,(right_img.shape[1],right_img.shape[0]))
                        temp_right = right_shift_img[:, (shift):, :]

                        right_shift_img = cv2.resize(temp_right,(right_img.shape[1],right_img.shape[0]))
                        right_shift_img_steer = center_steer - correction_factor * steer_correction
                        train_image_db.append(right_shift_img)
                        train_steer_db.append(right_shift_img_steer)

                #print(left_shift_img.shape, right_shift_img.shape)
                #rotation and lateral shift to be added

                #print(center_img.shape, left_img.shape, right_img.shape)
                if 0:
                    print(img_path + train_batch[0][0])
                    print(img_path + train_batch[0][1])
                    print(img_path + train_batch[0][2])
                    plt.figure()
                    plt.subplot(1,3,1)
                    plt.imshow(center_img)
                    plt.subplot(1,3,2)
                    plt.imshow(left_img)
                    plt.subplot(1,3,3)
                    plt.imshow(right_img)
                    #plt.show()

            #print("batch row:", row)
            yield shuffle(np.array(train_image_db), np.array(train_steer_db))

#def generate_val_batch(val_img_data, batch_size, img_path, cfg=None):
def generate_val_batch(cfg=None):
    """
    Cross Validation data generation
    """
    assert cfg != None
    val_img_data =  cfg.val_img
    batch_size =    cfg.batch_size
    img_path =      cfg.img_path

    assert batch_size != 0
    assert img_path != None
    assert val_img_data != None
    rgb2yuv = cfg.yuv_flag
    val_size = len(val_img_data)
    assert val_size != 0
    if 0:
        val_img_data = val_img_data[:]
        val_size = 16

    while 1:
        #print("generate_val_batch")
        for row in  range(0, int(val_size/batch_size)*batch_size,  batch_size):
            end_idx = row + batch_size
            local_batch_size = batch_size
            if 0:
                if(end_idx >= val_size):
                    end_idx = val_size-1
                    local_batch_size = end_idx - row
            val_batch = val_img_data[row:end_idx]
            val_image_db = []
            val_steer_db = []
            #print("val", row, end_idx, local_batch_size)
            for img_index in range(local_batch_size):
                center_img = cv2.imread(img_path + val_batch[img_index][0])
                center_steer = float(val_batch[img_index][3])

                if rgb2yuv != 1:
                    val_image_db.append(center_img)
                else:
                    center_yuv = cv2.cvtColor(center_img, cv2.COLOR_BGR2YUV)
                    val_image_db.append(center_yuv)

                val_steer_db.append(center_steer)

                #print(center_img.shape, left_img.shape, right_img.shape)
                if 0:
                    print(img_path + val_batch[0][0])
                    print(img_path + val_batch[0][1])
                    print(img_path + val_batch[0][2])
                    plt.figure()
                    plt.subplot(1,3,1)
                    plt.imshow(center_img)
                    plt.subplot(1,3,2)
                    plt.imshow(left_img)
                    plt.subplot(1,3,3)
                    plt.imshow(right_img)
                    #plt.show()

                #print("center_sterr:", center_steer)
            yield np.array(val_image_db), np.array(val_steer_db)

from keras.models import  Sequential
from keras.layers import  Dense, Reshape, Lambda, Flatten, Activation, Conv2D #, Activation, Flatten, MaxPool
import  keras.initializers as init
from keras.layers.convolutional import Cropping2D
from keras import regularizers
from keras.layers import Dropout


def driveNet(model):
    """
    Model Architecture based on Nvidia end to end Conv net used in its driving car
    """
    #conv layer 24@31x98

    #kernal_init = "random_normal"
    kernal_init = "glorot_uniform"
    kernel_init = init.RandomNormal(mean=0.0, stddev=0.01, seed=None)
    reg_lambda = 0.0004
    reg_lambda = 0.00099
    reg_lambda = 0.00010
    regularization = regularizers.l2(reg_lambda)
    if 0: # keras 1 interface
        model.add(Conv2D(nb_filter= 24, nb_row=5, nb_col=5, subsample = (2,2), border_mode='valid', activation="relu", init=kernal_init))
        model.add(Conv2D(nb_filter=36, nb_row = 5,nb_col=5, subsample = (2,2), border_mode='valid', activation="relu",  init=kernal_init))
        model.add(Conv2D(nb_filter=48, nb_row = 5, nb_col=5, subsample = (2,2), border_mode='valid', activation="relu",  init=kernal_init))
        model.add(Conv2D(nb_filter=64, nb_row = 3, nb_col=3, subsample = (2,2), border_mode='valid', activation="relu",  init=kernal_init))
        model.add(Conv2D(nb_filter=64, nb_row = 3, nb_col=3, subsample = (2,2), border_mode='valid', activation="relu",  init=kernal_init))
    else:
        if 1:
            #print("Add new layer")
            #    model.add(Conv2D(filters= 24, kernel_size=(3, 3), strides = (1,1), padding='valid', activation="relu", kernel_initializer=kernal_init,
            #    kernel_regularizer=regularization))

            model.add(Conv2D(filters= 24, kernel_size=(5, 5), strides = (2,2), padding='valid', activation="relu", kernel_initializer=kernal_init,
            kernel_regularizer=regularization))
            model.add(Conv2D(filters=36, kernel_size= (5,5),   strides = (2,2),    padding='valid', activation="relu",  kernel_initializer=kernal_init,
            kernel_regularizer=regularization))
            model.add(Conv2D(filters=48, kernel_size= (5,5), strides = (2,2), padding='valid', activation="relu",  kernel_initializer=kernal_init,
            kernel_regularizer=regularization))

            model.add(Conv2D(filters=64, kernel_size= (3,3), strides = (1,1), padding='valid', activation="relu",  kernel_initializer=kernal_init,
            kernel_regularizer=regularization))
            model.add(Conv2D(filters=64, kernel_size= (3,3), strides = (1,1), padding='valid', activation="relu",  kernel_initializer=kernal_init,
            kernel_regularizer=regularization))

            model.add(Conv2D(filters=64, kernel_size= (3,3), strides = (1,2), padding='valid', activation="relu",  kernel_initializer=kernal_init,
            kernel_regularizer=regularization))

    model.add(Flatten())
    #model.add(Dense(500, activation="relu", kernel_initializer=kernal_init,
    #kernel_regularizer=regularization))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation="relu", kernel_initializer=kernal_init,
    kernel_regularizer=regularization))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation="relu", kernel_initializer=kernal_init,
    kernel_regularizer=regularization))
    #model.add(Dropout(0.5))
    model.add(Dense(10, activation="relu", kernel_initializer=kernal_init,
    kernel_regularizer=regularization))
    #model.add(Dropout(0.5))
    model.add(Dense(1, kernel_initializer=kernal_init, kernel_regularizer=regularization))




#def train_model(train_img, val_img, img_path, input_shape, batch_size = 16, load_model_file=None, save_model=None, model_cfg = None):
def train_model(model_cfg = None):
    """
        Keras sequential model used to create deep convnet, train and evaluate the sample test data
    """

    assert model_cfg != None

    train_img   =    model_cfg.train_img
    val_img     =    model_cfg.val_img
    img_path    =    model_cfg.img_path

    input_shape =    model_cfg.input_shape
    batch_size  =    model_cfg.batch_size

    load_model_file = model_cfg.load_file
    save_model  =     model_cfg.save_file
    epochs      =     model_cfg.epochs

    assert ( (train_img != None )  and (len(train_img) != 0) )
    assert ( (val_img != None ) and (len(val_img) != 0) )
    assert img_path != None

    assert batch_size != 0
    assert ( (input_shape[0] != 0) and (input_shape[1] != 0) )
    assert epochs != 0

    print(load_model_file, save_model)

    if 1:
        if load_model_file == None:
            model = Sequential()
            model.add(Lambda(lambda x: (x-128.0)/128.0, input_shape=input_shape, output_shape=input_shape))
            if 1:
            	model.add(Cropping2D(cropping=((64,0),(0,0))))
            	driveNet(model)

            if 0: #testing basic model
            	target_shape = input_shape[0]*input_shape[1]*input_shape[2]
            	print("!!!", target_shape, input_shape)
            	#model.add(Flatten())
            	model.add(Reshape((target_shape,) ))
            	#model.add(Reshape((target_shape, ), input_shape= input_shape))
            	#model.add(Lambda(lambda x: (x-128.0)/128.0))
            	#model.add(Dense(200))
            	#model.add(Activation('relu'))
            	model.add(Dense(1))

            model.compile(optimizer= 'adam', loss='mse')

        else:
            print("loading previous saved file {0}".format(load_model_file))
            model = load_model(load_model_file)
            #model.load(load_model)

        #train_generator = generate_train_batch(train_img, batch_size, img_path, cfg=model_cfg)
        #val_generator = generate_val_batch(val_img, batch_size, img_path, cfg=model_cfg)

        train_generator = generate_train_batch(cfg=model_cfg)
        val_generator = generate_val_batch(cfg=model_cfg)

        hist = model.fit_generator(train_generator, steps_per_epoch=int(len(train_img[:])/batch_size), validation_data=val_generator,
                validation_steps=int(len(val_img[:])/batch_size), epochs=epochs)

        if save_model == None:
            save_model = "my_model_1_user_comb.h5"
        print("saving the model as ", save_model)
        model.save(save_model)

        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.title('model mean squared error loss')
        plt.ylabel('mean squared error loss')
        plt.xlabel('epoch')
        plt.legend(['training set', 'validation set'], loc='upper right')
        plt.show()

def main(argv):

    model_cfg = cfg()
    load_file = None
    save_model_file = None

    #need to use arg parser module to pass cmdline argument
    if len(sys.argv) == 4:
        load_file = argv[3]
    if len(sys.argv) >= 3:
        save_model_file = argv[2]

    print(load_file, save_model_file)
    #parse the csv file and create the list of train, val images
    # containing path of the images, target parameters
    train_img_data, val_img_data, img_path = parse_train_csv(argv[1])

    input_shape = plt.imread(img_path + train_img_data[0][0]).shape
    print(input_shape)

    #set cfg class object
    model_cfg.set_load_file_name(load_file)
    model_cfg.set_save_file_name(save_model_file)

    model_cfg.val_img =     val_img_data
    model_cfg.train_img =   train_img_data
    model_cfg.img_path =    img_path

    # SGD related cfg field
    model_cfg.set_epochs(epochs=80)
    model_cfg.set_batch_size(32)
    model_cfg.input_shape = input_shape

    # Augment related cfg field
    model_cfg.flip_augment_flag = 0
    model_cfg.lateral_shift_augment_flag = 1
    model_cfg.yuv_flag = 1


    print(model_cfg)

    #train_model(train_img_data, val_img_data, img_path, input_shape, batch_size = 32,
    #save_model=save_model_file, load_model_file=load_file, model_cfg = model_cfg)
    train_model(model_cfg)


if __name__ == "__main__":
    print("Behavioural Cloning for autonomus car")
    main(sys.argv)
