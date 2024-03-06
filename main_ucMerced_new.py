import keras.losses
import numpy as np
import os
import pandas as pd
import random
import tensorflow as tf
# import tensorflow_probability as tfp
from keras import Model
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import load_model
from psutil import virtual_memory
from tensorboard import program
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras import layers, applications
from tensorflow.keras.optimizers import Ftrl, Adagrad, Adadelta, Adam, Adamax, Nadam, RMSprop, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from dataset import get_dataset, process_dataset
from utils import plot_samples, plot_aug_samples, data_augmentation

from keras.utils import np_utils

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def hardware_check():
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        print('GPU not found')
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(virtual_memory().total / 1e9))


if __name__ == '__main__':
    print('Tensorflow version: {}'.format(tf.__version__))
    hardware_check()

    # Experiment Name
    da_method = 2  # 0 (no_data_aug), 1 (flips)

    # Gloabls
    do_print = False
    download_dts = False
    dts_root_path = '/work/dataset/'
    sample_shape = (256, 256, 3)

    # Set random seeds
    SEED = 1699806
    # os.environ['PYTHONHASHSEED']=str(SEED)
    random.seed(SEED)  # python's
    np.random.seed(SEED)  # numpy's
    tf.random.set_seed(SEED)  # tensorflow's

    # get the datasets
    dataset, info = get_dataset('uc_merced', path_dts=dts_root_path, download=download_dts)
    # save the dataset's labels
    dataset_labels = info.features['label'].names
    # save the number of classes
    num_classes = info.features['label'].num_classes
    # save the number of samples
    num_samples = info.splits.total_num_examples

    print('INFO: The dataset is composed of {} samples for {} classes'.format(num_samples, num_classes))

    TRAIN_TEST_SPLIT = 0.7
    x_train, y_train, x_test, y_test = process_dataset(dataset=dataset, num_samples=num_samples,
                                                       sample_shape=sample_shape, num_classes=num_classes,
                                                       tt_split=TRAIN_TEST_SPLIT)
    

    if do_print:
        plot_samples(x_test, y_test, hparam_root_path, dataset_labels, samples_to_print=5)

    # Data Augmentation
    da_method_chosen, transform_parameters = data_augmentation(method=da_method)

    # VAL_TRAIN_SPLIT = 0.2  # how many samples (in fraction) are reserved for validation
    
    print(y_train.shape, y_test.shape, x_train.shape, x_test.shape)

    datagen = ImageDataGenerator(
        rotation_range=transform_parameters['theta'],
        width_shift_range=transform_parameters['tx'],
        height_shift_range=transform_parameters['ty'],
        zoom_range=transform_parameters['zoom'],
        horizontal_flip=transform_parameters['flip_horizontal'],
        vertical_flip=transform_parameters['flip_vertical'],
        shear_range=transform_parameters['shear_range'],
        # validation_split=VAL_TRAIN_SPLIT
    )
    
    datagen.fit(x_train)
    
    #reshape into images
    x_train = x_train.reshape(x_train.shape[0], sample_shape[0], sample_shape[1], sample_shape[2])
    x_test = x_test.reshape(x_test.shape[0], sample_shape[0], sample_shape[1], sample_shape[2])
    
    #convert integers to float; normalise and center the mean
    x_train=x_train.astype("float32")  
    x_test=x_test.astype("float32")
    mean=np.mean(x_train)
    std=np.std(x_train)
    x_test=(x_test-mean)/std
    x_train=(x_train-mean)/std

    print('Method {} applied for Data Augmentation'.format(da_method_chosen))

    if do_print:
        plot_aug_samples(x_train, y_train, hparam_root_path, dataset_labels, datagen, transform_parameters, samples_to_print=5)

    def train_test_model(opt):
    
        run_dir='/work/save_model/' + opt + '/'
        
        base_model = applications.mobilenet_v2.MobileNetV2(include_top=False, weights='imagenet', input_shape=sample_shape)
        base_model = applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=sample_shape)
    	
        for layer in base_model.layers:
               layer.trainable = True
        
        avg_pool = layers.AveragePooling2D((8,8), strides=1)(base_model.output)
        flatten = layers.Flatten()(avg_pool)
        fc1 = layers.Dense(512, activation='relu')(flatten)
        fc2 = layers.Dense(256, activation='relu')(fc1)
        fc3 = layers.Dense(num_classes, activation='softmax')(fc2)
        
        model_aug = Model(inputs=base_model.input, outputs=fc3) # Model(inputs=inputs, outputs=out)
        model_aug.summary()
        
        # optimizer
        if opt == 'Adam_opt':
        	optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.9999, epsilon=1e-08, amsgrad=True)
        elif opt == 'Adamax_opt':
        	optimizer = Adamax(learning_rate=0.001, beta_1=0.6, beta_2=0.99, epsilon=1e-06)
       	elif opt == 'Nadam_opt':
       		optimizer = Nadam(learning_rate=0.001, beta_1=0.99, beta_2=0.99, epsilon=1e-06)
       	elif opt == 'RMSprop_opt':
       		optimizer = RMSprop(learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-06, centered=False)
       	elif opt == 'SGD_opt_def':
       		optimizer = SGD(learning_rate=0.01, momentum=0.0, nesterov=False)
       		
       	elif opt == 'Adagrad_def':
       		optimizer = Adagrad()
       	elif opt == 'FTRL_def':
       		optimizer = Ftrl()
       	elif opt == 'Adadelta_def':
       		optimizer = Adadelta()
        
        elif opt == 'Adam_def':
        	optimizer = Adam()
        elif opt == 'Adamax_def':
        	optimizer = Adamax()
       	elif opt == 'Nadam_def':
       		optimizer = Nadam()
       	elif opt == 'RMSprop_def':
       		optimizer = RMSprop()
       		
       	else: 
       		print('Error')
       		exit(0)
       		
        # compile
        model_aug.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        # callback
        callbacks = [ModelCheckpoint(run_dir + '_best_model.h5', monitor='val_accuracy', save_best_only=True, verbose=0)]
        # run sample training
        print('Running sample training...')
        history = model_aug.fit(x_train, y_train, batch_size=32,
                                epochs=30,
                                validation_data=(x_test, y_test),
                                callbacks=callbacks,
                                verbose=1)

        # save history to csv:
        hist_df = pd.DataFrame(history.history)
        hist_csv_file = run_dir + '_history.csv'
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)

        model_aug = load_model(run_dir + '_best_model.h5', compile=True)

        # obtain final accuracy
        _, accuracy = model_aug.evaluate(x_test, y_test)
        
        print('...................................................')
        print(opt, accuracy)
        print('...................................................')

        return
    
    train_test_model(opt='Adam_opt')
    train_test_model(opt='Adam_def')
    train_test_model(opt='Nadam_opt')
    train_test_model(opt='Nadam_def')
    train_test_model(opt='RMSprop_opt')
    train_test_model(opt='RMSprop_def')
    train_test_model(opt='Adamax_opt')
    train_test_model(opt='Adamax_def')
    train_test_model(opt='SGD_opt_def')


    print('\n#-------------------#\n# Process completed #\n#-------------------#')
