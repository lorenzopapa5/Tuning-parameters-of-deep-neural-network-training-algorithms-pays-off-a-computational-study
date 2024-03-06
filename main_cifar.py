import keras.losses
import numpy as np
import os
import pandas as pd
import random
import keras as k
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
from keras.datasets import cifar100, cifar10
from keras.utils import np_utils
# from dataset import get_cifar10_dataset
from utils import plot_samples, plot_aug_samples, data_augmentation

from keras.initializers import GlorotUniform, lecun_normal

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

    # Gloabls
    img_rows, img_cols , channels= 32, 32, 3
    # Set random seeds
    SEED = 1699806
    os.environ['PYTHONHASHSEED']=str(SEED)
    random.seed(SEED)  # python's
    np.random.seed(SEED)  # numpy's
    tf.random.set_seed(SEED)  # tensorflow's
    
    
    # Load data
    # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    
    # Data augmentation    
    datagen = ImageDataGenerator(rotation_range=15, horizontal_flip=True, vertical_flip=True, width_shift_range=0.1, height_shift_range=0.1) #zoom_range=0.3
    datagen.fit(x_train)
    
    #reshape into images
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
    sample_shape = (img_rows, img_cols, channels)
    
    #convert integers to float; normalise and center the mean
    x_train=x_train.astype("float32")  
    x_test=x_test.astype("float32")
    mean=np.mean(x_train)
    std=np.std(x_train)
    x_test=(x_test-mean)/std
    x_train=(x_train-mean)/std
    
    # labels
    num_classes=100
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)


    # ------------------------------------------------------------------- TRAIN NO AUG -------------------------------------------------------------------------------- # 
    def train_test_model_no_aug(exp_name, EPOCHS, opt):
    
        run_dir = '/work/save_model/' + exp_name + '/'
        if not os.path.exists(run_dir):
                os.makedirs(run_dir)

        def _conv_block(input_tensor, filters, kernel, dr_range, pool):
            x = layers.Conv2D(filters=filters,
                              kernel_size=kernel,
                              activation='relu',
                              padding='same',
                              strides=1,
                              kernel_initializer=GlorotUniform(seed=SEED),
                              input_shape=sample_shape)(input_tensor)
            # x = tf.keras.activations.swish(x)  # A ReLU differentiable in all its points!
            if pool:
                x = layers.MaxPooling2D(pool_size=(2, 2), padding='valid')(x)
                # x = layers.AveragePooling2D(pool_size=(2, 2), padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(dr_range)(x)

            return x

        def _dense_block(input_tensor, filters, dr_range):
            x = layers.Dense(units=filters,
                             kernel_initializer=GlorotUniform(seed=SEED),
                             activation='relu')(input_tensor)
            x = layers.Dropout(dr_range)(x)

            return x

        # define model
        inputs = layers.Input(shape=sample_shape)
        conv0 = _conv_block(inputs, filters=16, kernel=(11, 11), dr_range=0.3, pool=True)
        conv1 = _conv_block(conv0, filters=32, kernel=(9, 9), dr_range=0.3, pool=True)
        conv2 = _conv_block(conv1, filters=64, kernel=(7, 7), dr_range=0.3, pool=True)
        conv3 = _conv_block(conv2, filters=128, kernel=(5, 5), dr_range=0.3, pool=True)
        conv4 = _conv_block(conv3, filters=256, kernel=(3, 3), dr_range=0.3, pool=True)
        flatten = layers.Flatten()(conv4)
        fc0 = _dense_block(flatten, filters=256, dr_range=0.25)
        fc1 = layers.Dense(num_classes)(fc0)
        out = layers.Activation('softmax')(fc1)

        # optimizer
        if opt == 'Adam':
        	optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.9999, epsilon=1e-08, amsgrad=True)
        elif opt == 'Adamax':
        	optimizer = Adamax(learning_rate=0.001, beta_1=0.6, beta_2=0.99, epsilon=1e-06)
       	elif opt == 'Nadam':
       		optimizer = Nadam(learning_rate=0.001, beta_1=0.99, beta_2=0.99, epsilon=1e-06)
       	elif opt == 'RMSprop':
       		optimizer = RMSprop(learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-06, centered=False)
       	elif opt == 'SGD':
       		optimizer = SGD(learning_rate=0.01, momentum=0.0, nesterov=False)
       	elif opt == 'Adagrad':
       		optimizer = Adagrad()
       	elif opt == 'FTRL':
       		optimizer = Ftrl()
       	elif opt == 'Adadelta':
       		optimizer = Adadelta()
       	else: 
       		print('Error')
       		exit(0)
       		
        model_no_aug = Model(inputs=inputs, outputs=out)
        model_no_aug.summary()
        # compile 
        model_no_aug.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        # callback
        callbacks = [ModelCheckpoint(run_dir + 'best_model_no_aug.h5', monitor='val_accuracy', save_best_only=True, verbose=0)]
        # run sample training
        print('Running sample training...')
        history_no_aug = model_no_aug.fit(x_train, y_train, batch_size=128, epochs=EPOCHS, validation_data=(x_test, y_test), verbose=1, callbacks=callbacks)
        # save history to csv
        hist_df_no_aug = pd.DataFrame(history_no_aug.history)
        hist_csv_file = run_dir + 'history_no_aug.csv'
        with open(hist_csv_file, mode='w') as f:
	        hist_df_no_aug.to_csv(f)
	
	#training accuracy without dropout
        test_acc=model_no_aug.evaluate(x_test, y_test, batch_size=128) # SBAGLIATO PRIMA CARICA BEST MODEL
        print(test_acc)
 
        return
        
        
        
    # ------------------------------------------------------------------- TRAIN AUG -------------------------------------------------------------------------------- #         
    def train_test_model_aug(exp_name, EPOCHS, opt):
    
        run_dir = '/work/save_model/' + exp_name + '/'
        if not os.path.exists(run_dir):
                os.makedirs(run_dir)
        
        tf.random.set_seed(SEED)  # tensorflow's
        
             
        def _conv_block(input_tensor, filters, kernel, dr_range, pool):
            x = layers.Conv2D(filters=filters,
                              kernel_size=kernel,
                              activation='relu',
                              padding='same',
                              strides=1,
                              kernel_initializer=GlorotUniform(seed=SEED),
                              input_shape=sample_shape)(input_tensor)
            # x = tf.keras.activations.swish(x)  # A ReLU differentiable in all its points!
            if pool:
                x = layers.MaxPooling2D(pool_size=(2, 2), padding='valid')(x)
                # x = layers.AveragePooling2D(pool_size=(2, 2), padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(dr_range)(x)

            return x

        def _dense_block(input_tensor, filters, dr_range):
            x = layers.Dense(units=filters,
                             kernel_initializer=GlorotUniform(seed=SEED),
                             activation='relu')(input_tensor)
            x = layers.Dropout(dr_range)(x)

            return x

        # define model
        inputs = layers.Input(shape=sample_shape)
        conv0 = _conv_block(inputs, filters=16, kernel=(11, 11), dr_range=0.3, pool=False)
        conv00 = _conv_block(conv0, filters=16, kernel=(3, 3), dr_range=0.3, pool=True)
        conv1 = _conv_block(conv00, filters=32, kernel=(9, 9), dr_range=0.3, pool=False)
        conv11 = _conv_block(conv1, filters=32, kernel=(3, 3), dr_range=0.3, pool=True)
        conv2 = _conv_block(conv11, filters=64, kernel=(7, 7), dr_range=0.3, pool=False)
        conv22 = _conv_block(conv2, filters=64, kernel=(3, 3), dr_range=0.3, pool=True)
        conv3 = _conv_block(conv22, filters=128, kernel=(5, 5), dr_range=0.3, pool=False)
        conv33 = _conv_block(conv3, filters=128, kernel=(3, 3), dr_range=0.3, pool=True)
        conv4 = _conv_block(conv33, filters=256, kernel=(3, 3), dr_range=0.3, pool=False)
        conv44 = _conv_block(conv4, filters=256, kernel=(3, 3), dr_range=0.3, pool=True)
        flatten = layers.Flatten()(conv44)
        fc0 = _dense_block(flatten, filters=256, dr_range=0.25)
        fc1 = layers.Dense(num_classes)(fc0)
        out = layers.Activation('softmax')(fc1)
        
        
        # optimizer
        if opt == 'Adam_opt':
        	optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.9999, epsilon=1e-08, amsgrad=True)
        elif opt == 'Adamax_opt':
        	optimizer = Adamax(learning_rate=0.001, beta_1=0.6, beta_2=0.99, epsilon=1e-06)
       	elif opt == 'Nadam_opt':
       		optimizer = Nadam(learning_rate=0.001, beta_1=0.99, beta_2=0.99, epsilon=1e-06)
       	elif opt == 'RMSprop_opt':
       		optimizer = RMSprop(learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-06, centered=False)
       	elif opt == 'SGD_opt':
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
        elif opt == 'SGD_def':
       		optimizer = SGD()
       		
       	else: 
       		print('Error')
       		exit(0)
       		
       		
        
        # base_model = applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=sample_shape)
        # base_model = applications.mobilenet_v2.MobileNetV2(include_top=False, weights='imagenet', input_shape=sample_shape)
        
        #for layer in base_model.layers:
        #       layer.trainable = True
        
        #flatten = layers.Flatten()(base_model.output)
        #fc1 = layers.Dense(512, activation='relu')(flatten)
        #fc2 = layers.Dense(256, activation='relu')(fc1)
        #fc3 = layers.Dense(num_classes, activation='softmax')(fc2)
        
        #model_aug = Model(inputs=base_model.input, outputs=fc3)
 
        model_aug = Model(inputs=inputs, outputs=out)
        model_aug.summary()
	# compile
        model_aug.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	# callback
        callbacks = [#TensorBoard(log_dir=log_dir, update_freq='epoch', histogram_freq=2),
	             ModelCheckpoint(run_dir + 'best_model_aug.h5', monitor='val_accuracy', save_best_only=True, verbose=0),
	             #hp.KerasCallback(log_dir, hparams)
	             ]
	# run sample training
        print('Running sample training...')
        history_aug = model_aug.fit_generator(datagen.flow(x_train, y_train, batch_size=128), steps_per_epoch = len(x_train) / 128, epochs=EPOCHS, validation_data=(x_test, y_test), callbacks=callbacks)
	# save history to csv:
        hist_df_aug = pd.DataFrame(history_aug.history)
        hist_csv_file = run_dir + 'history_aug.csv'
        with open(hist_csv_file, mode='w') as f:
                hist_df_aug.to_csv(f)
	   
	#training accuracy without dropout
        test_acc=model_aug.evaluate(x_test, y_test, batch_size=128) # SBAGLIATO PRIMA CARICA BEST MODEL
        print(test_acc)

        return 
        
        
    # ------------------------------------------------------------------- TRAIN LBFGS -------------------------------------------------------------------------------- #         
    def train_LBFGS(exp_name, seed):
    
        run_dir = '/work/save_model/' + exp_name + '/'
        if not os.path.exists(run_dir):
                os.makedirs(run_dir)
    
    	# Set random seeds
        SEED = seed
        os.environ['PYTHONHASHSEED']=str(SEED)
        random.seed(SEED)  # python's
        np.random.seed(SEED)  # numpy's
        tf.random.set_seed(SEED)  # tensorflow's
    	
        print(f'The SEED is: {SEED}')
    	
        def _conv_block(input_tensor, filters, kernel, dr_range, pool):
            x = layers.Conv2D(filters=filters,
                              kernel_size=kernel,
                              # activation='relu',
                              padding='same',
                              strides=2, # 1
                              kernel_initializer=lecun_normal(seed=SEED), # GlorotUniform, lecun_normal
                              input_shape=sample_shape)(input_tensor)
            x = tf.keras.activations.swish(x)  # A ReLU differentiable in all its points!
            if pool:
                x = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
                # x = layers.AveragePooling2D(pool_size=(2, 2), padding='same')(x)
            x = layers.BatchNormalization()(x)
            # x = layers.Dropout(dr_range)(x)

            return x

        def _dense_block(input_tensor, filters, dr_range):
            x = layers.Dense(units=filters,
                             kernel_initializer=lecun_normal(seed=SEED), # GlorotUniform, lecun_normal
                             activation='relu')(input_tensor)
            # x = layers.Dropout(dr_range)(x)

            return x

        # define model
        inputs = layers.Input(shape=sample_shape)
        conv0 = _conv_block(inputs, filters=16, kernel=(11, 11), dr_range=0.3, pool=False)
        conv1 = _conv_block(conv0, filters=32, kernel=(9, 9), dr_range=0.3, pool=False)
        conv2 = _conv_block(conv1, filters=64, kernel=(7, 7), dr_range=0.3, pool=False)
        conv3 = _conv_block(conv2, filters=128, kernel=(5, 5), dr_range=0.3, pool=False)
        conv4 = _conv_block(conv3, filters=256, kernel=(3, 3), dr_range=0.3, pool=False)
        flatten = layers.Flatten()(conv4)
        fc0 = _dense_block(flatten, filters=256, dr_range=0.25)
        fc1 = layers.Dense(num_classes)(fc0)
        out = layers.Activation('softmax')(fc1)
        
                
        model = Model(inputs=inputs, outputs=out)
        # model.summary()

        from bfgs import bfgs_train
        bfgs_train(model=model, train_x=x_train, train_y=y_train, test_x=x_test, test_y=y_test)

        return 

    # train_LBFGS(exp_name='LBFGS_LC_Opt_cifar10', seed=1699806) 
    # train_LBFGS(exp_name='LBFGS_LC_Opt_cifar10', seed=1000) 
    # train_LBFGS(exp_name='LBFGS_LC_Opt_cifar10', seed=100) 
    # train_LBFGS(exp_name='LBFGS_LC_Opt_cifar10', seed=10) 
    # train_LBFGS(exp_name='LBFGS_LC_Opt_cifar10', seed=0) 
    # exit(0)
    
    train_test_model_aug(exp_name='TEST_BUTTA', EPOCHS=400, opt='SGD_opt')
    
    train_test_model_aug(exp_name='SGD_opt', EPOCHS=400, opt='SGD_opt')
    train_test_model_aug(exp_name='SGD_def', EPOCHS=400, opt='SGD_def') 
    # train_test_model_no_aug(exp_name='Adam_opt', EPOCHS=400, opt='Adam') 
    train_test_model_aug(exp_name='Adam_opt', EPOCHS=400, opt='Adam_opt') 
    # train_test_model_no_aug(exp_name='Adamax_opt', EPOCHS=400, opt='Adamax')
    train_test_model_aug(exp_name='Adamax_opt', EPOCHS=400, opt='Adamax_opt')
    # train_test_model_no_aug(exp_name='Nadam_opt', EPOCHS=400, opt='Nadam')
    train_test_model_aug(exp_name='Nadam_opt', EPOCHS=400, opt='Nadam_opt')
    # train_test_model_no_aug(exp_name='RMSprop_opt', EPOCHS=400, opt='RMSprop')
    train_test_model_aug(exp_name='RMSprop_opt', EPOCHS=400, opt='RMSprop_opt')
    train_test_model_aug(exp_name='Adam_def', EPOCHS=400, opt='Adam_def')
    train_test_model_aug(exp_name='Adamax_def', EPOCHS=400, opt='Adamax_def')
    train_test_model_aug(exp_name='Nadam_def', EPOCHS=400, opt='Nadam_def')
    train_test_model_aug(exp_name='RMSprop_def', EPOCHS=400, opt='RMSprop_def')
    
    
    # train_test_model_no_aug(exp_name='Adagrad_opt', EPOCHS=400, opt='Adagrad')
    # train_test_model_aug(exp_name='Adagrad_opt', EPOCHS=400, opt='Adagrad')
    # train_test_model_no_aug(exp_name='FTRL_opt', EPOCHS=400, opt='FTRL')
    # train_test_model_aug(exp_name='FTRL_opt', EPOCHS=400, opt='FTRL')
    # train_test_model_no_aug(exp_name='Adadelta_opt', EPOCHS=400, opt='Adadelta')
    # train_test_model_aug(exp_name='Adadelta_opt', EPOCHS=400, opt='Adadelta')

    print('\n#-------------------#\n# Process completed #\n#-------------------#')
