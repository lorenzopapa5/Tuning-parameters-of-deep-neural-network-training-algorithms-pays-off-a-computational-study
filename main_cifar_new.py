import os, argparse, random
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Model, layers
from keras.callbacks import ModelCheckpoint
from keras.datasets import cifar10, cifar100
from keras.initializers import GlorotUniform
from tensorflow.keras.optimizers import Adam, Adamax, Nadam, RMSprop, SGD, Adagrad, Ftrl, Adadelta
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from psutil import virtual_memory

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def hardware_check():
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        print('GPU not found')
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(virtual_memory().total / 1e9))

def load_data(dataset):
    if dataset == 'cifar10':
        return cifar10.load_data()
    else:
        return cifar100.load_data()

def prepare_data(x_train, x_test):
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    mean = np.mean(x_train)
    std = np.std(x_train)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    return x_train, x_test

def get_optimizer(opt):
    optimizers = {
        'adam': Adam(), #learning_rate=0.001, beta_1=0.9, beta_2=0.9999, epsilon=1e-08, amsgrad=True),
        'adamax': Adamax(), #learning_rate=0.001, beta_1=0.6, beta_2=0.99, epsilon=1e-06),
        'nadam': Nadam(), #learning_rate=0.001, beta_1=0.99, beta_2=0.99, epsilon=1e-06),
        'rmsprop': RMSprop(), #learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-06, centered=False),
        'sgd': SGD(learning_rate=0.01, momentum=0.0, nesterov=False),
        'adagrad': Adagrad(),
        'ftrl': Ftrl(),
        'adadelta': Adadelta()
    }
    return optimizers.get(opt, 'Error')

def build_model(sample_shape, num_classes, seed, model_name):

    if model_name == 'mobilenet':

        base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False, weights='imagenet', input_shape=sample_shape)
        for layer in base_model.layers:
            layer.trainable = True

        flatten = layers.Flatten()(base_model.output)
        fc1 = layers.Dense(512, activation='relu')(flatten)
        fc2 = layers.Dense(256, activation='relu')(fc1)
        fc3 = layers.Dense(num_classes, activation='softmax')(fc2)
        model = Model(inputs=base_model.input, outputs=fc3)

        return model
    
    elif model_name == 'resnet':

        base_model = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=sample_shape)
        for layer in base_model.layers:
            layer.trainable = True

        flatten = layers.Flatten()(base_model.output)
        fc1 = layers.Dense(512, activation='relu')(flatten)
        fc2 = layers.Dense(256, activation='relu')(fc1)
        fc3 = layers.Dense(num_classes, activation='softmax')(fc2)
        model = Model(inputs=base_model.input, outputs=fc3)

        return model
    
    else:

        def _conv_block(input_tensor, filters, kernel, dr_range, pool):
            x = layers.Conv2D(
                filters=filters,
                kernel_size=kernel,
                activation='relu',
                padding='same',
                strides=1,
                kernel_initializer=GlorotUniform(seed=seed),
                input_shape=sample_shape
            )(input_tensor)
            if pool:
                x = layers.MaxPooling2D(pool_size=(2, 2), padding='valid')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(dr_range)(x)
            return x

        def _dense_block(input_tensor, filters, dr_range):
            x = layers.Dense(
                units=filters,
                kernel_initializer=GlorotUniform(seed=seed),
                activation='relu'
            )(input_tensor)
            x = layers.Dropout(dr_range)(x)
            return x
        
        if model_name == 'baseline':

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

            return Model(inputs=inputs, outputs=out)

        elif model_name == 'wide':

            inputs = layers.Input(shape=sample_shape)
            conv0 = _conv_block(inputs, filters=16*2, kernel=(11, 11), dr_range=0.3, pool=True)
            conv1 = _conv_block(conv0, filters=32*2, kernel=(9, 9), dr_range=0.3, pool=True)
            conv2 = _conv_block(conv1, filters=64*2, kernel=(7, 7), dr_range=0.3, pool=True)
            conv3 = _conv_block(conv2, filters=128*2, kernel=(5, 5), dr_range=0.3, pool=True)
            conv4 = _conv_block(conv3, filters=256*2, kernel=(3, 3), dr_range=0.3, pool=True)
            flatten = layers.Flatten()(conv4)
            fc0 = _dense_block(flatten, filters=256, dr_range=0.25)
            fc1 = layers.Dense(num_classes)(fc0)
            out = layers.Activation('softmax')(fc1)

            return Model(inputs=inputs, outputs=out)

        elif model_name == 'deep':

            inputs = layers.Input(shape=sample_shape)
            conv0 = _conv_block(inputs, filters=16, kernel=(11, 11), dr_range=0.3, pool=False)
            conv00 = _conv_block(conv0, filters=16, kernel=(11, 11), dr_range=0.3, pool=True)

            conv1 = _conv_block(conv00, filters=32, kernel=(9, 9), dr_range=0.3, pool=False)
            conv11 = _conv_block(conv1, filters=32, kernel=(9, 9), dr_range=0.3, pool=True)

            conv2 = _conv_block(conv11, filters=64, kernel=(7, 7), dr_range=0.3, pool=False)
            conv22 = _conv_block(conv2, filters=64, kernel=(7, 7), dr_range=0.3, pool=True)

            conv3 = _conv_block(conv22, filters=128, kernel=(5, 5), dr_range=0.3, pool=False)
            conv33 = _conv_block(conv3, filters=128, kernel=(5, 5), dr_range=0.3, pool=True)

            conv4 = _conv_block(conv33, filters=256, kernel=(3, 3), dr_range=0.3, pool=False)
            conv43 = _conv_block(conv4, filters=256, kernel=(3, 3), dr_range=0.3, pool=True)

            flatten = layers.Flatten()(conv43)
            fc0 = _dense_block(flatten, filters=256, dr_range=0.25)
            fc1 = layers.Dense(num_classes)(fc0)
            out = layers.Activation('softmax')(fc1)

            return Model(inputs=inputs, outputs=out)

        elif model_name == 'deepwide':

            inputs = layers.Input(shape=sample_shape)
            conv0 = _conv_block(inputs, filters=16*2, kernel=(11, 11), dr_range=0.3, pool=False)
            conv00 = _conv_block(conv0, filters=16*2, kernel=(11, 11), dr_range=0.3, pool=True)

            conv1 = _conv_block(conv00, filters=32*2, kernel=(9, 9), dr_range=0.3, pool=False)
            conv11 = _conv_block(conv1, filters=32*2, kernel=(9, 9), dr_range=0.3, pool=True)

            conv2 = _conv_block(conv11, filters=64*2, kernel=(7, 7), dr_range=0.3, pool=False)
            conv22 = _conv_block(conv2, filters=64*2, kernel=(7, 7), dr_range=0.3, pool=True)

            conv3 = _conv_block(conv22, filters=128*2, kernel=(5, 5), dr_range=0.3, pool=False)
            conv33 = _conv_block(conv3, filters=128*2, kernel=(5, 5), dr_range=0.3, pool=True)

            conv4 = _conv_block(conv33, filters=256*2, kernel=(3, 3), dr_range=0.3, pool=False)
            conv43 = _conv_block(conv4, filters=256*2, kernel=(3, 3), dr_range=0.3, pool=True)

            flatten = layers.Flatten()(conv43)
            fc0 = _dense_block(flatten, filters=256, dr_range=0.25)
            fc1 = layers.Dense(num_classes)(fc0)
            out = layers.Activation('softmax')(fc1)

            return Model(inputs=inputs, outputs=out)

        else:
            print('ERROR in network selection')
            exit(0)





# --------------------------- TRAIN MODEL ----------------------------------------------------------------------------------------------------------------------------------

def train_test_model(exp_name, epochs, opt, model_name, 
                     x_train, y_train, x_test, y_test, sample_shape, num_classes, seed, datagen=None, batch_size=128):
    run_dir = '/work/project/save_model/' + exp_name + '/'
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    optimizer = get_optimizer(opt)
    if optimizer == 'Error':
        print('Error')
        exit(0)

    model = build_model(sample_shape, num_classes, seed, model_name)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    callbacks = [ModelCheckpoint(run_dir + f'best_model_{"aug" if datagen else "no_aug"}.h5', monitor='val_accuracy', save_best_only=True, verbose=0)]
    print('Running sample training...')
    
    if datagen:
        history = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size), epochs=epochs, validation_data=(x_test, y_test), verbose=1, callbacks=callbacks)
    else:
        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), verbose=1, callbacks=callbacks)
    
    hist_df = pd.DataFrame(history.history)
    hist_csv_file = run_dir + f'history_{"aug" if datagen else "no_aug"}.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

    test_acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    print(test_acc)

if __name__ == '__main__':

    # Setup in cmd line
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True) 
    parser.add_argument('--network', type=str, required=True)
    parser.add_argument('--opt', type=str, required=True)
    parser.add_argument('--dts', type=str, default='cifar10')
    
    args = parser.parse_args()

    print('Tensorflow version: {}'.format(tf.__version__))
    hardware_check()

    # Globals
    img_rows, img_cols, channels = 32, 32, 3
    SEED = args.seed
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # Load data
    dataset = args.dts
    (x_train, y_train), (x_test, y_test) = load_data(dataset)
    
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    datagen.fit(x_train)

    x_train, x_test = prepare_data(x_train, x_test)
    sample_shape = (img_rows, img_cols, channels)

    # Labels
    num_classes = 10 if dataset == 'cifar10' else 100
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    # train_test_model(exp_name=args.opt + '_no_aug_seed_' + str(args.seed), 
    #                  epochs=1, 
    #                  opt=args.opt,
    #                  model_name=args.network,
    #                  x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, sample_shape=sample_shape, num_classes=num_classes, seed=SEED)
    train_test_model(exp_name=args.opt + '_' + args.network + '_dts_' + args.dts + '_aug_seed_' + str(args.seed), 
                     epochs=50, 
                     opt=args.opt,
                     model_name=args.network,
                     x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, sample_shape=sample_shape, num_classes=num_classes, seed=SEED, datagen=datagen)

    print('\n#-------------------#\n# Process completed #\n#-------------------#')
