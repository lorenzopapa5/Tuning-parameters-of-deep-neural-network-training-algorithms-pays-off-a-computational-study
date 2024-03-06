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
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Ftrl, Adagrad, Adadelta, Adam, Adamax, Nadam, RMSprop, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from dataset import get_dataset, process_dataset
from utils import plot_samples, plot_aug_samples, data_augmentation

'''
From Terminal
    tensorboard dev upload --logdir='/home/lorenzo/Results_OMML/hparams/'
'''

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
    exp_name = 'TEST_BUTTA'
    da_method = 0  # 0 (no_data_aug), 1 (flips)

    # Gloabls
    do_print = False
    download_dts = False
    HP_EPOCHS = 15
    dts_root_path = '/work/dataset/'
    hparam_root_path = '/work/hparams/' + exp_name + '__aug-' + str(da_method) + '/'
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

    VAL_TRAIN_SPLIT = 0.2  # how many samples (in fraction) are reserved for validation

    datagen = ImageDataGenerator(
        rotation_range=transform_parameters['theta'],
        width_shift_range=transform_parameters['tx'],
        height_shift_range=transform_parameters['ty'],
        zoom_range=transform_parameters['zoom'],
        horizontal_flip=transform_parameters['flip_horizontal'],
        vertical_flip=transform_parameters['flip_vertical'],
        shear_range=transform_parameters['shear_range'],
        validation_split=VAL_TRAIN_SPLIT
    )

    datagen.fit(x_train)
    print('Method {} applied for Data Augmentation'.format(da_method_chosen))

    if do_print:
        plot_aug_samples(x_train, y_train, hparam_root_path, dataset_labels, datagen, transform_parameters, samples_to_print=5)

    # Grid search hyperparameters tuning
    log_dir = hparam_root_path

    HP_seed_w = hp.HParam('seeds', hp.Discrete([1699806]))

    METRIC_ACCURACY = 'accuracy'

    with tf.summary.create_file_writer(hparam_root_path + '/logs/hparam_tuning').as_default():
        hp.hparams_config(
            hparams=[HP_seed_w],
            metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')]
        )


    def train_test_model(hparams, epochs, input_shape, num_classes, bs, run_dir):

        from keras.initializers import GlorotUniform, lecun_normal

        def _conv_block(input_tensor, filters, kernel, dr_range, pool):
            x = layers.Conv2D(filters=filters,
                              kernel_size=kernel,
                              # activation='relu',
                              padding='valid',
                              strides=2,
                              kernel_initializer=GlorotUniform(seed=hparams[HP_seed_w]),
                              input_shape=input_shape)(input_tensor)
            x = tf.keras.activations.swish(x)  # A ReLU differentiable in all its points!
            if pool:
                x = layers.MaxPooling2D(pool_size=(2, 2), padding='valid')(x)
                # x = layers.AveragePooling2D(pool_size=(2, 2), padding='valid')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(dr_range)(x)

            return x

        def _dense_block(input_tensor, filters, dr_range):
            x = layers.Dense(units=filters,
                             kernel_initializer=GlorotUniform(seed=hparams[HP_seed_w]),
                             activation='relu')(input_tensor)
            x = layers.Dropout(dr_range)(x)

            return x

        # define model
        inputs = layers.Input(shape=input_shape)
        conv0 = _conv_block(inputs, filters=16, kernel=(11, 11), dr_range=0.3, pool=False)
        conv1 = _conv_block(conv0, filters=32, kernel=(9, 9), dr_range=0.3, pool=False)
        conv2 = _conv_block(conv1, filters=64, kernel=(7, 7), dr_range=0.3, pool=False)
        conv3 = _conv_block(conv2, filters=128, kernel=(5, 5), dr_range=0.3, pool=False)
        conv4 = _conv_block(conv3, filters=256, kernel=(3, 3), dr_range=0.3, pool=False)
        flatten = layers.Flatten()(conv4)
        fc0 = _dense_block(flatten, filters=512, dr_range=0.25)
        fc1 = layers.Dense(num_classes)(fc0)
        out = layers.Activation('softmax')(fc1)

        model = Model(inputs=inputs, outputs=out)
        model.summary()

        # from bfgs import bfgs_train
        # bfgs_train(model=model, train_x=x_train, train_y=y_train, test_x=x_test, test_y=y_test)
        # exit(0)

        # optimizer
        opt = Adadelta()
        # compile
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        # callback
        callbacks = [TensorBoard(log_dir=log_dir, update_freq='epoch', histogram_freq=2),
                     ModelCheckpoint(run_dir + '_best_model.h5', monitor='val_accuracy', save_best_only=True, verbose=0),
                     hp.KerasCallback(log_dir, hparams)]
        # run sample training
        print('Running sample training...')
        history = model.fit(datagen.flow(x_train, y_train, batch_size=bs, subset="training"),
                            epochs=epochs,
                            validation_data=datagen.flow(x_train, y_train, batch_size=bs, subset="validation"),
                            callbacks=callbacks,
                            verbose=1)

        # save history to csv:
        hist_df = pd.DataFrame(history.history)
        hist_csv_file = run_dir + '_history.csv'
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)

        model = load_model(run_dir + '_best_model.h5', compile=True)

        # obtain final accuracy
        _, accuracy = model.evaluate(x_test, y_test)

        return accuracy


    def run(run_dir, hparams):
        with tf.summary.create_file_writer(run_dir).as_default():
            hp.hparams(hparams)  # record the values used in this trial
            accuracy = train_test_model(hparams, HP_EPOCHS, sample_shape, num_classes, bs=32, run_dir=run_dir)
            tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)


    # run tensorboard
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', log_dir, '--host', '0.0.0.0', '--load_fast', 'false'])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")

    # Run the grid search
    session_num = 0

    for seeds in HP_seed_w.domain.values:
        hparams = {
            HP_seed_w: seeds
        }
        run_name = 'run-{}'.format(session_num)
        print('### Starting trial: {} ###'.format(run_name))
        print({h.name: hparams[h] for h in hparams})
        run(log_dir + '/' + run_name, hparams)
        session_num += 1

    print('\n#-------------------#\n# Process completed #\n#-------------------#')
