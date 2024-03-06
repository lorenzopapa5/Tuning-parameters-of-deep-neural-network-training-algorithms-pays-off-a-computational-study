import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras.utils import to_categorical
import skimage.transform as st


def get_dataset(dataset_name, path_dts, download=True):
    """
    Load the correct dataset
    """
    # load the training split of the dataset and additional infos
    train_set, info = tfds.load(dataset_name,
                                split='train',
                                data_dir=path_dts,
                                download=download,
                                as_supervised=True,  # automatically separaters inputs and targets
                                with_info=True,
                                shuffle_files=False) # no shuffle for same split

    return train_set, info

"""
def get_cifar10_dataset(seed, path_dts, download=False):
    # load the training split of the dataset and additional infos
    train_ds, test_ds = tfds.load('cifar10',
                                   split=['train','test'],
                                   data_dir=path_dts,
                                   download=download,
                                   as_supervised=True,  # automatically separaters inputs and targets
                                   with_info=False,
                                   shuffle_files=True) # no shuffle for same split

    def normalize_resize(image, label):
    	image = tf.cast(image, tf.float32)
    	image = tf.divide(image, 255)
    	image = tf.image.resize(image, (32, 32))
    	return image, label
    
    train = train_ds.map(normalize_resize).cache().shuffle(seed).batch(64).repeat()
    test = test_ds.map(normalize_resize).cache().batch(64)
    
    return train, test
"""

def reshape(img):
    """
    Reshape the image to 256x256
    """
    return st.resize(np.float32(img), output_shape=(256, 256), anti_aliasing=True)
    # cv2.resize(np.float32(img), dsize=(256, 256), interpolation=cv2.INTER_CUBIC)


def process_dataset(dataset, num_samples, sample_shape, num_classes, tt_split):
    """
    Process the dataset outputting the train/test arrays
    """
    # define the samples and labels
    xs = np.zeros((num_samples, sample_shape[0], sample_shape[1], sample_shape[2]))
    ys = np.zeros((num_samples))
    i = 0
    # populate the samples from the dataset
    for sample in dataset.take(-1):
        x = sample[0]
        y = sample[1]
        # correct sample shape if required
        if x.shape == sample_shape:
            xs[i] = x
        else:
            xs[i] = reshape(x)
        ys[i] = y
        i += 1
    # apply train/test splitting
    split_index = int(num_samples * tt_split)
    x_train = xs[0:split_index]
    y_train = ys[0:split_index]
    x_test = xs[split_index:]
    y_test = ys[split_index:]
    # change to one-hot encoding
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    # normalize samples in [0,1] range
    x_train = x_train.astype('float32')
    x_train /= 255
    x_test = x_test.astype('float32')
    x_test /= 255

    return x_train, y_train, x_test, y_test


