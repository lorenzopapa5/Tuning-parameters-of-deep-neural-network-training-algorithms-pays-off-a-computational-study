import matplotlib.pyplot as plt
import numpy as np
import os


def plot_samples(x, y, save_path, dataset_labels, samples_to_print=1):

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for j in range(samples_to_print):
        i = np.random.randint(0, len(x))
        plt.suptitle('Label: {} ({})'.format(np.argmax(y[i]), dataset_labels[np.argmax(y[i])]), fontsize=12)
        plt.imshow(x[i, :].astype(np.float32))
        ax = plt.gca()
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        plt.savefig(save_path + '/seq_' + str(i) + '.png')


def plot_aug_samples(x, y, save_path, dataset_labels, datagen, transform_parameters, samples_to_print=1):

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for j in range(samples_to_print):
        i = np.random.randint(0, len(x))
        fig, (a1, a2) = plt.subplots(1, 2)
        fig.suptitle('Label: {} ({})'.format(np.argmax(y[i]), dataset_labels[np.argmax(y[i])]), fontsize=12)
        a1.imshow(x[i].astype(np.float32))
        a1.set_title('original')
        a2.imshow(datagen.apply_transform(x[i], transform_parameters).astype(np.float32))
        a2.set_title('altered')
        a1.axes.xaxis.set_ticks([])
        a1.axes.yaxis.set_ticks([])
        a2.axes.xaxis.set_ticks([])
        a2.axes.yaxis.set_ticks([])
        plt.savefig(save_path + '/seq_aug_' + str(j) + '.png')


def data_augmentation(method):
    if method == 0:
        DA_method_chosen = 'no_data_aug'
        transform_parameters = {
            'theta': 0.0,
            'zoom': 0.0,
            'shear_range': 0.0,
            'flip_horizontal': False,
            'flip_vertical': False,
            'tx': 0.0,
            'ty': 0.0
        }
    elif method == 1:
        DA_method_chosen = 'flips'
        transform_parameters = {
            'theta': 0.0,
            'zoom': 0.0,
            'shear_range': 0.0,
            'flip_horizontal': True,
            'flip_vertical': True,
            'tx': 0.0,
            'ty': 0.0
        }
    else:
        DA_method_chosen = 'data_aug'
        transform_parameters = {
            'theta': 10.0,
            'zoom': 0.1,
            'shear_range': 0.1,
            'flip_horizontal': True,
            'flip_vertical': True,
            'tx': 0.1,
            'ty': 0.1}

    return DA_method_chosen, transform_parameters
