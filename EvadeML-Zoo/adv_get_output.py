from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

from keras import backend as K
from keras.datasets import mnist
from keras.utils import np_utils

import keras
import numpy as np
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# for solving some specific problems, don't care
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


def get_mnist():
    image_size = 28
    num_channels = 1
    num_classes = 10

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_test = X_test.reshape(X_test.shape[0], image_size, image_size, num_channels)
    X_test = X_test.astype('float32')
    X_test /= 255
    Y_test = np_utils.to_categorical(y_test, num_classes)

    X_train = X_train.reshape(X_train.shape[0], image_size, image_size, num_channels)
    X_train = X_train.astype('float32') / 255

    Y_train = np_utils.to_categorical(y_train, num_classes)

    return X_train, Y_train, X_test, Y_test


# Get output of one specific layer
def getlayer_output(l_in, l_out, x, model):
    # get_k_layer_output = K.function([model.layers[l_in].input, K.learning_phase()],[model.layers[l_out].output])
    get_k_layer_output = K.function([model.layers[l_in].input, 0], [model.layers[l_out].output])
    return get_k_layer_output([x])[0]



if __name__ == '__main__':
    # 1. Get adversarial 'mnist' dataset.
    x_adv_fpath = './results/advs_1.npy'
    adv_data = np.load(x_adv_fpath)
    print(adv_data.shape)

    # 2. Load a trained 'MNIST_carlini' model.
    dataset_name = 'MNIST'
    model_name = 'carlini'
    model_weights_fpath = "%s_%s.keras_weights.h5" % (dataset_name, model_name)
    model_weights_fpath = os.path.join('downloads/trained_models', model_weights_fpath)

    from models.carlini_models import carlini_mnist_model
    model = carlini_mnist_model(logits=False, input_range_type=1, pre_filter=lambda x:x)
    model.load_weights(model_weights_fpath)
    model.summary()

    # 3. get output of each layer and store them in npy
    start_layer = 0
    outputs = {}
    for hl_idx in range(start_layer, len(model.layers)):
        n_neurons = model.layers[hl_idx].output_shape[-1]  # neurons in every layer
        print('layer name: ', model.layers[hl_idx].name, '# of neurons: ', n_neurons)

        layer = getlayer_output(0, hl_idx, adv_data, model).copy()
        print('h layer', model.layers[hl_idx].name, layer.shape)

        file_name = './adv_output/{0}_check_values.npy'.format(model.layers[hl_idx].name)
        np.save(file_name, layer)