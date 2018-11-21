
import warnings

import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10
from sklearn.metrics import accuracy_score

from dbn.tensorflow import SupervisedDBNClassification

num_classes = 3
use_color = True
use_all = False
train_ex = 300
test_ex = 100
batch_size = 32
n_epochs_rbm = 100
n_iter_backprop = 1000
learning_rate_rbm = 0.1
learning_rate = 0.1
img_shape = (32, 32, 3)
if not use_color:
    img_shape = (32, 32)


def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def get_image(label, x, y):
    for i in range(0, len(y)):
        if y[i] == label:
            return x[i]
    return None


def image_subset(index, x, y):
    xs = []
    ys = []
    for i in range(len(x)):
        if y[i] < index:
            xs.append(x[i])
            ys.append(y[i])
    return np.array(xs), np.array(ys)


dbn = SupervisedDBNClassification(
    hidden_layers_structure=[1024, 512, 256],
    learning_rate_rbm=learning_rate_rbm,
    learning_rate=learning_rate,
    n_epochs_rbm=n_epochs_rbm,
    n_iter_backprop=n_iter_backprop,
    batch_size=batch_size,
    activation_function='sigmoid',
    dropout_p=0.2
)

(cx_train, cy_train), (cx_test, cy_test) = cifar10.load_data()

cx_train, cy_train = image_subset(num_classes, cx_train, cy_train)
cx_test, cy_test = image_subset(num_classes, cx_test, cy_test)

if use_all:
    train_ex = len(cx_train)
    test_ex = len(cx_test)
print('Using {} training and {} testing'.format(train_ex, test_ex))

if use_color:
    x_train = np.array([x.flatten() / 255 for x in cx_train[:train_ex]])
    x_test = np.array([x.flatten() / 255 for x in cx_test[:test_ex]])
else:
    x_train = np.array([rgb2gray(x).flatten() / 255 for x in cx_train[:train_ex]])
    x_test = np.array([rgb2gray(x).flatten() / 255 for x in cx_test[:test_ex]])
y_train = cy_train[:train_ex].flatten()
y_test = cy_test[:test_ex].flatten()

with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=RuntimeWarning)
    dbn.fit(x_train, y_train)

predictions = list(dbn.predict(x_test))
accuracy = accuracy_score(y_test, predictions)
print('Accuracy: {0}'.format(accuracy))

if not use_color:
    plt.set_cmap('gray')


fig = plt.figure()
for i in range(10):
    subplt = plt.subplot(2, 10, i + 1)
    hot_index = predictions[i]
    subplt.set_title('Act')
    subplt.axis('off')
    act_image = np.reshape(x_test[i], img_shape)
    if use_color:
        subplt.imshow(act_image)
    else:
        subplt.matshow(act_image)
    plt.draw()
    subplt = plt.subplot(2, 10, 10 + i + 1)
    subplt.set_title('Pred')
    subplt.axis('off')
    pred_image = np.reshape(get_image(hot_index, x_train, y_train), img_shape)
    if use_color:
        subplt.imshow(pred_image)
    else:
        subplt.matshow(pred_image)
    plt.draw()
fig.savefig('dbn.png')
plt.show()

