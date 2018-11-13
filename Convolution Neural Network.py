import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential
from keras import backend as K, utils
from keras.callbacks import EarlyStopping
from keras.datasets import cifar10
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.optimizers import rmsprop


def get_image(label, x, y):
    for i in range(0, len(y)):
        if np.argmax(y[i]) == label:
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


batch_size = 32
num_classes = 3
epochs = 50
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train, y_train = image_subset(num_classes, x_train, y_train)
x_test, y_test = image_subset(num_classes, x_test, y_test)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu',
                 padding='same',
                 input_shape=(32, 32, 3)))
model.add(Conv2D(32, (3, 3),
                 activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3),
                 padding='same',
                 activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

opt = rmsprop(lr=0.0001, decay=1e-6)
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True,
          verbose=2)

score = model.evaluate(x_test, y_test)
print('Test loss: {0}'.format(score[0]))
print('Test accuracy: {0}'.format(score[1]))

fig = plt.figure('Predictions on CIFAR', facecolor='gray')

predictions = model.predict(x_test, verbose=0)

for i in range(10):
    subplt = plt.subplot(2, 10, i + 1)
    hot_index = np.argmax(predictions[i])
    subplt.set_title('Act')
    subplt.axis('off')
    subplt.imshow(x_test[i])
    plt.draw()
    subplt = plt.subplot(2, 10, 10 + i + 1)
    subplt.set_title('Pred')
    subplt.axis('off')
    subplt.imshow(get_image(hot_index, x_train, y_train))
    plt.draw()
fig.savefig('cnn.png')
plt.show()

K.clear_session()