import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Dropout, BatchNormalization, Reshape, LeakyReLU, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# MNIST dataset loading and preprocessing
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1) / 255

k_size = (3, 3)

# Encoder
encoder_input = Input(shape=(28, 28, 1))

# 28 * 28
x = Conv2D(32, k_size, padding='same')(encoder_input)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

# 28 * 28 -> 14 * 14
x = Conv2D(64, k_size, strides=2, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

# 14 * 14 -> 7 * 7
x = Conv2D(64, k_size, strides=2, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

# 7 * 7
x = Conv2D(64, k_size, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

x = Flatten()(x)

encoder_output = Dense(2)(x)
encoder = Model(encoder_input, encoder_output)

# Decoder
# Input은 Encoder의 output 형태로 들어가야함
decoder_input = Input(shape=(2, ))

# Encoder와 거울처럼 반대되도록 쌓아준다
x = Dense(7 * 7 * 64)(decoder_input)
x = Reshape((7, 7, 64))(x)

# 7 * 7 -> 7 * 7 conv2D <-> conv2DTrans-
x = Conv2DTranspose(64, k_size, strides=1, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

# 7 * 7 -> 14 * 14
x = Conv2DTranspose(64, k_size, strides=2, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

# 14 * 14 -> 28 * 28
x = Conv2DTranspose(64, k_size, strides=2, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

# 28 * 28
x = Conv2DTranspose(32, k_size, strides=1, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

decoder_output = Conv2DTranspose(1, 3, strides=1, padding='same', activation='sigmoid')(x)
decoder = Model(decoder_input, decoder_output)

encoder_shape = encoder_input
x = encoder(encoder_shape)
decoder_out = decoder(x)

ae = Model(encoder_shape, decoder_out)

ae.compile(optimizer='adam',
           loss='MSE',
           metrics=['accuracy'])


modelpath= './simpleAE-MNIST_{epoch:03d}.hdf5'
checkpoint = ModelCheckpoint(modelpath, 
                             monitor='val_accuracy',
                             verbose=0, 
                             save_best_only=True)

ae.fit(x_train, x_train, 
       epochs=10,
       batch_size=32, 
       verbose=2,
       callbacks=[checkpoint])

result = ae.predict(x_train)

#fig는 전체 plot, ax는 하나하나의 개체
fig, ax = plt.subplots(3, 3)
fig.set_size_inches(9, 6)
for i in range(9):
    ax[i//3, i%3].imshow(x_train[i].reshape(28,28), cmap='gray')
    ax[i//3, i%3].axis('off')
plt.tight_layout()
plt.title('original')
plt.show()

fig, ax = plt.subplots(3, 3)
fig.set_size_inches(9, 6)
for i in range(9):
    ax[i//3, i%3].imshow(result[i].reshape(28,28), cmap='gray')
    ax[i//3, i%3].axis('off')
plt.tight_layout()
plt.title('AE')
plt.show()
