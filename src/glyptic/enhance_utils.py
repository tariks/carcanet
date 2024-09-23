import numpy as np
import skimage as sk
from keras.layers import Input, MaxPooling2D
from keras.layers import (
    concatenate,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    ReLU,
)
from keras.models import Model
from keras.optimizers import Adam
from skimage.transform import resize

input_shape = (400, 288)

def get_unet(do=0., activation=ReLU):
    inputs = Input(input_shape + (3,))
    conv1 = Dropout(do)(activation()(Conv2D(32, (3, 3), padding="same")(inputs)))
    conv1 = Dropout(do)(activation()(Conv2D(32, (3, 3), padding="same")(conv1)))
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Dropout(do)(activation()(Conv2D(64, (3, 3), padding="same")(pool1)))
    conv2 = Dropout(do)(activation()(Conv2D(64, (3, 3), padding="same")(conv2)))
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Dropout(do)(activation()(Conv2D(128, (3, 3), padding="same")(pool2)))
    conv3 = Dropout(do)(activation()(Conv2D(128, (3, 3), padding="same")(conv3)))
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Dropout(do)(activation()(Conv2D(256, (3, 3), padding="same")(pool3)))
    conv4 = Dropout(do)(activation()(Conv2D(256, (3, 3), padding="same")(conv4)))
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Dropout(do)(activation()(Conv2D(512, (3, 3), padding="same")(pool4)))
    conv5 = Dropout(do)(activation()(Conv2D(512, (3, 3), padding="same")(conv5)))

    up6 = concatenate(
        [Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(conv5), conv4],
        axis=3,
    )
    conv6 = Dropout(do)(activation()(Conv2D(256, (3, 3), padding="same")(up6)))
    conv6 = Dropout(do)(activation()(Conv2D(256, (3, 3), padding="same")(conv6)))

    up7 = concatenate(
        [Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(conv6), conv3],
        axis=3,
    )
    conv7 = Dropout(do)(activation()(Conv2D(128, (3, 3), padding="same")(up7)))
    conv7 = Dropout(do)(activation()(Conv2D(128, (3, 3), padding="same")(conv7)))

    up8 = concatenate(
        [Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(conv7), conv2],
        axis=3,
    )
    conv8 = Dropout(do)(activation()(Conv2D(64, (3, 3), padding="same")(up8)))
    conv8 = Dropout(do)(activation()(Conv2D(64, (3, 3), padding="same")(conv8)))

    up9 = concatenate(
        [Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(conv8), conv1],
        axis=3,
    )
    conv9 = Dropout(do)(activation()(Conv2D(32, (3, 3), padding="same")(up9)))
    conv9 = Dropout(do)(activation()(Conv2D(32, (3, 3), padding="same")(conv9)))

    conv10 = Dropout(do)(Conv2D(1, (1, 1), activation="sigmoid")(conv9))

    model = Model(inputs=[inputs], outputs=[conv10])
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mean_squared_error")

    model.summary()

    return model

def postprocess(img: np.ndarray):
    img = sk.restoration.denoise_bilateral(img)
    img = sk.exposure.equalize_adapthist(img)
    return img

