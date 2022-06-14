from tensorflow import keras
from keras import applications
from keras.models import Model
from keras.layers import Dense
import os


def get_model():
    base_model = getattr(applications,"EfficientNetB3")(
        include_top=False,
        input_shape=(224, 224, 3),
        pooling="avg"
    )
    features = base_model.output
    pred_gender = Dense(units=2, activation="softmax", name="pred_gender")(features)
    pred_age = Dense(units=101, activation="softmax", name="pred_age")(features)
    model = Model(inputs=base_model.input, outputs=[pred_gender, pred_age])
    model.load_weights("multitask_agegender/mtask-efficientnet-ckpt/EfficientNetB3_224_weights.11-3.44.hdf5")
    return model


