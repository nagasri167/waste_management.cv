import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

train = "D:\DATASET\TRAIN"
test ="D:\DATASET\TEST"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
test_data = test_datagen.flow_from_directory(
    test,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

bmodel = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
bmodel.trainable = False

x = bmodel.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(128, activation='relu')(x)
output = Dense(3, activation='softmax')(x)

model = Model(inputs=bmodel.input, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# ✅ Fixed indentation here:
model.fit(
    train_data,
    validation_data=test_data,
    epochs=1,
    callbacks=[es]
)

bmodel.trainable = True
for layer in bmodel.layers[:-31]:
    layer.trainable = False

model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, validation_data=test_data, epochs=1, callbacks=[es])

model.save('waste_management.h5')