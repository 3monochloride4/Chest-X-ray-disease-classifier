from transfer_learning import *
from mobilenetv2 import *
from efficientnetb0 import *
from utils import create_df, apply_enhancement

import os
import math
import numpy as np
import pickle

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img,img_to_array,ImageDataGenerator
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.activations import swish
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold

augment_datagen = ImageDataGenerator(
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    rescale=1./255
)
default_datagen = ImageDataGenerator(
    rescale=1./255
)

classes_name = ["normal", "tuberculosis", "other_lung_disease"]

def get_labels_from_generator(generator):
    num_of_data = len(generator.filenames)
    batch_size = generator.batch_size
    num_of_batches = math.ceil(num_of_data / (1.0 * batch_size))
    labels = []
    for i in range(0, int(num_of_batches)):
        labels.extend(np.argmax(generator[i][1], axis=1))
    labels = np.array(labels)
    return labels

def get_images_from_generator(generator):
    num_of_data = len(generator.filenames)
    batch_size = generator.batch_size
    num_of_batches = math.ceil(num_of_data / (1.0 * batch_size))
    images = []
    for i in range(0, int(num_of_batches)):
        images.extend(generator[i][0])
    images = np.array(images)
    print(images.shape)
    return images

def train_eval_function(base_model,
                        dataset,
                        batch_size,
                        epochs,
                        loss_function,
                        optimizer,
                        learning_rate,
                        num_class,
                        input_size,
                        transfer_learning,
                        variables,
                        weight_path,
                        metrics_path):

    num_split = 5
    skfold = StratifiedKFold(num_split, shuffle=True)
    fold_accuracy = []
    fold_f1_score = []
    fold_loss = []

    if dataset.lower() == "clahe":
        _, _, files = os.walk("x-ray dataset_CLAHE", topdown=True)
        if len(files) == 0:
            apply_enhancement(algorithm="CLAHE", image_source_dir="x-ray dataset")
        dataset_dataframe = create_df(dataset_dir="x-ray dataset_CLAHE")
    elif dataset.lower() == "um":
        _, _, files = os.walk("x-ray dataset_UM", topdown=True)
        if len(files) == 0:
            apply_enhancement(algorithm="UM", image_source_dir="x-ray dataset")
        dataset_dataframe = create_df(dataset_dir="x-ray dataset_UM")
    elif dataset.lower() == "hef":
        _, _, files = os.walk("x-ray dataset_HEF", topdown=True)
        if len(files) == 0:
            apply_enhancement(algorithm="HEF", image_source_dir="x-ray dataset")
        dataset_dataframe = create_df(dataset_dir="x-ray dataset_HEF")
    elif dataset.lower() == "original":
        dataset_dataframe = create_df(dataset_dir="x-ray dataset")

    print(f"weight path : {weight_path}")
    print(f"metrics path: {metrics_path}")

    fold_n = 1
    for train_idx, test_idx in skfold.split(dataset_dataframe["images_path"], dataset_dataframe["labels"]):
        train_df = dataset_dataframe.iloc[train_idx]
        val_df = dataset_dataframe.iloc[test_idx]

        train_data = augment_datagen.flow_from_dataframe(train_df,
                                                         x_col="images_path",
                                                         y_col="labels",
                                                         shuffle=True,
                                                         batch_size=batch_size,
                                                         class_mode="categorical" if num_class>2 else "binary",
                                                         classes=classes_name,
                                                         color_mode='rgb',
                                                         target_size=(224, 224))

        val_data = default_datagen.flow_from_dataframe(val_df,
                                                       x_col="images_path",
                                                       y_col="labels",
                                                       shuffle=True,
                                                       batch_size=1,
                                                       class_mode="categorical" if num_class>2 else "binary",
                                                       classes=classes_name,
                                                       color_mode="rgb",
                                                       target_size=(224, 224))


        # initialize the complete model
        if base_model == "MobileNetV2":
            MobileNetV2 = MobileNetv2(input_shape=(input_size, input_size, 3))
            if transfer_learning.lower() == "yes":
                download_tl_weight(model_name="mobilenetv2")
                MobileNetV2.load_weights("MobileNetV2_pretrain_weight.hdf5")
                MobileNetV2.trainable = False
            base = MobileNetV2
            # build complete MobileNetV2 model with transfer learning
            inputs = base.input
            if transfer_learning.lower() == "yes":
                x = base(inputs, training=False)
            else:
                x = base(inputs)
            x = GlobalAveragePooling2D()(x)
            x = Dropout(.5)(x)
            outputs = Dense(num_class, activation="softmax")(x)
            model = Model(inputs, outputs)

        elif base_model == "EfficientNet B0":
            EfficientNet_B0 = EfficientNet(model_type="B0", input_shape=(224, 224, 3),
                                           drop_connect_rate=0.2)
            if transfer_learning.lower() == "yes":
                download_tl_weight(model_name="efficientnetb0")
                EfficientNet_B0.load_weights("EfficientNetB0_pretrain_weight.hdf5")
                EfficientNet_B0.trainable = False
            base = EfficientNet_B0
            # build complete EfficientNet_B0 model with transfer learning
            inputs = base.input
            if transfer_learning.lower() == "yes":
                x = base(inputs, training=False)
            else:
                x = base(inputs)
            x = GlobalAveragePooling2D()(x)
            x = Dropout(.5)(x)
            outputs = Dense(num_class, activation="softmax")(x)
            model = Model(inputs, outputs)

        if optimizer == "Adam":
            opt = tf.keras.optimizers.Adam(lr=learning_rate)
        elif optimizer == "SGD":
            opt = tf.keras.optimizers.SGD(lr=learning_rate)

        model.compile(optimizer=opt,
                      loss='categorical_crossentropy' if loss_function=="Categorical Cross Entropy" else "binary_crossentropy",
                      metrics=['accuracy'])

        print(f"=============================Fold no. {fold_n}=============================")
        STEP_TRAIN = train_data.n // train_data.batch_size
        STEP_VAL = val_data.n // val_data.batch_size
        history = model.fit(
            train_data,
            steps_per_epoch=STEP_TRAIN,
            epochs=epochs,
            validation_data=val_data,
            validation_steps=STEP_VAL,
            verbose=1,
            # callbacks=[lr_scheduler, early_stop]
        )
        result = model.evaluate(val_data)
        if base_model == "MobileNetV2":
            model.save(f"{weight_path}/{dataset.lower()}_dataset/mobilenetv2/{dataset.lower()}_mobilenetv2_fold_{fold_n}.hdf5")
        elif base_model == "EfficientNet B0":
            model.save(f"{weight_path}/{dataset.lower()}_dataset/efficientnet_b0/{dataset.lower()}_efficientnet_b0_fold_{fold_n}.hdf5")
        fold_accuracy.append(result[1])
        fold_loss.append(result[0])

        val_images = get_images_from_generator(val_data)
        y_pred = model.predict(val_images, steps=STEP_VAL)
        y_pred = np.argmax(y_pred, axis=1)
        y_true = get_labels_from_generator(val_data)
        f1_s = f1_score(y_true, y_pred, average="macro")
        fold_f1_score.append(f1_s)
        variables[0].set(f"fold {fold_n} selesai")
        variables[1].set(f"akurasi: {fold_accuracy[fold_n - 1]}| loss: {fold_loss[fold_n - 1]}| f1_score: {fold_f1_score[fold_n - 1]}")
        fold_n += 1
        print("")

    if base_model == "MobileNetV2":
        with open(f"{metrics_path}/{dataset.lower()}_dataset/mobilenetv2/ex1_{dataset.lower()}_accuracy.pkl", "wb") as file:
            pickle.dump(fold_accuracy, file)
        with open(f"{metrics_path}/{dataset.lower()}_dataset/mobilenetv2/ex1_{dataset.lower()}_f1_score.pkl", "wb") as file:
            pickle.dump(fold_f1_score, file)
    elif base_model == "EfficientNet B0":
        with open(f"{metrics_path}/{dataset.lower()}_dataset/efficientnet_b0/ex1_{dataset.lower()}_accuracy.pkl", "wb") as file:
            pickle.dump(fold_accuracy, file)
        with open(f"{metrics_path}/{dataset.lower()}_dataset/efficientnet_b0/ex1_{dataset.lower()}_f1_score.pkl", "wb") as file:
            pickle.dump(fold_f1_score, file)

    return model, fold_accuracy, fold_f1_score, fold_loss