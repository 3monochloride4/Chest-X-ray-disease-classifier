#for model architecture
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img,img_to_array,ImageDataGenerator
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.activations import swish
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, f1_score, classification_report

#from tensorflow.keras.applications import VGG16, MobileNet
#from tensorflow.keras.applications.vgg16 import preprocess_input

def download_tl_weight(model_name):
    if model_name == "MobileNetV2".lower():
        MobileNetV2_tl = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            alpha=1.0,
            include_top=False,
            weights='imagenet',
            pooling=None
        )
        MobileNetV2_tl.save_weights("MobileNetV2_pretrain_weight.hdf5")
    if model_name == "EfficientNetB0".lower():
        EfficientNetB0_tl = tf.keras.applications.EfficientNetB0(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet',
            pooling=None
        )
        EfficientNetB0_tl.save_weights("EfficientNetB0_pretrain_weight.hdf5")

