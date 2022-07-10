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

def expansion_block(x, t, filters, block_id):
    prefix = f"block_{block_id}_"
    total_filters = t*filters # t = expansion factor
    x = Conv2D(total_filters, 1, padding="same", use_bias=False, name=prefix+"expand")(x)

    x = BatchNormalization(name=prefix + "expand_bn")(x)
    x = ReLU(6, name = prefix + "expand_relu")(x)
    return x

def dwise_block(x, stride, block_id):
    prefix = f"block_{block_id}_"
    x = DepthwiseConv2D(3, strides=(stride, stride), padding="same", use_bias=False, name=prefix+"depthwise_conv")(x)

    x = BatchNormalization(name=prefix + "dw_bn")(x)
    x = ReLU(6, name = prefix + "dw_relu")(x)
    return x
    #depthwise_block(x, stride=2)

def projection_block(x, out_channels, block_id):
    prefix = f"block_{block_id}_"
    x = Conv2D(filters=out_channels, kernel_size=1, padding="same", use_bias=False, name=prefix+"compress")(x)

    x = BatchNormalization(name=prefix + "compress_bn")(x)
    return x

def Bottleneck(x, t, filters, out_channels, stride, block_id):
    y = expansion_block(x, t, filters, block_id)
    y = dwise_block(y, stride, block_id)
    y = projection_block(y, out_channels, block_id)
    if y.shape[-1] == x.shape[-1]:
        y = add([x, y])
    return y

def MobileNetv2(input_shape=(224, 224, 3), n_classes=1000):
    #input
    input = Input(input_shape)
    x = Conv2D(32, 3, strides=(2,2), padding="same", use_bias=False)(input)
    x = BatchNormalization(name="conv1_bn")(x)
    x =  ReLU(6, name="conv1_relu")(x)

    # 17 Bottlenecks
    x = dwise_block(x,stride=1,block_id=1)

    x = projection_block(x, out_channels=16,block_id=1)
    #(w, h, channel)
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 24, stride = 2,block_id = 2)

    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 24, stride = 1,block_id = 3)

    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 32, stride = 2,block_id = 4)

    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 32, stride = 1,block_id = 5)

    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 32, stride = 1,block_id = 6)

    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 64, stride = 2,block_id = 7)

    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 64, stride = 1,block_id = 8)

    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 64, stride = 1,block_id = 9)

    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 64, stride = 1,block_id = 10)

    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 96, stride = 1,block_id = 11)

    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 96, stride = 1,block_id = 12)

    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 96, stride = 1,block_id = 13)

    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 160, stride = 2,block_id = 14)

    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 160, stride = 1,block_id = 15)

    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 160, stride = 1,block_id = 16)

    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 320, stride = 1,block_id = 17)

    #conv2d
    x = Conv2D(filters = 1280, kernel_size=1, padding="same", use_bias=False, name="last_conv")(x)
    x = BatchNormalization(name="last_bn")(x)
    output = ReLU(6, name="last_relu")(x)

    #x = GlobalAveragePooling2D(name="global_average_pool")(x)
    #output = Dense(n_classes, activation="softmax")(x)

    model = Model(input, output)
    return model