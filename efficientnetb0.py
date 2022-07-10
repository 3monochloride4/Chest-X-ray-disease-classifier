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

models = ("B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7")
compound_scaling = {} ## dict containing compound scaling for each model (B0~B7) with format -> depth, width, resolution coef.
for phi, model in enumerate(models):
    compound_scaling[model] = (round(1.2 ** phi, 1), round(1.1 ** phi, 1), round(1.15 ** phi, 1))
compound_scaling

MBCONV_default_config = [  ## block config for MBCONV
    {  # stage 2
        "filters_in": 32,
        "filters_out": 16,
        "expand_ratio": 1,
        "kernel_size": 3,
        "stride": 1,
        "repeat": 1,
        "se_ratio": 0.25
        # "padding": "same"
    },
    {  # stage 3
        "filters_in": 16,
        "filters_out": 24,
        "expand_ratio": 6,
        "kernel_size": 3,
        "stride": 2,
        "repeat": 2,
        "se_ratio": 0.25
        # "padding": "same"
    },
    {  # stage 4
        "filters_in": 24,
        "filters_out": 40,
        "expand_ratio": 6,
        "kernel_size": 5,
        "stride": 2,
        "repeat": 2,
        "se_ratio": 0.25
        # "padding": "same"
    },
    {  # stage 5
        "filters_in": 40,
        "filters_out": 80,
        "expand_ratio": 6,
        "kernel_size": 3,
        "stride": 2,
        "repeat": 3,
        "se_ratio": 0.25
        # "padding": "same"
    },
    {  # stage 6
        "filters_in": 80,
        "filters_out": 112,
        "expand_ratio": 6,
        "kernel_size": 5,
        "stride": 1,
        "repeat": 3,
        "se_ratio": 0.25
        # "padding": "same"
    },
    {  # stage 7
        "filters_in": 112,
        "filters_out": 192,
        "expand_ratio": 6,
        "kernel_size": 5,
        "stride": 2,
        "repeat": 4,
        "se_ratio": 0.25
        # "padding": "same"
    },
    {  # stage 8
        "filters_in": 192,
        "filters_out": 320,
        "expand_ratio": 6,
        "kernel_size": 3,
        "stride": 1,
        "repeat": 1,
        "se_ratio": 0.25
        # "padding": "same"
    },

]


def round_width(filters, width_coefficient, divisor=8):
    ## to get integer value of number of channel
    filters = filters * width_coefficient
    new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def round_depth(num_repeats, depth_coefficient):
    ## to get integer value of number of layer repeats
    new_num_repeats = int(math.ceil(num_repeats * depth_coefficient))
    return new_num_repeats


def expansion_block(x, t, filters_in, block_id):
    prefix = f"block_{block_id}_"
    total_filters = t * filters_in  # t = expansion factor
    if t != 1:
        x = Conv2D(total_filters, 1, padding="same", use_bias=False, name=prefix + "expand")(x)

        x = BatchNormalization(name=prefix + "expand_bn")(x)
        x = Activation(swish, name=prefix + "expand_relu")(x)
    return x


def depthwise_block(x, kernel_size, stride, block_id):
    prefix = f"block_{block_id}_"
    if stride == 2:
        added_padding = kernel_size // stride
        x = ZeroPadding2D(padding=added_padding, name=prefix + "dw_pad")(x)
    x = DepthwiseConv2D(
        kernel_size,
        strides=(stride, stride),
        padding="same" if stride == 1 else "valid",
        use_bias=False,
        name=prefix + "depthwise_conv")(x)

    x = BatchNormalization(name=prefix + "dw_bn")(x)
    x = Activation(swish, name=prefix + "dw_relu")(x)
    return x


def squeeze_and_excitation(x, t, se_ratio, filters_in, block_id):
    prefix = f"block_{block_id}_"
    total_filters = t * filters_in
    if 0 < se_ratio <= 1:
        filters_se = max(1, int(filters_in * se_ratio))
        se = GlobalAveragePooling2D(name=prefix + "se_squeeze")(x)
        se = Reshape((1, 1, total_filters), name=prefix + "se_reshape")(se)
        se = Conv2D(
            filters_se,
            kernel_size=1,
            strides=(1, 1),
            padding="same",
            activation="swish",
            name=prefix + "se_reduce")(se)
        se = Conv2D(
            total_filters,
            kernel_size=1,
            strides=(1, 1),
            padding="same",
            activation="sigmoid",
            name=prefix + "se_expand")(se)
        x = multiply([x, se], name=prefix + "se_excite")
    return x


def projection_block(x, out_channels, block_id):
    prefix = f"block_{block_id}_"
    x = Conv2D(filters=out_channels, kernel_size=1, padding="same", use_bias=False, name=prefix + "project_conv")(x)

    x = BatchNormalization(name=prefix + "project_bn")(x)
    return x


def Inverted_Residual(
        x,
        t,
        filters_in,
        out_channels,
        kernel_size,
        stride,
        block_id,
        se_ratio=0.25,
        dropout_ratio=0):
    prefix = f"block_{block_id}_"
    y = expansion_block(x, t, filters_in, block_id)
    y = depthwise_block(y, kernel_size, stride, block_id)
    y = squeeze_and_excitation(y, t, se_ratio, filters_in, block_id)
    y = projection_block(y, out_channels, block_id)
    if y.shape[-1] == x.shape[-1] and stride == 1:
        if dropout_ratio > 0:
            y = Dropout(dropout_ratio)(y)
        y = add([x, y], name=prefix + "inverted_res")
    return y


def EfficientNet(model_type="B0", input_shape=(224, 224, 3), n_classes=1000, drop_connect_rate=0.2):
    if model_type in compound_scaling:
        depth_coefficient = compound_scaling[model_type][0]
        width_coefficient = compound_scaling[model_type][1]
        size = int(224 * compound_scaling[model_type][2])
        w = int(input_shape[1] * compound_scaling[model_type][2])
        h = int(input_shape[0] * compound_scaling[model_type][2])
        in_shape = (h, w, input_shape[2])
    # input
    divisor = 8

    input = Input(in_shape)
    x = Normalization(axis=3)(input)
    x = ZeroPadding2D(padding=1)(x)
    x = Conv2D(
        round_width(32, width_coefficient, divisor),
        3,
        strides=(2, 2),
        padding="valid",
        use_bias=False)(x)
    x = BatchNormalization(name="conv1_bn")(x)
    x = Activation(swish, name="conv1_swish")(x)

    # MBCONV BLOCK
    b = 0
    blocks = float(sum(round_depth(mbconv["repeat"], depth_coefficient) for mbconv in MBCONV_default_config))
    for mbconv in MBCONV_default_config:
        filters_in = round_width(mbconv["filters_in"], width_coefficient, divisor)
        out_channels = round_width(mbconv["filters_out"], width_coefficient, divisor)
        kernel_size = mbconv['kernel_size']
        t = mbconv['expand_ratio']
        stride = mbconv["stride"]
        se_ratio = mbconv['se_ratio']
        repeat = round_depth(mbconv['repeat'], depth_coefficient)

        for i in range(repeat):
            if i > 0:
                filters_in = out_channels
                stride = 1
            x = Inverted_Residual(
                x,
                t,
                filters_in,
                out_channels,
                kernel_size,
                stride,
                block_id=b,
                dropout_ratio=drop_connect_rate * b / blocks)
            b += 1

    # conv2d
    x = Conv2D(
        round_width(1280, width_coefficient, divisor),
        kernel_size=1,
        padding="same",
        use_bias=False,
        name="last_conv")(x)
    x = BatchNormalization(name="last_bn")(x)
    output = Activation(swish, name="last_swish")(x)

    # x = GlobalAveragePooling2D(name="global_average_pool")(x)
    # output = Dense(n_classes, activation="softmax")(x)

    model = Model(input, output)
    return model