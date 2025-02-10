import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input,Conv2D,BatchNormalization,Concatenate,Conv2DTranspose,MaxPooling2D,Activation,Layer,UpSampling2D
import os
from scipy.signal import convolve2d
import numpy as np
from glob import glob
import sklearn


from focal_loss import BinaryFocalLoss

class SRMFilterLayer(Layer):
    """Custom layer to apply SRM filtering inside the Keras computation graph."""
    def __init__(self, **kwargs):
        super(SRMFilterLayer, self).__init__(**kwargs)
        self.kernels = self.create_srm_kernels()

    def create_srm_kernels(self):
        """Convert predefined SRM kernels into TensorFlow filters."""
        srm_kernels = [
            (1/4) * np.array([
                [0, 0, 0, 0, 0],
                [0, -1, 2, -1, 0],
                [0, 2, -4, 2, 0],
                [0, -1, 2, -1, 0],
                [0, 0, 0, 0, 0]
            ]),
            (1/12) * np.array([
                [-1, 2, -2, 2, -1],
                [2, -6, 8, -6, 2],
                [-2, 8, -12, 8, -2],
                [2, -6, 8, -6, 2],
                [-1, 2, -2, 2, -1]
            ]),
            (1/2) * np.array([
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, -2, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ])
        ]

        srm_kernels = np.stack(srm_kernels, axis=-1)  # Shape: (5, 5, 1, 3)
        srm_kernels = np.expand_dims(srm_kernels, axis=-2)  # Shape: (5, 5, 3, 3)
        return tf.constant(srm_kernels, dtype=tf.float32)

    def call(self, inputs):
        """Apply SRM filtering using depth-wise convolution."""
        inputs = tf.image.rgb_to_grayscale(inputs)  # Convert to grayscale
        filtered = tf.nn.depthwise_conv2d(
            inputs, self.kernels, strides=[1, 1, 1, 1], padding='SAME'
        )
        return filtered

    def compute_output_shape(self, input_shape):
        return input_shape



class InPlaceABN(Layer):
    def __init__(self, activation='relu', momentum=0.99, epsilon=1e-3, **kwargs):

        super(InPlaceABN, self).__init__(**kwargs)
        self.momentum = momentum
        self.epsilon = epsilon
        self.activation = activation

    def build(self, input_shape):
        # Create trainable parameters for batch normalization
        self.gamma = self.add_weight(
            name='gamma', shape=(input_shape[-1],), initializer='ones', trainable=True
        )
        self.beta = self.add_weight(
            name='beta', shape=(input_shape[-1],), initializer='zeros', trainable=True
        )
        self.moving_mean = self.add_weight(
            name='moving_mean', shape=(input_shape[-1],), initializer='zeros', trainable=False
        )
        self.moving_variance = self.add_weight(
            name='moving_variance', shape=(input_shape[-1],), initializer='ones', trainable=False
        )
        super(InPlaceABN, self).build(input_shape)

    def call(self, inputs, training=None):
        # Compute batch statistics if in training mode
        if training:
            batch_mean, batch_var = tf.nn.moments(inputs, axes=[0, 1, 2], keepdims=False)
            # Update moving averages
            self.moving_mean.assign(self.moving_mean * self.momentum + batch_mean * (1 - self.momentum))
            self.moving_variance.assign(self.moving_variance * self.momentum + batch_var * (1 - self.momentum))
        else:
            # Use moving averages during inference
            batch_mean = self.moving_mean
            batch_var = self.moving_variance

        # Apply Batch Normalization
        normalized_inputs = (inputs - batch_mean) / tf.sqrt(batch_var + self.epsilon)
        bn_output = self.gamma * normalized_inputs + self.beta


        return tf.nn.relu(bn_output)


    def compute_output_shape(self, input_shape):
        return input_shape


def dice_loss(y_true, y_pred, epsilon=1e-7):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])

    denominator = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3]) + epsilon

    dice_coefficient = numerator / denominator
    loss = 1 - dice_coefficient
    return tf.reduce_mean(loss)

fl=BinaryFocalLoss(gamma=3)


def fused_loss(y_true, y_pred, kappa=1.0):
    dice = dice_loss(y_true, y_pred)
    focal = fl(y_true, y_pred)
    total_loss = dice + kappa * focal
    return total_loss



def encoder(input_shape,encoder_name):
    encoder_model = tf.keras.applications.DenseNet121(weights="imagenet", include_top=False, input_shape=input_shape)
    encoder_model._name = encoder_name  

    encoder_model.trainable =False

    inputs = tf.keras.Input(shape=input_shape)

    ouput= encoder_model(inputs)

    s1=encoder_model.get_layer('conv1_relu').output   #output for 4th conct operation
    s2=encoder_model.get_layer('pool1').output        #output for 3rd conct operation
    s3=encoder_model.get_layer('pool2_pool').output   #output for 2nd conct operation
    s4=encoder_model.get_layer('pool3_pool').output   #output for 1st conct operation
    s5=encoder_model.get_layer('pool4_pool').output   #ouput of encoder

    print(s1.shape)
    print(s2.shape)
    print(s3.shape)
    print(s4.shape)
    print(s5.shape)

    return Model(inputs=inputs, outputs=[s1,s2,s3,s4,s5],name=encoder_name)


def conv_block(input,num_filters):# yellow part in decoder

    x=Conv2D(num_filters,3,padding="same")(input)
    x=InPlaceABN(activation='relu')(x)
    x=Conv2D(num_filters,3,padding="same")(x)
    x=InPlaceABN(activation='relu')(x)
    return x

def decoder_block(input,skip_features,num_filters):# blue part in decoder

    x=Conv2DTranspose(num_filters,(2,2),strides=2,padding="same")(input)
    print(x.shape)
    x=Concatenate()([x,skip_features])
    x = Conv2D(num_filters, 3, padding="same")(x)
    x = InPlaceABN(activation='relu')(x)
    return x


def decoder(input_shape, skip_connections, num_filters_list):
    input_shapes = [tuple(layer.shape[1:]) for layer in skip_connections]  # Extract shapes safely

    # Create input layers for each skip connection
    inputs = [Input(shape=shape) for shape in input_shapes]    
    rgb_noise_input = Input(shape=(256, 256, 3))

    x = inputs[-1]
    s1,s2,s3,s4,s5=skip_connections   #rgb+noise output
    
    x=decoder_block(x,s4,num_filters_list[0])
    x=conv_block(x,num_filters_list[0])

    x=decoder_block(x,s3,num_filters_list[1])
    x=conv_block(x,num_filters_list[1])

    x=decoder_block(x,s2,num_filters_list[2])
    x=conv_block(x,num_filters_list[2])

    x=decoder_block(x,s1,num_filters_list[3])
    x=conv_block(x,num_filters_list[3])

    x=decoder_block(x,rgb_noise_input,num_filters_list[4])
    x=conv_block(x,num_filters_list[4])

    x = Conv2D(filters=1, kernel_size=(1, 1), activation=None)(x)
    output = Activation('sigmoid')(x)

    print(f"final output of the model=>{output.shape}")

    return Model(inputs=[inputs,rgb_noise_input], outputs=output,name="decoder")


def print_shape(string,list):
    for x in list:
        print(f"shape of {string}=>{x.shape}")

def DS_UNet(input_shape=(256, 256, 3), num_filters_list=[256, 512, 256, 64, 32]):

    # RGB Stream Encoder
    rgb_input = Input(shape=input_shape, name="rgb_input")
    rgb_encoder = encoder(input_shape,"rgb_encoder")
    rgb_encoder._name = "rgb_encoder"  
    rgb_features = rgb_encoder(rgb_input)

    print_shape("rgb stream",rgb_features)

    # Noise Stream Encoder with SRM Filtering
    noise_input = Input(shape=input_shape, name="noise_input")
    noise_filtered = SRMFilterLayer()(noise_input)  # Use the custom SRM layer

    noise_encoder = encoder(input_shape,"noise_encoder")
    noise_encoder._name = "noise_encoder"  

    noise_features = noise_encoder(noise_filtered)

    print_shape("noiose stream",noise_features)

    fused_features=[]
    for index, layer in enumerate(noise_features):
        fused_features.append(tf.keras.layers.Add(name=f"fusion{index}")([rgb_features[index], noise_features[index]]))
        print(f"Fusion at layer {index}: {fused_features[index].shape}")

    print_shape("fused",fused_features)

    noise_filtered = Conv2D(3, (1,1), padding="same", activation=None)(noise_filtered) # noise filter should have same channels as rgb image for elemeent wise addition 

    rgb_noise = tf.keras.layers.Add()([rgb_input, noise_filtered]) # rgb image and noise additn

    decoder_model = decoder(
    input_shape=fused_features[-1].shape[1:], 
    skip_connections=fused_features, 
    num_filters_list=num_filters_list, 
    )    

    decoder_output = decoder_model([*fused_features, rgb_noise])

    # Model Definition
    model = Model(inputs=[rgb_input, noise_input], outputs=decoder_output, name="DS_UNet")
    
    return model


model=DS_UNet()
model.summary()


print("done hai bhaiya ")