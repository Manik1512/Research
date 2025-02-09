import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input,Conv2D,BatchNormalization,Concatenate,Conv2DTranspose,MaxPooling2D,Activation,Layer,UpSampling2D
import os
from glob import glob
import sklearn


from focal_loss import BinaryFocalLoss



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


# import tensorflow as tf

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


def encoder(input_shape):
    encoder_model = tf.keras.applications.DenseNet121(weights="imagenet", include_top=False, input_shape=input_shape)
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

    return Model(inputs=inputs, outputs=[s1,s2,s3,s4,s5],name="encoder")
    #return Model(inputs=inputs, outputs=transition_block3)

#pool2_relu->1st dense block end  (None, 128, 128, 256)
#pool3_relu  (None, 64, 64, 512)
#poo4_relu  (32,32,1024)
#relu



encoder_part=encoder((256,256,3))
s1,s2,s3,s4,s5=encoder_part.outputs   #num_filters_list=[256,512,256,64,32]




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
    inputs = Input(shape=input_shape)
    x = inputs
    s1,s2,s3,s4,s5=skip_connections

    x=decoder_block(x,s4,num_filters_list[0])
    x=conv_block(x,num_filters_list[0])

    x=decoder_block(x,s3,num_filters_list[1])
    x=conv_block(x,num_filters_list[1])

    x=decoder_block(x,s2,num_filters_list[2])
    x=conv_block(x,num_filters_list[2])

    x=decoder_block(x,s1,num_filters_list[3])
    x=conv_block(x,num_filters_list[3])

    # x=decoder_block(x,s1,num_filters_list[4])
    # x=conv_block(x,num_filters_list[4])

    x = Conv2D(filters=1, kernel_size=(1, 1), activation=None)(x)
    output = Activation('sigmoid')(x)


    return Model(inputs=inputs, outputs=output,name="decoder")



decoder_part = decoder(input_shape=(8,8,512), skip_connections=encoder_part.outputs, num_filters_list=[256,512,256,64,32])
# decoder_model.summary()


def DSUNET(encoder, decoder):
    inputs = encoder.input
    encoded = encoder(inputs)
    outputs = decoder(encoded)
    autoencoder = Model(inputs, outputs, name="DSUNET")
    return autoencoder

print("done")