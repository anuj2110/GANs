from keras.layers import Dense,BatchNormalization,Input,Flatten
from keras.layers.advanced_activations import LeakyReLU,PReLU
from keras.layers import Activation 
from keras.layers.convolutional import Conv2D,Conv2DTranspose,UpSampling2D
from keras.models import Model 
from keras.layers import add

def residual_blocks(model,kernel_size,filters,strides):
    gen_model = model
    model = Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,padding="same")(model)
    model = BatchNormalization(momentum=0.5)(model)
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)
    
    model = Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,padding="same")(model)
    model = BatchNormalization(momentum=0.5)(model)
     
    model = add([gen_model,model])
    return model

def upsampling(model,kernel_size,filters,strides):
    model = Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,padding="same")(model)
    model = UpSampling2D(size=2)(model)
    model = LeakyReLU(alpha=0.2)(model)

    return model

def disc_conv_block(model,kernel_size,filters,strides):
    model = Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,padding="same")(model)
    model = BatchNormalization(momentum = 0.5)(model)
    model = LeakyReLU(alpha = 0.2)(model)

    return model

class generator(object):
    def __init__(self,noise_shape):
        self.shape=noise_shape
    def generator_model(self):

        gen_input = Input(shape = self.shape)

        model = Conv2D(filters=64,kernel_size=9,strides=1,padding="same")(gen_input)
        model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)

        gen_model = model

        for i in range(16):
            model = residual_blocks(model,3,64,1)
        
        model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(model)
        model = BatchNormalization(momentum = 0.5)(model)
        model = add([gen_model, model])
        
        for i in range(2):
            model = upsampling(model,3,256,1)
        
        model = Conv2D(filters = 3, kernel_size = 9, strides = 1, padding = "same")(model)
        model = Activation('tanh')(model)
           
        generator = Model(inputs = gen_input, outputs = model)
        generator.summary()

        return generator

class discriminator(object):
    def __init__(self,image_shape):
        self.image_shape = image_shape
    
    def discriminator_model(self):
        disc_input = Input(shape = self.image_shape)

        model = Conv2D(filters=64,kernel_size=3,strides=1,padding="same")(disc_input)
        model = LeakyReLU(alpha=0.2)(model)

        model = disc_conv_block(model, 3, 64, 2)
        model = disc_conv_block(model, 3, 128, 1)
        model = disc_conv_block(model, 3, 128, 2)
        model = disc_conv_block(model, 3, 256, 1)
        model = disc_conv_block(model, 3, 256, 2)
        model = disc_conv_block(model, 3, 512, 1)
        model = disc_conv_block(model, 3, 512, 2)

        model = Flatten()(model)

        model = Dense(1024)(model)
        model = LeakyReLU(alpha=0.2)(model)

        model = Dense(1)(model)
        model = Activation('sigmoid')(model) 
        
        discriminator = Model(inputs = disc_input, outputs = model)
        discriminator.summary()
        return discriminator