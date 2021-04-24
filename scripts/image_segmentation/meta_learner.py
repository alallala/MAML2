"""
    Date: Feb 11st 2020
    Author: Hilbert XU
    Abstract: MetaLeaner model
"""
from task_generator import TaskGenerator

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as keras_backend
import os
import numpy as np 
import cv2

from seg_commonblocks import Conv2dBn
from seg_utils import freeze_model, filter_keras_submodules
from seg_backbonesfactory import Backbones

os.environ['CUDA_VISIBLE_DEVICES'] = '/device:GPU:0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


backend = keras.backend
layers = keras.layers
models = keras.models
keras_utils = keras.utils


def get_submodules_from_kwargs(kwargs):
    backend = kwargs['backend']
    layers = kwargs['layers']
    models = kwargs['models']
    utils = kwargs['utils']
    for key in kwargs.keys():
        if key not in ['backend', 'layers', 'models', 'utils']:
            raise TypeError('Invalid keyword argument: %s', key)
    return backend, layers, models, utils
    


def get_submodules():
    return {
        'backend': keras.backend,
        'models': keras.models,
        'layers': keras.layers,
        'utils': keras.utils,
    }

def Conv3x3BnReLU(filters, use_batchnorm, name=None):
    kwargs = {'backend': keras.backend,'models': keras.models,'layers': keras.layers,'utils': keras.utils}

    def wrapper(input_tensor):
        return Conv2dBn(
            filters,
            kernel_size=3,
            activation='relu',
            kernel_initializer='he_uniform',
            padding='same',
            use_batchnorm=use_batchnorm,
            name=name,
            **kwargs
        )(input_tensor)

    return wrapper


def DecoderUpsamplingX2Block(filters, stage, use_batchnorm=False):
    up_name = 'decoder_stage{}_upsampling'.format(stage)
    conv1_name = 'decoder_stage{}a'.format(stage)
    conv2_name = 'decoder_stage{}b'.format(stage)
    concat_name = 'decoder_stage{}_concat'.format(stage)

    concat_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    def wrapper(input_tensor, skip=None):
        x = layers.UpSampling2D(size=2, name=up_name)(input_tensor)

        if skip is not None:
            x = layers.Concatenate(axis=concat_axis, name=concat_name)([x, skip])

        x = Conv3x3BnReLU(filters, use_batchnorm, name=conv1_name)(x)
        x = Conv3x3BnReLU(filters, use_batchnorm, name=conv2_name)(x)

        return x

    return wrapper


def DecoderTransposeX2Block(filters, stage, use_batchnorm=False):
    transp_name = 'decoder_stage{}a_transpose'.format(stage)
    bn_name = 'decoder_stage{}a_bn'.format(stage)
    relu_name = 'decoder_stage{}a_relu'.format(stage)
    conv_block_name = 'decoder_stage{}b'.format(stage)
    concat_name = 'decoder_stage{}_concat'.format(stage)

    concat_axis = bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    def layer(input_tensor, skip=None):

        x = layers.Conv2DTranspose(
            filters,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding='same',
            name=transp_name,
            use_bias=not use_batchnorm,
        )(input_tensor)

        if use_batchnorm:
            x = layers.BatchNormalization(axis=bn_axis, name=bn_name)(x)

        x = layers.Activation('relu', name=relu_name)(x)

        if skip is not None:
            x = layers.Concatenate(axis=concat_axis, name=concat_name)([x, skip])

        x = Conv3x3BnReLU(filters, use_batchnorm, name=conv_block_name)(x)

        return x

    return layer

def build_unet(
        backbone,
        decoder_block,
        skip_connection_layers,
        decoder_filters=(256, 128, 64, 32, 16),
        n_upsample_blocks=5,
        classes=1,
        activation='sigmoid',
        use_batchnorm=True,
):
    input_ = backbone.input
    x = backbone.output

    # extract skip connections
    skips = ([backbone.get_layer(name=i).output if isinstance(i, str)
              else backbone.get_layer(index=i).output for i in ['block5_conv3', 'block4_conv3', 'block3_conv3', 'block2_conv2', 'block1_conv2']]) # skip_connection_layers])

    # add center block if previous operation was maxpooling (for vgg models)
    if isinstance(backbone.layers[-1], layers.MaxPooling2D):
        x = Conv3x3BnReLU(512, use_batchnorm, name='center_block1')(x)
        x = Conv3x3BnReLU(512, use_batchnorm, name='center_block2')(x)

    # building decoder blocks
    for i in range(n_upsample_blocks):

        if i < len(skips):
            skip = skips[i]
        else:
            skip = None

        x = decoder_block(decoder_filters[i], stage=i, use_batchnorm=use_batchnorm)(x, skip)

    # model head (define number of output classes)
    x = layers.Conv2D(
        filters=classes,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='final_conv',
    )(x)
    x = layers.Activation(activation, name=activation)(x)

    # create keras model instance
    model = keras.models.Model(input_, x)

    return model

class MetaLearner():

    def __init__(self,args=None):
    
        self.classes = args.classes   
        '''it should be 1+1 (background + cloud) ???'''
        self.decoder_filters =(256, 128, 64, 32, 16) 
        self.backbone_name='vgg16',
        self.input_shape=(None, None, 3),
        self.activation='sigmoid',
        self.weights=None,
        self.encoder_weights='imagenet',
        self.encoder_freeze=False,
        self.encoder_features='default',
        self.decoder_block_type='upsampling',
        self.decoder_use_batchnorm=True
        
    
    def initialize_Unet(self): 
    
        kwargs = get_submodules()
        global backend, layers, models, keras_utils
        submodule_args = filter_keras_submodules(kwargs)
        backend, layers, models, keras_utils = get_submodules_from_kwargs(submodule_args)
        #if self.decoder_block_type == 'upsampling':
        decoder_block = DecoderUpsamplingX2Block
        #elif self.decoder_block_type == 'transpose':
        #   decoder_block = DecoderTransposeX2Block
        #else:
        #    raise ValueError('Decoder block type should be in ("upsampling", "transpose"). '
        #                     'Got: {}'.format(self.decoder_block_type))

        backbone = Backbones.get_backbone(
            name='vgg16', #self.backbone_name
            input_shape=(None,None,3), #self.input_shape,
            weights='imagenet', #self.encoder_weights,
            include_top=False,
            #**kwargs,
        )

        #if self.encoder_features == 'default':
        self.encoder_features = Backbones.get_feature_layers('vgg16', n=4) #self.backbone_name
 
        model = build_unet(
            backbone=backbone,
            decoder_block= decoder_block,
            skip_connection_layers= 'default', #self.encoder_features,
            decoder_filters=(256, 128, 64, 32, 16), #self.decoder_filters,
            classes= 1 ,#self.classes,
            activation='sigmoid', #self.activation,
            n_upsample_blocks=len((256, 128, 64, 32, 16)), #self.decoder_filters
            use_batchnorm=True #self.decoder_use_batchnorm,
        )

        # lock encoder weights for fine-tuning
        '''encored feeze is False''' 
        #if self.encoder_freeze:
        #    freeze_model(self.backbone, **kwargs)

        # loading model weights
        '''weights are None'''
        #if self.weights is not None:
        #    model.load_weights(self.weights)
        
        return model
        

    def initialize(cls,model):
    
        ip_size = (1,256,256,3)
        model.build(ip_size)
        
        return model
        
    def inner_weights(self,model):
        weights = model.trainable_weights
        return weights
        
        
    def hard_copy(cls,model,args):
        ml_instance = cls(args)
        copied_model = ml_instance.initialize_Unet()
        copied_model.build((None,None,None,3))
        copied_model.set_weights(model.get_weights())
        
    def meta_update(cls,model,args,alpha=0.01,grads=None): #grads are computed over trainable weights
        
        '''
        :parama cls: class MetaLeaner
        :param model: model to be copied
        :param alpha: the inner learning rate when update fast weights
        :param grads: gradients to generate fast weights
        
        :return model with fast weights
        '''

        copied_model = cls.initialize_Unet()
        
        copied_model.build((1,256,256,3))
        
        #make hard copy
        for j in range(len(copied_model.layers)):
                    copied_model.layers[j].kernel = model.layers[j].kernel
                    copied_model.layers[j].bias = model.layers[j].bias
                                
        #copied_model.set_weights(model.get_weights())
        
        #manually update weights, we just consider trainable weights
        #because gradients passed in input are computed from inner weights function
        #by watching inner trainable weights
        
        k=0
        for j in range(0,len(copeid_model.layers)):
                    copeid_model.layers[j].kernel = tf.subtract(model.layers[j].kernel,
                                tf.multiply(alpha, grads[k]))
                    copeid_model.layers[j].bias = tf.subtract(model.layers[j].bias,
                                tf.multiply(alpha, grads[k+1]))
                    k += 2
        '''
        
        new_weights  = []
        g = 0
        for i in range(0,len(copied_model.weights)):
            if copied_model.weights[i].trainable:
                new_weights.append(np.array(copied_model.weights[i] - alpha*grads[g]))
                g += 1
            else:
                new_weights.append(np.array(copied_model.weights[i]))

        copied_model.set_weights(new_weights)
        '''

        return copied_model
        
        
    
                
                
        
         
        
        
