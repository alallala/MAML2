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
        
        copied_model = cls.initialize_Unet()
        copied_model.build((1,256,256,3))
        
        
        copied_model.get_layer("block1_conv1").kernel = model.get_layer("block1_conv1").kernel 
        copied_model.get_layer("block1_conv1").bias = model.get_layer("block1_conv1").bias
        
        copied_model.get_layer("block1_conv2").kernel = model.get_layer("block1_conv2").kernel
        copied_model.get_layer("block1_conv2").bias = model.get_layer("block1_conv2").bias 
        
        
        copied_model.get_layer("block2_conv1").kernel = model.get_layer("block2_conv1").kernel
        copied_model.get_layer("block2_conv1").bias = model.get_layer("block2_conv1").bias
         
        copied_model.get_layer("block2_conv2").kernel = model.get_layer("block2_conv2").kernel
        copied_model.get_layer("block2_conv2").bias = model.get_layer("block2_conv2").bias
        
        
        copied_model.get_layer("block3_conv1").kernel = model.get_layer("block3_conv1").kernel
        copied_model.get_layer("block3_conv1").bias = model.get_layer("block3_conv1").bias
       
        copied_model.get_layer("block3_conv2").kernel = model.get_layer("block3_conv2").kernel
        copied_model.get_layer("block3_conv2").bias = model.get_layer("block3_conv2").bias
        
        copied_model.get_layer("block3_conv3").kernel = model.get_layer("block3_conv3").kernel
        copied_model.get_layer("block3_conv3").bias = model.get_layer("block3_conv3").bias
        
        
        copied_model.get_layer("block4_conv1").kernel = model.get_layer("block4_conv1").kernel
        copied_model.get_layer("block4_conv1").bias = model.get_layer("block4_conv1").bias
        
        copied_model.get_layer("block4_conv2").kernel = model.get_layer("block4_conv2").kernel
        copied_model.get_layer("block4_conv2").bias = model.get_layer("block4_conv2").bias
        
        copied_model.get_layer("block4_conv3").kernel = model.get_layer("block4_conv3").kernel
        copied_model.get_layer("block4_conv3").bias = model.get_layer("block4_conv3").bias
   
         
        copied_model.get_layer("block5_conv1").kernel = model.get_layer("block5_conv1").kernel
        copied_model.get_layer("block5_conv1").bias = model.get_layer("block5_conv1").bias
        
        copied_model.get_layer("block5_conv2").kernel = model.get_layer("block5_conv2").kernel
        copied_model.get_layer("block5_conv2").bias = model.get_layer("block5_conv2").bias
        
        copied_model.get_layer("block5_conv3").kernel = model.get_layer("block5_conv3").kernel
        copied_model.get_layer("block5_conv3").bias = model.get_layer("block5_conv3").bias
         
        
        copied_model.get_layer("center_block1_conv").kernel = model.get_layer("center_block1_conv").kernel
        copied_model.get_layer("center_block1_bn").gamma = model.get_layer("center_block1_bn").gamma
        copied_model.get_layer("center_block1_bn").beta = model.get_layer("center_block1_bn").beta
        
        
        copied_model.get_layer("center_block2_conv").kernel = model.get_layer("center_block2_conv").kernel
        copied_model.get_layer("center_block2_bn").gamma = model.get_layer("center_block2_bn").gamma
        copied_model.get_layer("center_block2_bn").beta = model.get_layer("center_block2_bn").beta


        copied_model.get_layer("decoder_stage0a_conv").kernel = model.get_layer("decoder_stage0a_conv").kernel 
        copied_model.get_layer("decoder_stage0a_bn").gamma = model.get_layer("decoder_stage0a_bn").gamma 
        copied_model.get_layer("decoder_stage0a_bn").beta = model.get_layer("decoder_stage0a_bn").beta 
        
        copied_model.get_layer("decoder_stage0b_conv").kernel = model.get_layer("decoder_stage0b_conv").kernel 
        copied_model.get_layer("decoder_stage0b_bn").gamma = model.get_layer("decoder_stage0b_bn").gamma 
        copied_model.get_layer("decoder_stage0b_bn").beta = model.get_layer("decoder_stage0b_bn").beta 

        copied_model.get_layer("decoder_stage1a_conv").kernel = model.get_layer("decoder_stage1a_conv").kernel 
        copied_model.get_layer("decoder_stage1a_bn").gamma = model.get_layer("decoder_stage1a_bn").gamma 
        copied_model.get_layer("decoder_stage1a_bn").beta = model.get_layer("decoder_stage1a_bn").beta 
        
        copied_model.get_layer("decoder_stage1b_conv").kernel = model.get_layer("decoder_stage1b_conv").kernel 
        copied_model.get_layer("decoder_stage1b_bn").gamma = model.get_layer("decoder_stage1b_bn").gamma 
        copied_model.get_layer("decoder_stage1b_bn").beta = model.get_layer("decoder_stage1b_bn").beta
        
        
        copied_model.get_layer("decoder_stage2a_conv").kernel = model.get_layer("decoder_stage2a_conv").kernel 
        copied_model.get_layer("decoder_stage2a_bn").gamma = model.get_layer("decoder_stage2a_bn").gamma 
        copied_model.get_layer("decoder_stage2a_bn").beta = model.get_layer("decoder_stage1a_bn").beta 
        
        copied_model.get_layer("decoder_stage2b_conv").kernel = model.get_layer("decoder_stage2b_conv").kernel 
        copied_model.get_layer("decoder_stage2b_bn").gamma = model.get_layer("decoder_stage2b_bn").gamma 
        copied_model.get_layer("decoder_stage2b_bn").beta = model.get_layer("decoder_stage2b_bn").beta
        
        copied_model.get_layer("decoder_stage3a_conv").kernel = model.get_layer("decoder_stage3a_conv").kernel 
        copied_model.get_layer("decoder_stage3a_bn").gamma = model.get_layer("decoder_stage3a_bn").gamma 
        copied_model.get_layer("decoder_stage3a_bn").beta = model.get_layer("decoder_stage3a_bn").beta 
        
        copied_model.get_layer("decoder_stage3b_conv").kernel = model.get_layer("decoder_stage3b_conv").kernel 
        copied_model.get_layer("decoder_stage3b_bn").gamma = model.get_layer("decoder_stage3b_bn").gamma 
        copied_model.get_layer("decoder_stage3b_bn").beta = model.get_layer("decoder_stage3b_bn").beta
        
     
        copied_model.get_layer("decoder_stage4a_conv").kernel = model.get_layer("decoder_stage4a_conv").kernel 
        copied_model.get_layer("decoder_stage4a_bn").gamma = model.get_layer("decoder_stage4a_bn").gamma 
        copied_model.get_layer("decoder_stage4a_bn").beta = model.get_layer("decoder_stage4a_bn").beta 
        
        copied_model.get_layer("decoder_stage4b_conv").kernel = model.get_layer("decoder_stage4b_conv").kernel 
        copied_model.get_layer("decoder_stage4b_bn").gamma = model.get_layer("decoder_stage4b_bn").gamma 
        copied_model.get_layer("decoder_stage4b_bn").beta = model.get_layer("decoder_stage4b_bn").beta
       
        copied_model.get_layer("final_conv").kernel = model.get_layer("final_conv").kernel 
        copied_model.get_layer("final_conv").bias = model.get_layer("final_conv").bias
        
        
        
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

        #copied_model = keras.models.clone_model(model)
        #copied_model.set_weights(model.get_weights())
        
     
        copied_model.get_layer("block1_conv1").kernel = model.get_layer("block1_conv1").kernel 
        copied_model.get_layer("block1_conv1").bias = model.get_layer("block1_conv1").bias
        
        copied_model.get_layer("block1_conv2").kernel = model.get_layer("block1_conv2").kernel
        copied_model.get_layer("block1_conv2").bias = model.get_layer("block1_conv2").bias 
        
        
        copied_model.get_layer("block2_conv1").kernel = model.get_layer("block2_conv1").kernel
        copied_model.get_layer("block2_conv1").bias = model.get_layer("block2_conv1").bias
         
        copied_model.get_layer("block2_conv2").kernel = model.get_layer("block2_conv2").kernel
        copied_model.get_layer("block2_conv2").bias = model.get_layer("block2_conv2").bias
        
        
        copied_model.get_layer("block3_conv1").kernel = model.get_layer("block3_conv1").kernel
        copied_model.get_layer("block3_conv1").bias = model.get_layer("block3_conv1").bias
       
        copied_model.get_layer("block3_conv2").kernel = model.get_layer("block3_conv2").kernel
        copied_model.get_layer("block3_conv2").bias = model.get_layer("block3_conv2").bias
        
        copied_model.get_layer("block3_conv3").kernel = model.get_layer("block3_conv3").kernel
        copied_model.get_layer("block3_conv3").bias = model.get_layer("block3_conv3").bias
        
        
        copied_model.get_layer("block4_conv1").kernel = model.get_layer("block4_conv1").kernel
        copied_model.get_layer("block4_conv1").bias = model.get_layer("block4_conv1").bias
        
        copied_model.get_layer("block4_conv2").kernel = model.get_layer("block4_conv2").kernel
        copied_model.get_layer("block4_conv2").bias = model.get_layer("block4_conv2").bias
        
        copied_model.get_layer("block4_conv3").kernel = model.get_layer("block4_conv3").kernel
        copied_model.get_layer("block4_conv3").bias = model.get_layer("block4_conv3").bias
   
         
        copied_model.get_layer("block5_conv1").kernel = model.get_layer("block5_conv1").kernel
        copied_model.get_layer("block5_conv1").bias = model.get_layer("block5_conv1").bias
        
        copied_model.get_layer("block5_conv2").kernel = model.get_layer("block5_conv2").kernel
        copied_model.get_layer("block5_conv2").bias = model.get_layer("block5_conv2").bias
        
        copied_model.get_layer("block5_conv3").kernel = model.get_layer("block5_conv3").kernel
        copied_model.get_layer("block5_conv3").bias = model.get_layer("block5_conv3").bias
         
        
        copied_model.get_layer("center_block1_conv").kernel = model.get_layer("center_block1_conv").kernel
        copied_model.get_layer("center_block1_bn").gamma = model.get_layer("center_block1_bn").gamma
        copied_model.get_layer("center_block1_bn").beta = model.get_layer("center_block1_bn").beta
        
        
        copied_model.get_layer("center_block2_conv").kernel = model.get_layer("center_block2_conv").kernel
        copied_model.get_layer("center_block2_bn").gamma = model.get_layer("center_block2_bn").gamma
        copied_model.get_layer("center_block2_bn").beta = model.get_layer("center_block2_bn").beta


        copied_model.get_layer("decoder_stage0a_conv").kernel = model.get_layer("decoder_stage0a_conv").kernel 
        copied_model.get_layer("decoder_stage0a_bn").gamma = model.get_layer("decoder_stage0a_bn").gamma 
        copied_model.get_layer("decoder_stage0a_bn").beta = model.get_layer("decoder_stage0a_bn").beta 
        
        copied_model.get_layer("decoder_stage0b_conv").kernel = model.get_layer("decoder_stage0b_conv").kernel 
        copied_model.get_layer("decoder_stage0b_bn").gamma = model.get_layer("decoder_stage0b_bn").gamma 
        copied_model.get_layer("decoder_stage0b_bn").beta = model.get_layer("decoder_stage0b_bn").beta 

        copied_model.get_layer("decoder_stage1a_conv").kernel = model.get_layer("decoder_stage1a_conv").kernel 
        copied_model.get_layer("decoder_stage1a_bn").gamma = model.get_layer("decoder_stage1a_bn").gamma 
        copied_model.get_layer("decoder_stage1a_bn").beta = model.get_layer("decoder_stage1a_bn").beta 
        
        copied_model.get_layer("decoder_stage1b_conv").kernel = model.get_layer("decoder_stage1b_conv").kernel 
        copied_model.get_layer("decoder_stage1b_bn").gamma = model.get_layer("decoder_stage1b_bn").gamma 
        copied_model.get_layer("decoder_stage1b_bn").beta = model.get_layer("decoder_stage1b_bn").beta
        
        
        copied_model.get_layer("decoder_stage2a_conv").kernel = model.get_layer("decoder_stage2a_conv").kernel 
        copied_model.get_layer("decoder_stage2a_bn").gamma = model.get_layer("decoder_stage2a_bn").gamma 
        copied_model.get_layer("decoder_stage2a_bn").beta = model.get_layer("decoder_stage1a_bn").beta 
        
        copied_model.get_layer("decoder_stage2b_conv").kernel = model.get_layer("decoder_stage2b_conv").kernel 
        copied_model.get_layer("decoder_stage2b_bn").gamma = model.get_layer("decoder_stage2b_bn").gamma 
        copied_model.get_layer("decoder_stage2b_bn").beta = model.get_layer("decoder_stage2b_bn").beta
        
        copied_model.get_layer("decoder_stage3a_conv").kernel = model.get_layer("decoder_stage3a_conv").kernel 
        copied_model.get_layer("decoder_stage3a_bn").gamma = model.get_layer("decoder_stage3a_bn").gamma 
        copied_model.get_layer("decoder_stage3a_bn").beta = model.get_layer("decoder_stage3a_bn").beta 
        
        copied_model.get_layer("decoder_stage3b_conv").kernel = model.get_layer("decoder_stage3b_conv").kernel 
        copied_model.get_layer("decoder_stage3b_bn").gamma = model.get_layer("decoder_stage3b_bn").gamma 
        copied_model.get_layer("decoder_stage3b_bn").beta = model.get_layer("decoder_stage3b_bn").beta
        
     
        copied_model.get_layer("decoder_stage4a_conv").kernel = model.get_layer("decoder_stage4a_conv").kernel 
        copied_model.get_layer("decoder_stage4a_bn").gamma = model.get_layer("decoder_stage4a_bn").gamma 
        copied_model.get_layer("decoder_stage4a_bn").beta = model.get_layer("decoder_stage4a_bn").beta 
        
        copied_model.get_layer("decoder_stage4b_conv").kernel = model.get_layer("decoder_stage4b_conv").kernel 
        copied_model.get_layer("decoder_stage4b_bn").gamma = model.get_layer("decoder_stage4b_bn").gamma 
        copied_model.get_layer("decoder_stage4b_bn").beta = model.get_layer("decoder_stage4b_bn").beta
       
        copied_model.get_layer("final_conv").kernel = model.get_layer("final_conv").kernel 
        copied_model.get_layer("final_conv").bias = model.get_layer("final_conv").bias
        
        
        #manually update weights, we just consider trainable weights
        #because gradients passed in input are computed from inner weights function
        #by watching inner trainable weights
          
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
        
        copied_model.get_layer("block1_conv1").kernel = copied_model.get_layer("block1_conv1").kernel - alpha * grads[0]
        copied_model.get_layer("block1_conv1").bias = copied_model.get_layer("block1_conv1").bias - alpha * grads[1]
        
        copied_model.get_layer("block1_conv2").kernel = copied_model.get_layer("block1_conv2").kernel- alpha * grads[2]
        copied_model.get_layer("block1_conv2").bias = copied_model.get_layer("block1_conv2").bias - alpha * grads[3]
        
        
        copied_model.get_layer("block2_conv1").kernel = copied_model.get_layer("block2_conv1").kernel - alpha * grads[4]
        copied_model.get_layer("block2_conv1").bias = copied_model.get_layer("block2_conv1").bias - alpha * grads[5]
         
        copied_model.get_layer("block2_conv2").kernel = copied_model.get_layer("block2_conv2").kernel - alpha * grads[6]
        copied_model.get_layer("block2_conv2").bias = copied_model.get_layer("block2_conv2").bias - alpha * grads[7]
        
        
        copied_model.get_layer("block3_conv1").kernel = copied_model.get_layer("block3_conv1").kernel - alpha * grads[8]
        copied_model.get_layer("block3_conv1").bias = copied_model.get_layer("block3_conv1").bias - alpha * grads[9]
       
        copied_model.get_layer("block3_conv2").kernel = copied_model.get_layer("block3_conv2").kernel - alpha * grads[10]
        copied_model.get_layer("block3_conv2").bias = copied_model.get_layer("block3_conv2").bias - alpha * grads[11]
        
        copied_model.get_layer("block3_conv3").kernel = copied_model.get_layer("block3_conv3").kernel - alpha * grads[12]
        copied_model.get_layer("block3_conv3").bias = copied_model.get_layer("block3_conv3").bias - alpha * grads[13]
        
        
        copied_model.get_layer("block4_conv1").kernel = copied_model.get_layer("block4_conv1").kernel - alpha * grads[14]
        copied_model.get_layer("block4_conv1").bias = copied_model.get_layer("block4_conv1").bias - alpha * grads[15]
        
        copied_model.get_layer("block4_conv2").kernel = copied_model.get_layer("block4_conv2").kernel - alpha * grads[16]
        copied_model.get_layer("block4_conv2").bias = copied_model.get_layer("block4_conv2").bias - alpha * grads[17]
        
        copied_model.get_layer("block4_conv3").kernel = copied_model.get_layer("block4_conv3").kernel - alpha * grads[18]
        copied_model.get_layer("block4_conv3").bias = copied_model.get_layer("block4_conv3").bias - alpha * grads[19]
   
         
        copied_model.get_layer("block5_conv1").kernel = copied_model.get_layer("block5_conv1").kernel - alpha * grads[20]
        copied_model.get_layer("block5_conv1").bias = copied_model.get_layer("block5_conv1").bias - alpha * grads[21]
        
        copied_model.get_layer("block5_conv2").kernel = copied_model.get_layer("block5_conv2").kernel - alpha * grads[22]
        copied_model.get_layer("block5_conv2").bias = copied_model.get_layer("block5_conv2").bias - alpha * grads[23]
        
        copied_model.get_layer("block5_conv3").kernel = copied_model.get_layer("block5_conv3").kernel - alpha * grads[24]
        copied_model.get_layer("block5_conv3").bias = copied_model.get_layer("block5_conv3").bias - alpha * grads[25]
         
        
        copied_model.get_layer("center_block1_conv").kernel = copied_model.get_layer("center_block1_conv").kernel - alpha * grads[26]
        copied_model.get_layer("center_block1_bn").gamma = copied_model.get_layer("center_block1_bn").gamma - alpha * grads[27]
        copied_model.get_layer("center_block1_bn").beta = copied_model.get_layer("center_block1_bn").beta - alpha * grads[28]
        
        
        copied_model.get_layer("center_block2_conv").kernel = copied_model.get_layer("center_block2_conv").kernel - alpha * grads[29]
        copied_model.get_layer("center_block2_bn").gamma = copied_model.get_layer("center_block2_bn").gamma - alpha * grads[30]
        copied_model.get_layer("center_block2_bn").beta = copied_model.get_layer("center_block2_bn").beta - alpha * grads[31]


        copied_model.get_layer("decoder_stage0a_conv").kernel = copied_model.get_layer("decoder_stage0a_conv").kernel - alpha * grads[32]
        copied_model.get_layer("decoder_stage0a_bn").gamma = copied_model.get_layer("decoder_stage0a_bn").gamma - alpha * grads[33]
        copied_model.get_layer("decoder_stage0a_bn").beta = copied_model.get_layer("decoder_stage0a_bn").beta - alpha * grads[34]
        
        copied_model.get_layer("decoder_stage0b_conv").kernel = copied_model.get_layer("decoder_stage0b_conv").kernel - alpha * grads[35]
        copied_model.get_layer("decoder_stage0b_bn").gamma = copied_model.get_layer("decoder_stage0b_bn").gamma - alpha * grads[36]
        copied_model.get_layer("decoder_stage0b_bn").beta = copied_model.get_layer("decoder_stage0b_bn").beta - alpha * grads[37]

        copied_model.get_layer("decoder_stage1a_conv").kernel = copied_model.get_layer("decoder_stage1a_conv").kernel - alpha * grads[38]
        copied_model.get_layer("decoder_stage1a_bn").gamma = copied_model.get_layer("decoder_stage1a_bn").gamma - alpha * grads[39]
        copied_model.get_layer("decoder_stage1a_bn").beta = copied_model.get_layer("decoder_stage1a_bn").beta - alpha * grads[40]
        
        copied_model.get_layer("decoder_stage1b_conv").kernel = copied_model.get_layer("decoder_stage1b_conv").kernel - alpha * grads[41]
        copied_model.get_layer("decoder_stage1b_bn").gamma = copied_model.get_layer("decoder_stage1b_bn").gamma - alpha * grads[42]
        copied_model.get_layer("decoder_stage1b_bn").beta = copied_model.get_layer("decoder_stage1b_bn").beta - alpha * grads[43]
        
        
        copied_model.get_layer("decoder_stage2a_conv").kernel = copied_model.get_layer("decoder_stage2a_conv").kernel - alpha * grads[44]
        copied_model.get_layer("decoder_stage2a_bn").gamma = copied_model.get_layer("decoder_stage2a_bn").gamma - alpha * grads[45]
        copied_model.get_layer("decoder_stage2a_bn").beta = copied_model.get_layer("decoder_stage1a_bn").beta - alpha * grads[46]
        
        copied_model.get_layer("decoder_stage2b_conv").kernel = copied_model.get_layer("decoder_stage2b_conv").kernel - alpha * grads[47]
        copied_model.get_layer("decoder_stage2b_bn").gamma = copied_model.get_layer("decoder_stage2b_bn").gamma - alpha * grads[48]
        copied_model.get_layer("decoder_stage2b_bn").beta = copied_model.get_layer("decoder_stage2b_bn").beta - alpha * grads[49]
        
        copied_model.get_layer("decoder_stage3a_conv").kernel = copied_model.get_layer("decoder_stage3a_conv").kernel - alpha * grads[50]
        copied_model.get_layer("decoder_stage3a_bn").gamma = copied_model.get_layer("decoder_stage3a_bn").gamma - alpha * grads[51]
        copied_model.get_layer("decoder_stage3a_bn").beta = copied_model.get_layer("decoder_stage3a_bn").beta - alpha * grads[52]
        
        copied_model.get_layer("decoder_stage3b_conv").kernel = copied_model.get_layer("decoder_stage3b_conv").kernel - alpha * grads[53]
        copied_model.get_layer("decoder_stage3b_bn").gamma = copied_model.get_layer("decoder_stage3b_bn").gamma - alpha * grads[54]
        copied_model.get_layer("decoder_stage3b_bn").beta = copied_model.get_layer("decoder_stage3b_bn").beta - alpha * grads[55]
        
     
        copied_model.get_layer("decoder_stage4a_conv").kernel = copied_model.get_layer("decoder_stage4a_conv").kernel - alpha * grads[56]
        copied_model.get_layer("decoder_stage4a_bn").gamma = copied_model.get_layer("decoder_stage4a_bn").gamma - alpha * grads[57]
        copied_model.get_layer("decoder_stage4a_bn").beta = copied_model.get_layer("decoder_stage4a_bn").beta - alpha * grads[58]
        
        copied_model.get_layer("decoder_stage4b_conv").kernel = copied_model.get_layer("decoder_stage4b_conv").kernel - alpha * grads[59]
        copied_model.get_layer("decoder_stage4b_bn").gamma = copied_model.get_layer("decoder_stage4b_bn").gamma - alpha * grads[60]
        copied_model.get_layer("decoder_stage4b_bn").beta = copied_model.get_layer("decoder_stage4b_bn").beta - alpha * grads[61]
       
        copied_model.get_layer("final_conv").kernel = copied_model.get_layer("final_conv").kernel - alpha * grads[62]
        copied_model.get_layer("final_conv").bias = copied_model.get_layer("final_conv").bias - alpha * grads[63]
        
        return copied_model
        
        
    
                
                
        
         
        
        
