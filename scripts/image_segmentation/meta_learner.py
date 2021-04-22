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

os.environ['CUDA_VISIBLE_DEVICES'] = '/device:GPU:0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


backend = None
layers = None
models = None
keras_utils = None


def get_submodules():
    return {
        'backend': backend,
        'models': models,
        'layers': layers,
        'utils': keras_utils,
    }

def Conv3x3BnReLU(filters, use_batchnorm, name=None):
    kwargs = get_submodules()

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
              else backbone.get_layer(index=i).output for i in skip_connection_layers])

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
    model = models.Model(input_, x)

    return model

class MetaLearner():
    def __init__(self,args=None):
    
        self.classes = args.n_way #ADD TO MAIN  
        '''it should be 1+1 (background + cloud)'''
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
        

     
    def initialize_Unet(self): #it should be initialize()

        global backend, layers, models, keras_utils
        submodule_args = filter_keras_submodules(kwargs)
        backend, layers, models, keras_utils = get_submodules_from_kwargs(submodule_args)

        if decoder_block_type == 'upsampling':
            decoder_block = DecoderUpsamplingX2Block
        elif decoder_block_type == 'transpose':
            decoder_block = DecoderTransposeX2Block
        else:
            raise ValueError('Decoder block type should be in ("upsampling", "transpose"). '
                             'Got: {}'.format(decoder_block_type))

        backbone = Backbones.get_backbone(
            backbone_name=self.backbone_name,
            input_shape=self.input_shape,
            weights=self.encoder_weights,
            include_top=False,
            **kwargs,
        )

        if self.encoder_features == 'default':
            self.encoder_features = Backbones.get_feature_layers(self.backbone_name, n=4)

        model = build_unet(
            backbone=self.backbone,
            decoder_block=self.decoder_block,
            skip_connection_layers=self.encoder_features,
            decoder_filters=self.decoder_filters,
            classes=self.classes,
            activation=self.activation,
            n_upsample_blocks=len(self.decoder_filters),
            use_batchnorm=self.decoder_use_batchnorm,
        )

        # lock encoder weights for fine-tuning
        if self.encoder_freeze:
            freeze_model(self.backbone, **kwargs)

        # loading model weights
        if self.weights is not None:
            model.load_weights(self.weights)

        return model
        
    def inner_weights(model):
        weights = [layer.trainable_weights for layer in model.layers]
        return weights
        
        
    def hard_copy(cls,model,args):
        ml_instance = cls(args)
        copied_model = ml_instance.initialize_Unet()
        copied_model.set_weights(model.get_weights())
        
    def meta_update(cls,model,args,alpha=0.01,grads=None): #grads are computed over trainable weights
        
        '''
        :parama cls: class MetaLeaner
        :param model: model to be copied
        :param alpha: the inner learning rate when update fast weights
        :param grads: gradients to generate fast weights
        
        :return model with fast weights
        '''
        ml_instance = cls(args)
        copied_model = ml_instance.initialize_Unet()
        copied_model.set_weights(model.get_weights())
        
        #manually update weights, we just consider trainable weights
        #because gradients passed in input are computed from inner weights function
        #by watching inner trainable weights
        
        new_weights  = []
        g = 0
        for i in range(0,len(copied_model.weights)):
            if copied_model.weights[i].trainable:
                new_weights.append(copied_model.weights[i] - alpha*grads[g])
                g += 1
            else:
                new_weights.append(copied_model.weights[i])
         
        copied_model.set_weights(new_weights)
        
        return copied_model
        
        
    
                
                
        
         
        
        
