'''
    Date: 14th Feb 2020
    Author: HilbertXu
    Abstract: Code for generating meta-train tasks using clarity dataset
              Meta learning is different from general supervised learning
              The basic training element in training process is TASK(N-way K-shot)
              A batch contains several tasks
              tasks: containing N-way K-shot for meta-train, N-way N-query for meta-test
'''

from __future__ import print_function
import argparse
import csv
import glob
import os
import sys
import random
import numpy as np
from tqdm import tqdm
from tqdm._tqdm import trange
from PIL import Image
import tensorflow as tf
import time
import tifffile
from skimage.transform import resize
from tensorflow import keras 
from tensorflow.keras.preprocessing.image import array_to_img

# for loading/processing the images  
from tensorflow.keras.preprocessing.image import load_img 
from tensorflow.keras.preprocessing.image import img_to_array 
from tensorflow.keras.applications.vgg16 import preprocess_input 

# models 
from tensorflow.keras.applications.vgg16 import VGG16 
from tensorflow.keras.models import Model

# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# for everything else
import matplotlib.pyplot as plt
from random import randint


def autoencoder_and_cluster(loaded_images,n_dim,n_clu):

    def construct_ae_model(input_shape):
    
        latent_dim = n_dim
        
        inputs = keras.layers.Input(shape=input_shape) # input is an 256x256 RGB image
        
        x = inputs
        ### ENCODER ###
        x = keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = keras.layers.MaxPool2D(pool_size=(2, 2))(x)

        x = keras.layers.Conv2D(filters=8, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = keras.layers.MaxPool2D(pool_size=(2, 2))(x)

        x = keras.layers.Conv2D(filters=4, kernel_size=(3, 3), padding='same', activation='relu')(x)
        
        x = keras.layers.MaxPool2D(pool_size=(2, 2))(x)

        volumeSize = keras.backend.int_shape(x)
        x = keras.layers.Flatten()(x)
        latent = keras.layers.Dense(latent_dim)(x)  

        encoder = Model(inputs, latent, name="encoder")
        
        ### DECODER ###
        
        latentInputs = keras.layers.Input(shape=(latent_dim,))
        x = keras.layers.Dense(np.prod(volumeSize[1:]))(latentInputs)
        x = keras.layers.Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)
        
        x = keras.layers.Conv2D(filters=4, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = keras.layers.UpSampling2D(size=(2, 2))(x)

        x = keras.layers.Conv2D(filters=8, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = keras.layers.UpSampling2D(size=(2, 2))(x)

        x = keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = keras.layers.UpSampling2D(size=(2, 2))(x)

        output = keras.layers.Conv2D(filters=3, kernel_size=(3, 3), padding='same',
                        activation='sigmoid')(x)
                        
        decoder = Model(inputs=latentInputs, outputs=output, name='decoder')
        
        # our autoencoder is the encoder + decoder
        autoencoder = Model(inputs, decoder(encoder(inputs)),
            name="autoencoder")
        
        return autoencoder
        
    input_shape = loaded_images[:,:,:,:3].shape[1:] #(256,256,3)
   
    #prepare data to train the autoencoder
    data_autoencoder = loaded_images[:,:,:,:3]
    random.shuffle(data_autoencoder)
    
    num_train = int(len(data_autoencoder)*0.8)
    
    x_train = data_autoencoder[:num_train,:,:,:]
    x_train = x_train.reshape(len(x_train),input_shape[0],input_shape[1],input_shape[2])
    
    x_val = data_autoencoder[num_train:,:,:,:]
    x_val = x_val.reshape(len(x_val),input_shape[0],input_shape[1],input_shape[2])
    
    
    ae_model = construct_ae_model(input_shape=input_shape)
    
    ae_model.compile(optimizer='adam', loss='binary_crossentropy')
    
    #model train
    ae_model.fit(x_train, x_train, epochs=3, batch_size=64, validation_data=(x_val, x_val), verbose=1)
    
    #perform dimensionality reduction on train dataset to be used for segmentation 
    
    encoder = Model(ae_model.input, ae_model.layers[-2].output)
 
    '''

    def extract_features(img, model):
            
            reshaped_img = img.reshape(1,256,256,3)
            # prepare image for model
            imgx = preprocess_input(reshaped_img)
            # get the feature vector
            features = model.predict(imgx, use_multiprocessing=True)
            return features
            
    data = {}
      
    # lop through each image in the dataset
    for idx in range(0,len(loaded_images)):
        # try to extract the features and update the dictionary
        try:
            feat = extract_features(loaded_images[:,:,:,:3][idx],encoder)
            data[idx] = feat
        except IOError as exc:
            raise RuntimeError('Failed to extract features') from exc
              
     
    # get a list of the images indexes
    images_indexes = np.array(list(data.keys()))

    # get a list of just the features
    feat = np.array(list(data.values()))

    # reshape so that there are samples with dimensionality of n_dim 
    fit_images = feat.reshape(-1,n_dim)
    '''
    
    encoded_imgs = encoder.predict(loaded_images[:,:,:,:3])
    
    encoded_imgs = encoded_imgs.reshape(-1,n_dim)

    #clustering
    kmeans = KMeans(n_clusters=n_clu, n_jobs=-1, random_state=22)
    kmeans.fit(encoded_imgs)
    
    images_indexes = [i for i in range(len(loaded_images))]
    # holds the cluster id and the images { id: [images] }
    groups = {}
    
    for img_idx, cluster in zip(images_indexes, kmeans.labels_):
        print("img_idx {} in cluster {}".format(img_idx,cluster))
        if cluster not in groups.keys():
            groups[cluster] = []
            groups[cluster].append(img_idx)
            to_display = img_to_array(loaded_images[img_idx][:,:,:3])
        else:
            groups[cluster].append(img_idx)
            
    return groups
    
      

def pca_and_cluster(loaded_images,cnn,n_dim,n_clu):

    if cnn:
        model_cls = VGG16()
        model_cls = Model(inputs = model_cls.inputs, outputs = model_cls.layers[-2].output)

        def extract_features(img, model_cls):
            
            reshaped_img = img.reshape(256,256,3)
            resized_img = resize(reshaped_img, (224, 224))
            reshaped_img = resized_img.reshape(1,224,224,3)
            # prepare image for model
            imgx = preprocess_input(reshaped_img)
            # get the feature vector
            features = model_cls.predict(imgx, use_multiprocessing=True)
            return features
           
        data = {}
      
        # lop through each image in the dataset
        for idx in range(0,len(loaded_images)):
            # try to extract the features and update the dictionary
            try:
                feat = extract_features(loaded_images[:,:,:,:3][idx],model_cls)
                data[idx] = feat
            except IOError as exc:
                raise RuntimeError('Failed to extract features') from exc
                  
         
        # get a list of the images indexes
        images_indexes = np.array(list(data.keys()))

        # get a list of just the features
        feat = np.array(list(data.values()))

        # reshape so that there are samples with dimensionality of 4096 
        fit_images = feat.reshape(-1,2)
    
    else:
    
        fit_images = loaded_images[:,:,:,:3]
        h,w,c = loaded_images.shape[1],loaded_images.shape[2],loaded_images.shape[3]
        
        #reshape data to be suitable for pca (n dim<=2)
        fit_images = fit_images.reshape(-1,h*w*c)
        #print("fit images shape: ",fit_images.shape)
        
    pca = PCA(n_components=n_dim, random_state=22)
    pca.fit(fit_images) 
    x = pca.transform(fit_images) 

    # cluster feature vectors
    kmeans = KMeans(n_clusters=n_clu,n_jobs=-1, random_state=22)
    kmeans.fit(x)

    images_indexes = [i for i in range(0,len(loaded_images))]
    
    # holds the cluster id and the images { id: [images] }
    groups = {}
    for img_idx, cluster in zip(images_indexes,kmeans.labels_):
        if cluster not in groups.keys():
            groups[cluster] = []
            groups[cluster].append(img_idx)
        else:
            groups[cluster].append(img_idx)
            
    return groups   
    
                  
def sequence(start, end):
    res = []
    x = start
    while x <= end:
        res.append(x)
        x += 1
    return res
    
def load_file(f,start,end):
    if f[-4:] == 'tiff':
        print("\nloading {}\n".format(str(f.split("/")[-1])))
        seq = sequence(start,end)
        img = tifffile.imread(f,key=seq).astype(np.float32) #/255.
    img = np.asarray(img, dtype=np.float32)
    return img
    
        
class TaskGenerator:
    def __init__(self, args=None):
        '''
        :param mode: train or test
        :param n_way: a train task contains images from different N classes
        :param k_shot: k images used for meta-train
        :param k_query: k images used for meta-test
        :param meta_batchsz: the number of tasks in a batch
        :param total_batch_num: the number of batches
        '''
        if args is not None:
            self.dataset = args.dataset
            self.mode = args.mode
            self.meta_batchsz = args.meta_batchsz
            self.n_way = args.n_way
            self.spt_num = args.k_shot
            self.qry_num = args.k_query
            #self.dim_output = self.n_way 
        
        if self.dataset == 'clarity':
            
            data = load_file('/content/drive/MyDrive/cloud_dataset.tiff',1000,1500)
            for_train = int(len(data)*0.8)
            for_val = int(for_train*0.2)
            self.metatrain_data = data[:for_train-for_val]
            self.metaval_data = data[for_train-for_val:for_train]
            self.metatest_data = data[for_train:]
  
        
        # Record the relationship between image label and the folder name in each task
        # self.label_map = []
    
    '''               
    def shuffle_set(self, set_x, set_y):
        # Shuffle
        set_seed = random.randint(0, 100)
        random.seed(set_seed)
        random.shuffle(set_x)
        random.seed(set_seed)
        random.shuffle(set_y)
        return set_x, set_y
    
    '''
    
    def convert_to_tensor(self, np_objects):
        return [tf.convert_to_tensor(obj) for obj in np_objects] 
    
    def generate_set(self, data, shuffle=False):
             
        # Function for slicing the dataset
        # support set & query set
        def _slice_set(ds):
            spt_x = list()
            spt_y = list()
            qry_x = list()
            qry_y = list()
            all_idxs = [i for i in range(0,len(ds))]
            spt_elem = random.sample(all_idxs, self.spt_num) #here a scenario should be chosen
            for e in spt_elem:
                all_idxs.remove(e)
            qry_elem = random.sample(all_idxs, self.spt_num) #and here images from the same scenario should be used 
            
            '''
            spt_elem = list(zip(*spt_elem))
            qry_elem = list(zip(*qry_elem))
            '''
            spt_x.extend([ds[:,:,:,:3][idx] for idx in spt_elem])
            spt_y.extend([ds[:,:,:,3:][idx] for idx in spt_elem])
            qry_x.extend([ds[:,:,:,:3][idx] for idx in qry_elem])
            qry_y.extend([ds[:,:,:,3:][idx] for idx in qry_elem])

            # Shuffle datasets
            '''
            spt_x, spt_y = self.shuffle_set(spt_x, spt_y)
            qry_x, qry_y = self.shuffle_set(qry_x, qry_y)
            '''
            # convert to tensor
            spt_x, spt_y = self.convert_to_tensor((np.array(spt_x), np.array(spt_y)))
            qry_x, qry_y = self.convert_to_tensor((np.array(qry_x), np.array(qry_y)))
            
            # resize images
            spt_x = tf.image.resize(spt_x,[128,128])
            spt_y = tf.image.resize(spt_y,[128,128])
            qry_x = tf.image.resize(qry_x,[128,128])
            qry_y = tf.image.resize(qry_y,[128,128])

            return spt_x, spt_y, qry_x, qry_y
            
        return _slice_set(data)
              
    def train_batch(self):
        '''
        :return: a batch of support set tensor and query set tensor
        
        '''
        data = self.metatrain_data 
        # Shuffle root folder in order to prevent repeat
        batch_set = []
        # self.label_map = []
        # Generate batch dataset
        # batch_spt_set: [meta_batchsz, n_way * k_shot, image_size] & [meta_batchsz, n_way * k_shot, n_way]
        # batch_qry_set: [meta_batchsz, n_way * k_query, image_size] & [meta_batchsz, n_way * k_query, n_way]
        
        for _ in range(self.meta_batchsz):
            '''
            sampled_folders_idx = np.array(np.random.choice(len(folders), self.n_way, False))
            np.random.shuffle(sampled_folders_idx)
            sampled_folders = np.array(folders)[sampled_folders_idx].tolist()
            folder_with_label = []
            # for i, folder in enumerate(sampled_folders):
            #     elem = (folder, i)
            #     folder_with_label.append(elem)
            labels = np.arange(self.n_way)
            np.random.shuffle(labels)
            labels = labels.tolist()
            folder_with_label = list(zip(sampled_folders, labels))
            '''
            support_x, support_y, query_x, query_y = self.generate_set(data)
            batch_set.append((support_x, support_y, query_x, query_y))
        
        return batch_set
    
    def test_batch(self,test):
        '''
        :return: a batch of support set tensor and query set tensor
        
        '''
        if test:
          data = self.metatest_data
          p_str = 'test'
        else:
          data = self.metaval_data
          p_str = 'validation'
        print ('Sample '+p_str+' batch of tasks from {} images'.format(len(data)))
        # Shuffle root folder in order to prevent repeat
        batch_set = []
        # self.label_map = [] 
        # Generate batch dataset
        # batch_spt_set: [meta_batchsz, n_way * k_shot, image_size] & [meta_batchsz, n_way * k_shot, n_way]
        # batch_qry_set: [meta_batchsz, n_way * k_query, image_size] & [meta_batchsz, n_way * k_query, n_way]
        for _ in range(self.meta_batchsz):
            '''
            sampled_folders_idx = np.array(np.random.choice(len(folders), self.n_way, False))
            np.random.shuffle(sampled_folders_idx)
            sampled_folders = np.array(folders)[sampled_folders_idx].tolist()
            folder_with_label = []
            # for i, folder in enumerate(sampled_folders):
            #     elem = (folder, i)
            #     folder_with_label.append(elem)
            labels = np.arange(self.n_way)
            np.random.shuffle(labels)
            labels = labels.tolist()
            folder_with_label = list(zip(sampled_folders, labels))
            '''
            support_x, support_y, query_x, query_y = self.generate_set(data)
            batch_set.append((support_x, support_y, query_x, query_y))
        # return [meta_batchsz * (support_x, support_y, query_x, query_y)]
        return batch_set

if __name__ == '__main__':

    my_array = load_file('/content/drive/MyDrive/cloud_dataset.tiff',0,30)
    print("dataset of 2000 images of size 256x256x3= 196608\nreduction to size 1000\nclustering on 10 groups\n")
    #autoencoder
    
    print("\ndimensionality reduction with autoencoder and clustering")
    #groups = autoencoder_and_cluster(my_array,1000,2)
    groups= pca_and_cluster(my_array,True,1000,2)
    
    for cluster_id in groups.keys():
    
        print("cluster {} has {} images".format(cluster_id,len(groups[cluster_id])))
        print(groups[cluster_id])
        print("\n")
        plt.figure(figsize = (25,25));
        # gets the list of images indexes for a cluster
        indexes = groups[cluster_id]
        # only allow up to 30 images to be shown at a time
        
        if len(indexes) > 30:
            print(f"Clipping cluster size from {len(indexes)} to 30")  
            indexes = indexes[:29]
                
        # plot each image in the cluster
        
        for i,idx in enumerate(indexes):
            print(i,idx) 
            plt.subplot(10,10,i+1);
            to_display = array_to_img(my_array[idx][:,:,:3])
            plt.imshow(to_display)
            plt.axis('off')

        plt.show()
        
    '''
    #pca
    print("\ndimensionality reduction with pca and clustering")
    groups_pca = pca_and_cluster(my_array,True,1000,10)
    show_clusters(my_array,groups_pca)
    '''   
   
