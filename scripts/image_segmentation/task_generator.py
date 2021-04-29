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
import cv2
import time
import tifffile
from skimage.transform import resize


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
        return [tf.convert_to_tensor(obj).image.resize(obj,[96,96]) for obj in np_objects]
    
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
    tasks = TaskGenerator()
    tasks.mode = 'train'
    for i in range(20):
        batch_set = tasks.train_batch()
        # tasks.print_label_map()
        print (len(batch_set))
        time.sleep(5)
    
    '''
    @TODO
    change to np.random.choice
    And find out the reason why so many repeat
    '''
