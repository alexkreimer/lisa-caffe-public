#!/usr/bin/env python

#Data layer for video.  Change flow_frames and RGB_frames to be the path to the flow and RGB frames.

import sys
sys.path.append('../../python')
import caffe
import io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import time
import pdb
import glob
import pickle as pkl
import random
import h5py
from multiprocessing import Pool
from threading import Thread
import skimage.io
import copy

class settings:
  train_images_list  = './train_images.txt'
  test_images_list   = './test_images.txt'  
  test_frames        = 16 #number of frames to process in a batch
  train_frames       = 16

def processImageCrop(im_info, transformer):
  im_path = im_info[0]
  im_crop = im_info[1] 
  im_reshape = im_info[2]
  im_flip = im_info[3]
  data_in = caffe.io.load_image(im_path)
  if (data_in.shape[0] < im_reshape[0]) | (data_in.shape[1] < im_reshape[1]):
    data_in = caffe.io.resize_image(data_in, im_reshape)
  if im_flip:
    data_in = caffe.io.flip_image(data_in, 1, flow) 
    data_in = data_in[im_crop[0]:im_crop[2], im_crop[1]:im_crop[3], :] 
  return transformer.preprocess('data_in', data_in)

class ImageProcessorCrop(object):
  def __init__(self, transformer):
    self.transformer = transformer
  def __call__(self, im_info):
    return processImageCrop(im_info, self.transformer)

class SequenceGenerator(object):

  def __init__(self, frames):
    self.frames = frames
    self.idx = 0

  def __call__(self):
    return self.frames[self.idx:self.idx+self.batch_size]
    
def advance_batch(result, sequence_generator, image_processor, pool):
  
    label_r, im_info = sequence_generator()
    tmp = image_processor(im_info[0])
    result['data'] = pool.map(image_processor, im_info)
    result['label'] = label_r
    cm = np.ones(len(label_r))
    cm[0::16] = 0
    result['clip_markers'] = cm

class BatchAdvancer():
    def __init__(self, result, sequence_generator, image_processor, pool):
      self.result = result
      self.sequence_generator = sequence_generator
      self.image_processor = image_processor
      self.pool = pool
 
    def __call__(self):
      return advance_batch(self.result, self.sequence_generator, self.image_processor, self.pool)

class SequenceRead(caffe.Layer):

  def initialize(self):
    self.idx      = 0
    self.channels = 3
    self.channels = 1
    self.height   = 1241
    self.width    = 376
    self.scale    = 4

  def setup(self, bottom, top):
    random.seed(10)
    self.initialize()
    
    with open(self.path_to_image_list, 'r') as fd:
      f_lines = fd.readlines()

    self.frames = []
    for ix, line in enumerate(f_lines):
      image_path, label = line.split()
      l = float(label)
      self.frames.append((image_path, l))

    self.reshape = (float(self.height)/self.scale, float(self.width)/self.scale)
    self.crop = (int(self.reshape[0]*.9), int(self.reshape[1]*.9))
    self.reshape = (int(self.reshape[0]), int(self.reshape[1]))

    print('self.reshape:', self.reshape)
    print('self.crop:', self.crop)
    
    shape = (len(self.frames), self.channels, self.height, self.width)
        
    self.transformer = caffe.io.Transformer({'data_in': shape})
    self.transformer.set_raw_scale('data_in', 255)
    image_mean = [103.939, 116.779, 128.68]
    channel_mean = np.zeros((3, self.crop[0], self.crop[1]))
    for channel_index, mean_val in enumerate(image_mean):
      channel_mean[channel_index, ...] = mean_val
    self.transformer.set_mean('data_in', channel_mean)
    self.transformer.set_channel_swap('data_in', (2, 1, 0))
    self.transformer.set_transpose('data_in', (2, 0, 1))

    self.thread_result = {}
    self.thread = None
    pool_size = 24

    self.image_processor = ImageProcessorCrop(self.transformer)
    self.sequence_generator = SequenceGenerator(self.frames)

    self.pool = Pool(processes=pool_size)
    self.batch_advancer = BatchAdvancer(self.thread_result, self.sequence_generator, self.image_processor, self.pool)
    self.dispatch_worker()
    self.top_names = ['data', 'label','clip_markers']
    print 'Outputs:', self.top_names
    if len(top) != len(self.top_names):
      raise Exception('Incorrect number of outputs (expected %d, got %d)' %
                      (len(self.top_names), len(top)))
    self.join_worker()
    for top_index, name in enumerate(self.top_names):
      if name == 'data':
        shape = (self.N, self.channels, self.height, self.width)
      elif name == 'label':
        shape = (self.N,)
      elif name == 'clip_markers':
        shape = (self.N,)
      top[top_index].reshape(*shape)

  def reshape(self, bottom, top):
    pass

  def forward(self, bottom, top):
  
    if self.thread is not None:
      self.join_worker() 

    #rearrange the data: The LSTM takes inputs as [video0_frame0, video1_frame0,...] but the data is currently arranged as [video0_frame0, video0_frame1, ...]
    new_result_data = [None]*len(self.thread_result['data']) 
    new_result_label = [None]*len(self.thread_result['label']) 
    new_result_cm = [None]*len(self.thread_result['clip_markers'])
    for i in range(self.frames):
      for ii in range(self.buffer_size):
        old_idx = ii*self.frames + i
        new_idx = i*self.buffer_size + ii
        new_result_data[new_idx] = self.thread_result['data'][old_idx]
        new_result_label[new_idx] = self.thread_result['label'][old_idx]
        new_result_cm[new_idx] = self.thread_result['clip_markers'][old_idx]

    for top_index, name in zip(range(len(top)), self.top_names):
      if name == 'data':
        for i in range(self.N):
          top[top_index].data[i, ...] = new_result_data[i] 
      elif name == 'label':
        top[top_index].data[...] = new_result_label
      elif name == 'clip_markers':
        top[top_index].data[...] = new_result_cm

    self.dispatch_worker()
      
  def dispatch_worker(self):
    assert self.thread is None
    self.thread = Thread(target=self.batch_advancer)
    self.thread.start()

  def join_worker(self):
    assert self.thread is not None
    self.thread.join()
    self.thread = None

  def backward(self, top, propagate_down, bottom):
    pass

# train/test specializations
class SequenceReadTrain(SequenceRead):

  def initialize(self):
    self.train_or_test = 'train'
    self.frames = train_frames
    self.path_to_image_list = settings.train_images_list
class SequenceReadTest(SequenceRead):

  def initialize(self):
    self.train_or_test = 'test'
    self.frames = test_frames
    self.path_to_image_list = settings.test_images_list
