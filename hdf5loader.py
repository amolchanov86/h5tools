#!/usr/bin/env python

## @package hdf5loader
# HDF5 loader for training machine learning algorithms
# Internal structure of HD5 files should be:
# Features and labels:
#   /feat - group containing different features
#   /label - group containing different labels
# Crossvalidation indices (for consistency)
#   /crossval_indx/[index]/train
#   /crossval_indx/[index]/val
#   /crossval_indx/[index]/test
# Names for particular crossvalidations (in case they have semantic meaning, like testing on different unseen objects)
#   /crossval_names

# Notes:
# TensorFlow image dimensions order : [batch, height, width, channels]

import sys
import getopt

import h5py
import numpy as np
from random import shuffle


## indxClass
class trainIndxClass:
    def __init__(self, name, train, val, test):
        self.name = name
        self.train = np.array(train)
        self.val = np.array(val)
        self.test = np.array(test)

        self.train = self.train.flatten()
        self.val   = self.val.flatten()
        self.test  = self.test.flatten()

        #Sorting since hdf5 does not support shuffled index access
        #i.e. all indices must be in ascending order, proably for efficient access
        np.ndarray.sort(self.train)
        np.ndarray.sort(self.val)
        np.ndarray.sort(self.test)


## pokeHDF5load
# loads data specific for the point of contact localization projectobject
class HDF5load:
    ## Member variables that can be read and modified directly
    # batch_size - default size of the retrieved batch

    ## Member variables that can be read but NOT MODIFIED (see special function members)
    # train_indx, val_indx, test_indx - set of currently used indices
    # nxtIndex - next index that will start a new batch
    # feat_names  - names of features from hdf5 file
    # label_names - names of labels from hdf5 file
    # crossval_indx  - a list of vectors of all crossvalidation indices
    # crossval_names - a list of all crossvalidation names

    ## Constructor
    # @param filename path/name for the hdf5 file
    def __init__(self, filename=None):
        self.f = h5py.File(filename, "r")

        #Reading indices for individual crossvalidations
        crossval_indx = self.f['/crossval_indx']
        print 'Indices found: '
        self.crossval_indx = []
        self.crossval_names = {}
        for indx_name in crossval_indx:
            count = int(indx_name) - 1
            self.crossval_names[self.f['/crossval_names'][count]] = count
            print 'Index = ', indx_name, ' Name = ', self.f['/crossval_names'][count]
            self.crossval_indx.append(
                trainIndxClass (
                self.f['/crossval_names'][count],
                self.f['/crossval_indx/' + indx_name + '/train'],
                self.f['/crossval_indx/' + indx_name + '/val'],
                self.f['/crossval_indx/' + indx_name + '/test']))

        # Getting feature names
        self.getFeatLabelNames()

        # Creating indices and epoch objects
        # I make them np.arrays to make them mutable to be able to use references (aliases) for abstraction
        # i.e. I use self.epoch / self.nxt_index to abstract from particular container
        self.epoch_train = np.array([0])
        self.nxt_index_train = np.array([0])
        self.epoch_end_train = np.array([0])
        self.epoch_val = np.array([0])
        self.nxt_index_val = np.array([0])
        self.epoch_end_val = np.array([0])
        self.epoch_test = np.array([0])
        self.nxt_index_test = np.array([0])
        self.epoch_end_test = np.array([0])
        self.epoch_all = np.array([0])
        self.nxt_index_all = np.array([0])
        self.epoch_end_all = np.array([0])

        # Reset current internal state for batch extraction and set 0 index set
        self.setCrossval(0, mode='train')

        # Set batch size
        self.batch_size = 32

        return

    ## Function sets the current mode (train/val/test) to make the object iterate through appropriate set
    # The function creates important aliases
    def setMode(self, mode='train', reset=False):
        mode = mode.lower()
        if mode == 'train':
            self.mode_cur = mode
            self.epoch = self.epoch_train
            self.nxt_index = self.nxt_index_train
            self.mode_indx = self.train_indx
            self.epoch_end = self.epoch_end_train
        elif mode == 'val':
            self.mode_cur = mode
            self.epoch = self.epoch_val
            self.nxt_index = self.nxt_index_val
            self.mode_indx = self.val_indx
            self.epoch_end = self.epoch_end_val
        elif mode == 'test':
            self.mode_cur = mode
            self.epoch = self.epoch_test
            self.nxt_index = self.nxt_index_test
            self.mode_indx = self.test_indx
            self.epoch_end = self.epoch_end_test
        elif mode == 'all':
            self.mode_cur = mode
            self.epoch = self.epoch_all
            self.nxt_index = self.nxt_index_all
            self.mode_indx = self.all_indx
            self.epoch_end = self.epoch_end_all

    ## Function returns the current mode (train/val/test)
    def getMode(self):
        return self.mode_cur

    ## Operations with epochs
    def getEpoch(self):
        return self.epoch[0]

    ## Function resets internal pointers for indices
    def resetIndx(self):
        self.epoch_train[0] = 0
        self.epoch_end_train[0] = 0
        self.nxt_index_train[0] = 0
        self.epoch_val[0] = 0
        self.epoch_end_val[0] = 0
        self.nxt_index_val[0] = 0
        self.epoch_test[0] = 0
        self.nxt_index_test[0] = 0
        self.epoch_end_test[0] = 0
        self.epoch_all[0] = 0
        self.nxt_index_all[0] = 0
        self.epoch_end_all[0] = 0

    ## Function sets crossvalidation set by indx (if provided numerical value) or name (if semantic name is known)
    def setCrossval(self, name, mode=None):
        if isinstance(name, str):
            self.cur_crossindx = self.crossval_names[name]
        else:
            self.cur_crossindx = name

        # Reset epochs and indices
        self.resetIndx()

        # self.train_indx = shuffle(  self.crossval_indx[self.cur_crossindx].train )
        self.train_indx = self.crossval_indx[self.cur_crossindx].train
        self.train_indx = np.ndarray.astype(self.train_indx, int)
        self.val_indx   = self.crossval_indx[self.cur_crossindx].val
        self.test_indx  = self.crossval_indx[self.cur_crossindx].test
        self.all_indx   = np.array(range(0, self.getSampNum() ))
        self.all_indx   = self.all_indx.flatten()

        if mode == None:
            self.setMode(self.mode_cur)
        else:
            self.setMode(mode)

    ## Shuffle current indices
    def indxShuffle(self):
        #shuffling in-place to preserve references
        np.random.shuffle(self.mode_indx)

    ## Order current indices
    def indxOrder(self):
        #sorting in-place to preserve references
        np.ndarray.sort(self.mode_indx)


    ## Get feature and label names
    def getFeatLabelNames(self):
        self.feat_names  = self.f['/feat'].keys()
        self.label_names = self.f['/label'].keys()
        return self.feat_names, self.label_names

    ## Function returns features and data given set of indices
    # the function should be overloaded in a specific class to extract proper feature sets
    def getData(self, indices):
        feat  = self.f['/feat/'  + self.feat_names[0] ][indices, :]
        label = self.f['/label/' + self.label_names[0] ][indices, :]
        return feat, label

    ## Getting feature/label shapes
    # there is a chance one would have to change these too in case
    # overloaded getData would return not a numpy array
    def getShapes(self):
        feat_sample, label_sample = self.getData([1])
        return feat_sample.shape, label_sample.shape

    ## Recover number of samples present (overal number)
    # overload this function in case you have different dimension for indices of samples
    def getSampNum(self):
        # print 'Feat names = ', self.feat_names, ' type = ', type(self.feat_names)
        feat = self.f['/feat/' + self.feat_names[0]]
        return feat.shape[0]

    ## Get training examples num
    def getTrainSamplesNum(self):
        return self.train_indx.size

    ## Get validation examples num
    def getValSamplesNum(self):
        return self.val_indx.size

    ## Get test examples num
    def getTestSamplesNum(self):
        return self.test_indx.size

    ## The function returns the next training batch
    def nxtBatchTrain(self, batch_size):
        start_indx = self.nxt_index_train[0]
        end_indx = start_indx + batch_size
        epoch_changed = 0
        if end_indx >= self.train_indx.size:
            end_indx = self.train_indx.size
            self.nxt_index_train[0] = 0
            self.epoch_train[0] += 1
            epoch_changed = 1
        else:
            self.nxt_index[0] = end_indx
        return (self.getData( self.train_indx[start_indx:end_indx] )), epoch_changed

    ## The function returns the next validation batch
    def nxtBatchVal(self, batch_size):
        start_indx = self.nxt_index_val[0]
        end_indx = start_indx + batch_size
        epoch_changed = 0
        if end_indx >= self.val_indx.size:
            end_indx = self.val_indx.size
            self.nxt_index_val[0] = 0
            self.epoch_val[0] += 1
            epoch_changed = 1
        else:
            self.nxt_index[0] = end_indx
        return (self.getData( self.val_indx[start_indx:end_indx] )), epoch_changed

    ## The function returns the next test batch
    def nxtBatchTest(self, batch_size):
        start_indx = self.nxt_index_test[0]
        end_indx = start_indx + batch_size
        epoch_changed = 0
        if end_indx >= self.test_indx.size:
            end_indx = self.test_indx.size
            self.nxt_index_test[0] = 0
            self.epoch_test[0] += 1
            epoch_changed = 1
        else:
            self.nxt_index[0] = end_indx
        return (self.getData( self.test_indx[start_indx:end_indx] )), epoch_changed

    ## The function returns the next training batch
    def nxtBatch(self, batch_size):
        start_indx = self.nxt_index[0]
        end_indx = start_indx + batch_size
        epoch_changed = 0
        if end_indx >= self.mode_indx.size:
            end_indx = self.mode_indx.size
            self.nxt_index[0] = 0
            self.epoch[0] += 1
            epoch_changed = 1
        else:
            self.nxt_index[0] = end_indx
        print 'start_indx = ', start_indx, ' end_indx = ', end_indx, ' epoch = ', self.epoch[0], ' epoch_changed = ', epoch_changed
        return (self.getData( self.mode_indx[start_indx:end_indx] )), epoch_changed

    ## Functions to make this class iterable
    def __iter__(self):
        return self

    def next(self):
        if self.epoch_end[0]:
            self.epoch_end[0] = 0
            raise StopIteration
        feat_label, self.epoch_end[0] = self.nxtBatch(self.batch_size)
        return feat_label[0], feat_label[1]

    ## Closing the internal HDF5 file
    def close(self):
        self.f.close()


class pokeHDF5load(HDF5load):
    ## Function returns features and data given set of indices
    # the function should be overloaded in a specific class to extract proper feature sets
    def getData(self, indices):
        feat  = self.f['/feat/'  + self.feat_names[0]  ][indices, :]
        label = self.f['/label/' + self.label_names[0] ][indices, :]
        return feat, label

## Class for training knowledge transfer of image segmentation
class imgtransferHDF5load(HDF5load):
    ## Function returns features and data given set of indices
    # the function should be overloaded in a specific class to extract proper feature sets
    def getData(self, indices):
        feat  = self.f['/feat/img'][indices, :]
        label = self.f['/label/hardlabel'][indices, :]
        softlabel = self.f['/label/softlabel'][indices, :]

        if len(indices) == 1:
            feat = np.expand_dims(feat, axis=0)
            label = np.expand_dims(label, axis=0)
            softlabel = np.expand_dims(softlabel, axis=0)

        return feat, (label,softlabel)

    ## Getting feature/label shapes
    # there is a chance one would have to change these too in case
    # overloaded getData would return not a numpy array
    def getShapes(self):
        feat_sample, label_sample = self.getData([1])
        return feat_sample.shape, label_sample[0].shape, label_sample[1].shape

def main(argv=None):
    dataset = pokeHDF5load('poke_alldata.h5')

    for cross_i in range(0,2):
        dataset.setCrossval(cross_i)
        print '--------------------------'
        print 'dataset indx changed to ', cross_i
        print 'train_indx:', dataset.train_indx, 'type = ', type(dataset.train_indx)
        print 'val_indx:', dataset.val_indx
        print 'test_indx', dataset.test_indx

    # for batch_i in range(0,10):
    #     print 'Batch = ', batch_i
    #     batch = dataset.nxtBatch(100)
    #     print batch

if __name__ == "__main__":
    main()