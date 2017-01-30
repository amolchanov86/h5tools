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
import argparse

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
class HDF5load(object):
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
    def __init__(self, filename=None, crossval_indx_set=0):
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

        # Initialize extra variables (extend it in cheldren's class)
        self.initVars()

        # Reset current internal state for batch extraction and set 0 index set
        self.setCrossval(crossval_indx_set, mode='train')

        # Set batch size
        self.batch_size = 32

        # Some auxiliary variables
        self.last_sampindx = [] # last index set of samples requested

        return

    ## Function for extensions (to initialize some additional variables added by children
    def initVars(self):
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
        # print 'Mode config: '
        # print ' mode_cur = ', self.mode_cur
        # print ' epoch = ', self.epoch
        # print ' nxt_indx = ', self.nxt_index
        # print ' mode_indx = ', self.mode_indx
        # print ' epoch_end = ', self.epoch_end

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

        # self.train_indx = shuffle(  self.crossval_indx[self.cur_crossindx].train )
        self.train_indx = self.crossval_indx[self.cur_crossindx].train
        self.train_indx = np.ndarray.astype(self.train_indx, int)
        self.val_indx   = self.crossval_indx[self.cur_crossindx].val
        self.test_indx  = self.crossval_indx[self.cur_crossindx].test
        self.all_indx   = np.array(range(0, self.getSampNum() ))
        self.all_indx   = self.all_indx.flatten()

        print 'H5 loader: Crossval name = ', self.getCrossvalName()

        # Reset epochs and indices
        self.resetIndx()

        if mode == None:
            self.setMode(self.mode_cur)
        else:
            self.setMode(mode)

    ## Get crossval name
    def getCrossvalName(self):
        return self.f['/crossval_names'][self.cur_crossindx]

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
        print 'epoch = ', self.epoch[0], 'start_indx = ', start_indx, ' end_indx = ', end_indx, ' epoch_changed = ', epoch_changed
        self.last_sampindx = self.mode_indx[start_indx:end_indx].tolist()
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


## The same as HDF5load, but
# when extracting batches it randomizes indices in the batch
# NOTES:
# self.nxt_index will not be changed in this class
class HDF5loadRandom(HDF5load):
    ## Function sets crossvalidation set by indx (if provided numerical value) or name (if semantic name is known)
    def setCrossval(self, name, mode=None):
        if isinstance(name, str):
            self.cur_crossindx = self.crossval_names[name]
        else:
            self.cur_crossindx = name

        print 'H5 loader: Crossval name = ', self.getCrossvalName()

        self.train_indx = self.crossval_indx[self.cur_crossindx].train
        self.train_indx = np.ndarray.astype(self.train_indx, int)
        self.val_indx   = self.crossval_indx[self.cur_crossindx].val
        self.test_indx  = self.crossval_indx[self.cur_crossindx].test
        self.all_indx   = np.array(range(0, self.getSampNum() ))
        self.all_indx   = self.all_indx.flatten()

        self.remain_indx_train = [self.train_indx.copy()]
        self.remain_indx_val   = [self.val_indx.copy()]
        self.remain_indx_test  = [self.test_indx.copy()]
        self.remain_indx_all   = [self.all_indx.copy()]

        # Reset epochs and indices
        self.resetIndx()

        if mode == None:
            self.setMode(self.mode_cur)
        else:
            self.setMode(mode)


    ## Function sets the current mode (train/val/test) to make the object iterate through appropriate set
    # The function creates important aliases
    def setMode(self, mode='train', reset=False):
        HDF5load.setMode(self, mode=mode, reset=reset)
        mode = mode.lower()

        # print 'H5 loader: switched to mode ', mode
        if mode == 'train':
            self.remain_indx = self.remain_indx_train
        elif mode == 'val':
            self.remain_indx = self.remain_indx_val
        elif mode == 'test':
            self.remain_indx = self.remain_indx_test
        elif mode == 'all':
            self.remain_indx = self.remain_indx_all
        # print 'H5 loader: indx set = ', self.remain_indx

    ## The function returns the next training batch
    def nxtBatch(self, batch_size):
        epoch_changed = 0
        population_size = self.remain_indx[0].size

        if batch_size >= population_size:
            epoch_changed = 1
            batch_indices = self.remain_indx[0]
            # Refill the population of indices
            self.remain_indx[0] = self.mode_indx.copy()
            batch_size = population_size
            self.epoch[0] += 1
        else:
            batch_indices  = np.random.choice(self.remain_indx[0], size=batch_size, replace=False)
            batch_indices = np.sort(batch_indices)
            self.remain_indx[0] = np.setdiff1d(self.remain_indx[0], batch_indices)
        batch_indices = batch_indices.tolist()
        self.last_sampindx = batch_indices
        return (self.getData(batch_indices)), epoch_changed

    def resetIndx(self):
        HDF5load.resetIndx(self)
        self.remain_indx_train[0] = self.train_indx.copy()
        self.remain_indx_val[0]   = self.val_indx.copy()
        self.remain_indx_test[0]  = self.test_indx.copy()
        self.remain_indx_all[0]   = self.all_indx.copy()

# --------------------------------------------------------------------------------------------------
## Wrapping MNIST
class MNISTload(object):
    def __init__(self, filename = None, crossval_indx_set = None):
        from tensorflow.examples.tutorials.mnist import input_data
        self.mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


    @property
    def epoch(self):
        """
        Sample statistics/diagnostics.
        :return: (List) list of names of diagnostics calculated per sample
        """
        return self.dataset_cur.epochs_completed

    def setMode(self, mode):
        if mode == 'train':
            self.dataset_cur = self.mnist.train
        elif mode == 'val':
            self.dataset_cur = self.mnist.validation
        elif mode == 'test':
            self.dataset_cur = self.mnist.test
        elif mode == 'all':
            print '!!!!!!!!!!!!!!!! WARNING: MNIST ALL = MNIST TRAIN'
            self.dataset_cur = self.mnist.train

    def nextBatch(self, batch_size):
        batch = self.dataset_cur.next_batch(batch_size)
        self.last_indices_requested \
            = range(self.dataset_cur._index_in_epoch, self.dataset_cur._index_in_epoch + batch_size)
        return np.reshape(batch[0], newshape=[-1,28,28,1]), batch[1]

    def getShapes(self):
        return [28,28,1], [1]

    def getTrainSamplesNum(self):
        return self.mnist.train.num_examples

    def getValSamplesNum(self):
        return self.mnist.validation.num_examples

    def getTestSamplesNum(self):
        return self.mnist.test.num_examples

    def getSampNum(self):
        return self.getTestSamplesNum + self.getValSamplesNum() + self.getTrainSamplesNum()

    def resetIndx(self):
        self.print_notimplemented()
        return None

    def getOutNum(self):
        return 10

    def getLabelType(self):
        return type(0)

    def getLabelVal(self):
        return range(0,10)

    def print_notimplemented(self):
        print '!!!!!!!!!!!!!!!! WARNING: NOT IMPLEMENTED'

#--------------------------------------------------------------------------------------------------
## Class for training knowledge transfer of image segmentation
class imgtransferHDF5load(HDF5loadRandom):
    ## Function returns features and data given set of indices
    # the function should be overloaded in a specific class to extract proper feature sets
    def getData(self, indices):
        feat  = self.f['/feat/img'][indices, :]
        label = self.f['/label/hardlabel'][indices, :]
        softlabel = self.f['/label/softlabel'][indices, :]

        if len(feat.shape) < 4:
            feat = np.expand_dims(feat, axis=0)
            label = np.expand_dims(label, axis=0)
            softlabel = np.expand_dims(softlabel, axis=0)
            # print 'Indices = ', indices, ' Shapes = ', feat.shape, label.shape, softlabel.shape

        return feat, (label,softlabel)

    ## Getting feature/label shapes
    # there is a chance one would have to change these too in case
    # overloaded getData would return not a numpy array
    def getShapes(self):
        feat_sample, label_sample = self.getData([1])
        return feat_sample.shape, label_sample[0].shape, label_sample[1].shape

## Class for training knowledge transfer of image segmentation
class imgTransfLogitHDF5load(HDF5loadRandom):
    ## Function returns features and data given set of indices
    # the function should be overloaded in a specific class to extract proper feature sets
    def getData(self, indices):
        feat = self.f['/feat/img'][indices, :]
        label = self.f['/label/hardlabel'][indices, :]
        softlabel = self.f['/label/logit'][indices, :]

        if len(feat.shape) < 4:
            feat = np.expand_dims(feat, axis=0)
            label = np.expand_dims(label, axis=0)
            softlabel = np.expand_dims(softlabel, axis=0)
            # print 'Indices = ', indices, ' Shapes = ', feat.shape, label.shape, softlabel.shape

        return feat, (label, softlabel)

    ## Getting feature/label shapes
    # there is a chance one would have to change these too in case
    # overloaded getData would return not a numpy array
    def getShapes(self):
        feat_sample, label_sample = self.getData([1])
        return feat_sample.shape, label_sample[0].shape, label_sample[1].shape

## Class for training knowledge transfer of image segmentation
class intFeatHDF5load(HDF5load):
    ## Function returns features and data given set of indices
    # the function should be overloaded in a specific class to extract proper feature sets
    def getData(self, indices):
        feat = self.f['/feat/img'][indices, :]
        label = self.f['/label/hardlabel'][indices, :]
        softlabel = self.f['/label/logit'][indices, :]
        intfeat1 = self.f['/label/squeeze2'][indices, :]
        intfeat2 = self.f['/label/squeeze4'][indices, :]
        intfeat3 = self.f['/label/squeeze6'][indices, :]

        if len(feat.shape) < 4:
            feat = np.expand_dims(feat, axis=0)
            label = np.expand_dims(label, axis=0)
            softlabel = np.expand_dims(softlabel, axis=0)
            intfeat1 = np.expand_dims(intfeat1, axis=0)
            intfeat2 = np.expand_dims(intfeat2, axis=0)
            intfeat3 = np.expand_dims(intfeat3, axis=0)
            # print 'Indices = ', indices, ' Shapes = ', feat.shape, label.shape, softlabel.shape
        # print "Shapes: int1 =", intfeat1.shape, ' int2 = ', intfeat2.shape, ' int3 = ', intfeat3.shape

        return feat, (label, softlabel, intfeat1, intfeat2, intfeat3)

    def getShapes(self):
        feat_sample, label_sample  = self.getData([1])
        return feat_sample.shape, label_sample[0].shape, label_sample[1].shape


## Main function to test functionality
def main(argv=None):
    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "-i", "--in_file",
        default="data.h5",
        help="dataset filename"
    )

    args = parser.parse_args()
    dataset = HDF5loadRandom(args.in_file)
    dataset.batch_size = 1

    # for cross_i in range(0,2):
    #     dataset.setCrossval(cross_i)
    cross_i = 0
    print '--------------------------'
    print 'dataset indx changed to ', cross_i
    print 'train_indx:', dataset.train_indx, 'type = ', type(dataset.train_indx)
    print 'val_indx:', dataset.val_indx
    print 'test_indx', dataset.test_indx


    change_mode_every = 10
    iter = -1
    modes = ['train', 'val', 'test']
    mode_indx = -1
    while dataset.epoch_train[0] < 10:
        for data,label in dataset:
            iter += 1
            print 'remain_indices = ', dataset.remain_indx
            if iter % change_mode_every == 0:
                print '______________________________________________________________________________'
                mode_indx += 1
                print 'Old_mode =', dataset.getMode(), ' remain_indices', dataset.remain_indx
                dataset.setMode(modes[mode_indx % 3])
                print 'New_mode =', dataset.getMode(), ' remain_indices', dataset.remain_indx
                print '______________________________________________________________________________'


if __name__ == "__main__":
    main()