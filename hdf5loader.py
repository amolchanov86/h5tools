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


## pokeHDF5load
# loads data specific for the point of contact localization projectobject
class HDF5load:
    ## Constructor
    # \return number of crossvalidations found
    # \param filename path/name for the hdf5 file
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

        # GEtting feature names
        self.getFeatLabelNames()

        # Reset current internal state for batch extraction and set 0 index set
        self.setCrossval(0)

        # Set batch size
        self.batch_size = 32

        return

    ## Function sets crossvalidation set by indx (if provided numerical value) or name (if semantic name is known)
    def setCrossval(self, name):
        if isinstance(name, str):
            self.cur_crossindx = self.crossval_names[name]
        else:
            self.cur_crossindx = name
        self.epoch = 0
        self.nxtIndex = 0
        # self.train_indx = shuffle(  self.crossval_indx[self.cur_crossindx].train )
        self.train_indx = self.crossval_indx[self.cur_crossindx].train
        self.train_indx = np.ndarray.astype(self.train_indx, int)
        self.val_indx   = self.crossval_indx[self.cur_crossindx].val
        self.test_indx  = self.crossval_indx[self.cur_crossindx].test

    ## Get feature and label names
    def getFeatLabelNames(self):
        self.feat_names  = self.f['/feat']
        self.label_names = self.f['/label']
        return self.feat_names, self.label_names

    ## Function returns features and data given set of indices
    # the function should be overloaded in a specific class to extract proper feature sets
    def getData(self, indices):
        feat  = self.f['/feat/'  + self.feat_names[0] ][indices, :]
        label = self.f['/label/' + self.label_names[0] ][indices, :]
        return feat, label

    ## The function returns the next batch
    def nxtBatch(self, batch_size):
        start_indx = self.nxtIndex
        end_indx = start_indx + batch_size - 1
        epoch_changed = 0
        if end_indx >= (self.train_indx.size - 1):
            end_indx = self.train_indx.size - 1
            self.nxtIndex = 0
            self.epoch = self.epoch + 1
            epoch_changed = 1
        else:
            self.nxtIndex = end_indx + 1
        return self.getData( self.train_indx[start_indx:end_indx] ), epoch_changed

    def nxtBatch(self):
        return self.nxtBatch(self.batch_size)

    ## Functions to make this class iterable
    def __iter__(self):
        return self

    def next(self):
        feat, label, epoch_end = self.nxtBatch()
        if not epoch_end:
            return feat, label
        else:
            raise StopIteration


class pokeHDF5load(HDF5load):
    ## Function returns features and data given set of indices
    # the function should be overloaded in a specific class to extract proper feature sets
    def getData(self, indices):
        feat  = self.f['/feat/'  + self.feat_names[0]  ][indices, :]
        label = self.f['/label/' + self.label_names[0] ][indices, :]
        return feat, label


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