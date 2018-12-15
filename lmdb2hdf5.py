#!/usr/bin/env python

"""
The module converts LMDB files containing images into an HDF5 file.
The user must provide two LMDB files: 1st file containing images and 2nd file containing by-pixel labels.
"""
# Notes:
# TensorFlow image dimensions order : [batch, height, width, channels]

import sys
import caffe
import lmdb
import numpy as np
import cv2
from caffe.proto import caffe_pb2
import h5py
import argparse

def lmdbGetSampNum(lmdb_cursor):
    samples_num = 0
    for key, value in lmdb_cursor:
        samples_num += 1
    return samples_num


def lmdbGetDimensions(lmdb_cursor):
    datum = caffe_pb2.Datum()
    for key, value in lmdb_cursor:
        datum.ParseFromString(value)

        label = datum.label
        data = caffe.io.datum_to_array(datum)

        # CxHxW to HxWxC in cv2
        image = np.transpose(data, (1, 2, 0))
        # print 'lmdbGetDimensions: image type = ', type(image), 'Shape of the image: ', image.shape
        break

    return image.shape

def lmdbFillDataset(lmdb_data_cursor, dset, samp_max):
    samp_i=-1
    data_datum = caffe_pb2.Datum()
    for key, value in lmdb_data_cursor:
        samp_i += 1
        if samp_i >= samp_max:
            break

        data_datum.ParseFromString(value)

        label = data_datum.label
        data = caffe.io.datum_to_array(data_datum)

        #CxHxW to HxWxC in cv2
        image = np.transpose(data, (1,2,0))
        dset[samp_i,:,:,:] = image
        print('{},{}'.format(key, label))

def main (argv):
    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "in_dir",
        default='dataset_lmdb',
        help="Directory with the data LMDB"
    )
    parser.add_argument(
        "label_dir",
        default='labels_lmdb',
        help="Directory with the label LMDB"
    )
    parser.add_argument(
        "out_filename",
        default='data.h5',
        help="HDF5 output file"
    )
    parser.add_argument(
        "-v",
        "--visualize",
        action='store_true',
        help="Showing images"
    )
    parser.add_argument(
        "--samples_maxnum",
        default=-1,
        type=int,
        help="Maximum number of samples"
    )
    args = parser.parse_args()

    data_dir = args.in_dir
    label_dir = args.label_dir
    h5filename = args.out_filename
    show_img = args.visualize
    samp_max = args.samples_maxnum

    # Parameters
    train_val_test_ratio = np.array( [0.8, 0.08, 0.12])
    #Renormalize ratios
    train_val_test_ratio /= train_val_test_ratio.sum()

    lbl_img_scale = 50
    crossval_num = 1

    ###########################################################
    # Opening images
    lmdb_data_env = lmdb.open(data_dir)
    lmdb_data_txn = lmdb_data_env.begin()
    lmdb_data_cursor = lmdb_data_txn.cursor()
    data_datum = caffe_pb2.Datum()

    # Opening labels
    lmdb_lbl_env = lmdb.open(label_dir)
    lmdb_lbl_txn = lmdb_lbl_env.begin()
    lmdb_lbl_cursor = lmdb_lbl_txn.cursor()
    lbl_datum = caffe_pb2.Datum()

    #Getting dimensionality of the datasets
    data_samples_num = lmdbGetSampNum(lmdb_data_cursor)
    data_img_shape = lmdbGetDimensions(lmdb_data_cursor)

    lbl_samples_num = lmdbGetSampNum(lmdb_lbl_cursor)
    lbl_img_shape = lmdbGetDimensions(lmdb_lbl_cursor)

    ## We must have a single variable for the number of samples that we use
    # samples_num = data_samples_num
    if samp_max <= 0:
        samples_num = data_samples_num
        samp_max = samples_num
    else:
        samples_num = samp_max

    print 'Images num  = ', data_samples_num, '  Labels num = ', lbl_samples_num
    print 'Data Img size (h,w,c) = ', data_img_shape, ' Label Img size (h,w,c) = ', lbl_img_shape

    ###########################################################
    ## Filling HDF5 file
    # Opening h5py file
    h5datafile = h5py.File(h5filename, "w")
    feat_dset = h5datafile.create_dataset("/feat/img",  (samples_num,) + data_img_shape, dtype='uint8')
    lbl_dset  = h5datafile.create_dataset("/label/img", (samples_num,) + lbl_img_shape, dtype='uint8')

    train_size = int(train_val_test_ratio[0] * samples_num)
    val_size   = max( int(train_val_test_ratio[1] * samples_num), 1)
    test_size = samples_num - train_size - val_size

    indx_train_dset  = h5datafile.create_dataset("/crossval_indx/0/train", (train_size,), dtype='uint')
    indx_val_dset  = h5datafile.create_dataset("/crossval_indx/0/val", (val_size,), dtype='uint')
    indx_test_dset = h5datafile.create_dataset("/crossval_indx/0/test", (test_size,), dtype='uint')

    crossval_names_dset = h5datafile.create_dataset("/crossval_names", (crossval_num,), dtype='S2')
    crossval_names_dset[0] = '0'

    # Filling hdf5 file with data
    lmdbFillDataset(lmdb_data_cursor, feat_dset, samp_max)
    lmdbFillDataset(lmdb_lbl_cursor, lbl_dset, samp_max)

    # Creating train/val/test indices
    remain_indx_set = np.array(range(samples_num))

    train_indx_set = np.random.choice(remain_indx_set, size=train_size, replace=False)
    remain_indx_set = np.setdiff1d(remain_indx_set, train_indx_set)

    val_indx_set = np.random.choice(remain_indx_set, size=val_size, replace=False)
    remain_indx_set = np.setdiff1d(remain_indx_set, val_indx_set)

    test_indx_set = remain_indx_set
    remain_indx_set = np.setdiff1d(remain_indx_set, test_indx_set)

    #Writing generated index sets into the hdf5 file
    indx_train_dset[:] = train_indx_set
    indx_val_dset[:]   = val_indx_set
    indx_test_dset[:]  = test_indx_set

    print 'Training indices = ', indx_train_dset[:]
    print 'Val indices = ',  indx_val_dset[:]
    print 'Test indices = ', indx_test_dset[:]

    h5datafile.close()

    if not show_img:
        sys.exit()

    ###########################################################
    ## Reading and showing the file that we created
    h5datafile = h5py.File(h5filename, "r")
    feat_dset = h5datafile.get("/feat/img")
    lbl_dset  = h5datafile.get("/label/img")

    # Showing images that we got
    for samp_i in range(0, samples_num):
        data_img = feat_dset[samp_i, :,:,:]
        lbl_img =  lbl_dset[samp_i, :,:,:]

        print 'image type = ', type(data_img), 'Shape of the image: ', data_img.shape
        print 'image type = ', type(lbl_img), 'Shape of the image: ', lbl_img.shape

        lbl_image_scaled = lbl_img_scale * lbl_img
        cv2.imshow('data_img', data_img)
        cv2.imshow('lbl_img', lbl_image_scaled)
        cv2.waitKey(0)


    h5datafile.close()


if __name__ == "__main__":
    main(sys.argv)