#!/usr/bin/env python

"""
THe module saves images + labels into HDF5 file
"""

# Notes:
# TensorFlow image dimensions order : [batch, height, width, channels]

import sys
import numpy as np
import cv2
import h5py
import argparse

# My modules
import fileutils as fu # See: https://github.com/amolchanov86/fileutils 


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
        "--extensions",
        default='jpg,png,JPG,jpeg,bmp',
        help="List of image extensions to search for"
    )
    parser.add_argument(
        "--out_filename",
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

    h5filename = args.out_filename
    show_img = args.visualize
    samp_max = args.samples_maxnum

    ###########################################################
    ## Parameters
    train_val_test_ratio = np.array( [0.8, 0.08, 0.12])
    #Renormalize ratios
    train_val_test_ratio /= train_val_test_ratio.sum()

    lbl_img_scale = 50
    crossval_num = 1

    # Get file extensions
    extensions = args.extensions.split(',')
    img_list = fu.find_files_ext(args.in_dir, args.extensions)
    label_list = fu.find_files_ext(args.label_dir, args.extensions)

    print img_list
    print label_list

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