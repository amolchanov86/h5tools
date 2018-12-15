#!/usr/bin/env python

import numpy as np
import h5py
import os

class dict2h5(object):
    """
    The class can save and load python dictionary into a hdf5 file
    """

    ## Saving a dictionary
    @classmethod
    def save(cls, dic, filename=None, h5file=None):
        ## The following check is not necessary
        # if os.path.exists(filename):
        #     raise ValueError('File %s exists, will not overwrite.' % filename)
        if filename is None:
            cls.__recursively_save_dict_contents_to_group__(h5file, '/', dic)
        else:
            with h5py.File(filename, 'a') as h5file:
                cls.__recursively_save_dict_contents_to_group__(h5file, '/', dic)



    @classmethod
    def __recursively_save_dict_contents_to_group__(cls, h5file, path, dic):
        # argument type checking
        if not isinstance(dic, dict):
            raise ValueError("must provide a dictionary")
        if not isinstance(path, str):
            raise ValueError("path must be a string")
        if not isinstance(h5file, h5py._hl.files.File):
            raise ValueError("must be an open h5py file")
        # save items to the hdf5 file
        for key, item in dic.items():
            if not isinstance(key, str):
                raise ValueError("dict keys must be strings to save to hdf5")
            # save strings, numpy.int64, and numpy.float64 types
            if isinstance(item, (np.int64, np.float64, str)):
                h5file[path + key] = item
                if not h5file[path + key].value == item:
                    raise ValueError('The data representation in the HDF5 file does not match the original dict.')
            # save numpy arrays
            elif isinstance(item, np.ndarray):
                h5file[path + key] = item
                if not np.array_equal(h5file[path + key].value, item):
                    raise ValueError('The data representation in the HDF5 file does not match the original dict.')
            # save dictionaries
            elif isinstance(item, dict):
                cls.__recursively_save_dict_contents_to_group__(h5file, path + key + '/', item)
            # attempt to convert to a numpy array
            else:
                h5file[path + key] = np.asarray(item)
                if not np.array_equal(h5file[path + key].value, item):
                    raise ValueError('The data representation in the HDF5 file does not match the original dict.')
            # other types cannot be saved and will result in an error
            # else:
            #     raise ValueError('Cannot save %s type.' % type(item))

    ## Load a hdf5 file
    @classmethod
    def load(cls, filename):
        """
        Loads HDF5 content into a dictionary
        """
        with h5py.File(filename, 'r') as h5file:
            return cls.__recursively_load_dict_contents_from_group__(h5file, '/')

    @classmethod
    def __recursively_load_dict_contents_from_group__(cls, h5file, path):
        """
        A helper function to for recursive loading into a dictionary
        """
        ans = {}
        for key, item in h5file[path].items():
            if isinstance(item, h5py._hl.dataset.Dataset):
                ans[key] = item.value
            elif isinstance(item, h5py._hl.group.Group):
                ans[key] = cls.__recursively_load_dict_contents_from_group__(h5file, path + key + '/')
        return ans

## Test
if __name__ == "__main__":

    filename = 'foo.hdf5'
    if os.path.exists(filename):
        os.remove(filename)
    ex = {
        'name': 'stefan',
        'age':  np.int64(24),
        'fav_numbers': np.array([2,4,4.3]),
        'fav_tensors': {
            'levi_civita3d': np.array([
                [[0,0,0],[0,0,1],[0,-1,0]],
                [[0,0,-1],[0,0,0],[1,0,0]],
                [[0,1,0],[-1,0,0],[0,0,0]]
            ]),
            'kronecker2d': np.identity(3)
        }
    }
    print ex
    dict2h5.save(ex, filename)
    loaded = dict2h5.load('foo.hdf5')
    print loaded
    np.testing.assert_equal(loaded, ex)
    print 'check passed!'
    ex2 = {
        'name2': 'stefan',
        'age2':  np.int64(24),
        'fav_numbers2': np.array([2,4,4.3]),
        'fav_tensors2': {
            'levi_civita3d': np.array([
                [[0,0,0],[0,0,1],[0,-1,0]],
                [[0,0,-1],[0,0,0],[1,0,0]],
                [[0,1,0],[-1,0,0],[0,0,0]]
            ]),
            'kronecker2d': np.identity(3)
        }
    }
    dict2h5.save(ex2, filename)
