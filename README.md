# HDF5 Tools

A toolset for manipulating HDF5 file format

# MODULES
- hdf5loader.py: the module provides a set of classes for loading batches of features+labels from HDF5 files for training purpose
- dict2hdf5.py: saving and loading of hdf5 files into a dictionary. See usage in the test provided inside of the module
- lmdb2hdf5.py: converts LMDB image+labels database into an HDF5 file
- img2hdf5.py: a python script loading images+labels from a directory and saving them into HDF5