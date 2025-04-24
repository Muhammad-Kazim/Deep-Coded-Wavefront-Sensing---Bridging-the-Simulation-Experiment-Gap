import pickle
import os
import numpy as np


def load_pkl(filename):
    """Reads in a geometry pickled object.

    Args:
        filename (str): path/to/file.pkl

    Returns:
        geometry: geometry object
    """
    
    if os.path.isfile(filename):
        print('Loading geometry object...')
        with open(filename, 'rb') as inp:
            geom = pickle.load(inp)
        
        return geom
    else:
        print('File does not exist.')
        

def normalization(field, totype='int16'):
    """normalizes field to 0-1.

    Args:
        field (float): 2d floating point arrays.
        totype (str): 'int16' or 'int8'
    """
    
    field = (field - field.min())/(field.max() - field.min())
    
    if totype == 'int16':
        return np.array(field*(2**16) - 1, dtype=np.uint16)
    elif totype == 'int8':
        return np.array(field*(2**8) - 1, dtype=np.uint16)
    else:
        print('Wrong totype')