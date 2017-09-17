import numpy
import scipy.io as sio

def our_function(text):
    print('Test line')
    print('%s %f' % (text, numpy.nan))

def array_func(arr):
    print('Array: ', arr)
    print('Mean: ', numpy.mean(arr))
    print('Median: ', numpy.median(arr))

def matrix_func(matfile):
    matrix = sio.loadmat(matfile)
    print('Passed matrix: ', matrix)
    print('Inspecting matrix: ', sio.whosmat(matfile))
    print('Matrix mean: ', numpy.mean(matrix['mat']))
    print('Matrix median: ', numpy.median(matrix['mat']))
