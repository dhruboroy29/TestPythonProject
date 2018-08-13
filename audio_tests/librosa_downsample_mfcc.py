from librosa import load, resample
from librosa.feature import mfcc
import numpy as np

y, sr = load('Roland-JV-2080-Pick-Bass-C2.wav', sr=44100)
print '################'
print '# Original-44K #'
print '################'
print 'Samples: ', y
print 'Number of samples:', len(y)
#print('sr: ', sr)
mfcc_orig = mfcc(y=y, sr=sr)
print 'MFCC: ', type(mfcc_orig)
print 'MFCC shape: ', np.shape(mfcc_orig)

y_8k = resample(y, sr, 8000)
print '##################'
print '# Downsampled-8K #'
print '##################'
print 'Samples: ', y_8k
print 'Number of samples:', len(y_8k)
#print('sr: ', sr)
mfcc_8k = mfcc(y=y_8k, sr=8000)
print 'MFCC: ', mfcc_8k
print 'MFCC shape: ', np.shape(mfcc_8k)
np.save('Roland-JV-2080-Pick-Bass-C2_mfcc_8k.npy', mfcc_8k)

# Read .npy for test
print '#############'
print '# Test Read #'
print '#############'
mfcc_read = np.load('Roland-JV-2080-Pick-Bass-C2_mfcc_8k.npy')
print(np.shape(mfcc_read))
print(mfcc_read)