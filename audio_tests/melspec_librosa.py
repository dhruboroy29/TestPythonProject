import librosa
import numpy as np
import soundfile as sf
import sys
import resampy
import os
import progressbar

CLASS_TO_INT = {
    'bus': 0,
    'busystreet': 1,
    'office': 2,
    'openairmarket': 3,
    'park': 4,
    'quietstreet': 5,
    'restaurant': 6,
    'supermarket': 7,
    'tube': 8,
    'tubestation': 9
}


def compute_logmelspecs(fname, sr, output_dir, flatten=False):
    basename = os.path.basename(fname).split(os.extsep)[0]
    output_path = os.path.join(output_dir, basename)

    audio, sr_orig = sf.read(fname, dtype='float32', always_2d=True)
    audio = audio.mean(axis=-1)

    if sr_orig != sr:
        audio = resampy.resample(audio, sr_orig, sr)

    hop_size = 0.1
    hop_length = int(hop_size * sr)
    frame_length = sr * 1

    audio_length = len(audio)
    if audio_length < frame_length:
        # Make sure we can have at least one frame of audio
        pad_length = frame_length - audio_length
    else:
        # Zero pad so we compute embedding on all samples
        pad_length = int(np.ceil(audio_length - frame_length)/hop_length) * hop_length \
                     - (audio_length - frame_length)

    if pad_length > 0:
        # Use (roughly) symmetric padding
        left_pad = pad_length // 2
        right_pad= pad_length - left_pad
        audio = np.pad(audio, (left_pad, right_pad), mode='constant')

    # Divide into overlapping 1 second frames
    x = librosa.util.utils.frame(audio, frame_length=frame_length, hop_length=hop_length).T

    # Get class label
    class_label = np.array([CLASS_TO_INT[basename[:-2]]])

    # Compute log-melspecs in each slice
    logmelspec = []
    for row in xrange(np.shape(x)[0]):
        y=x[row]
        #Compute log-melspecs on each window
        n_fft = 2048
        # n_win = 480
        # n_hop = n_win//2
        n_mels = 256
        n_hop = 242
        S = librosa.feature.melspectrogram(y, sr=sr, n_fft=n_fft, n_mels=n_mels, hop_length=n_hop).T
        # Convert to log scale (dB). We'll use the peak power as reference.
        log_S = librosa.logamplitude(S, ref_power=np.max)

        # Reshape as a 1-D array
        log_S = log_S.ravel()
        # Concatenate class label as first element
        log_S = np.concatenate((class_label,log_S))

        logmelspec.append(log_S)

    #np.savez_compressed(output_path, X=logmelspec, y=class_label)
    np.save(output_path, logmelspec)
    # Test read
    read = np.load(output_path+'.npy')
    print('ReadTest')


if __name__=="__main__":
    #in_path = '/Users/Balderdash/Downloads/dcase2013/audio/fold1'
    in_path = sys.argv[1]
    target_sr = 48000
    out_path = in_path.replace('audio', 'logmelspec_'+str(target_sr/1000)+'KHz')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    audio_files = os.listdir(in_path)

    print 'Input directory:', in_path, '\n'
    bar = progressbar.ProgressBar(maxval=len(audio_files), \
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    n_iter=1
    for file in audio_files:
        compute_logmelspecs(os.path.join(in_path, file), target_sr, out_path)
        bar.update(n_iter)
        n_iter += 1

    bar.finish()