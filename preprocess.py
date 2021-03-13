""" Written by Youval Vanlaer
13/03/2021 """

import librosa
import pretty_midi
import h5py
import argparse
import numpy as np
import os

rate = 16000
hop_size = 512
window_size = 7
min_note = 1
max_note = 88

def preprocess_wav(wavfile):
    print("wav2inputnp")
    bins_per_octave = 36
    n_bins = bins_per_octave * 7

    y,_ = librosa.load(wavfile, rate)
    S = np.abs(librosa.cqt(y,fmin=librosa.midi_to_hz(min_note), sr=rate, hop_length=hop_size,
                      bins_per_octave=bins_per_octave, n_bins=n_bins).T)
    min_S = np.min(S)

    print(np.min(S),np.max(S),np.mean(S))
    S = np.pad(S, ((window_size//2,window_size//2),(0,0)), 'constant', constant_values=min_S)

    inputs = []

    for i in range(S.shape[0]-window_size+1):
        inputs += [S[i:i+window_size,:]]

    inputs = np.array(inputs)
    return inputs

def preprocess_midi(midifile,times):
    pretty_midi.pretty_midi.MAX_TICK = 1e10
    midi = pretty_midi.PrettyMIDI(midifile)
    piano_roll = midi.get_piano_roll(fs=rate,times=times)[min_note:max_note+1].T
    piano_roll[piano_roll > 0] = 1
    return piano_roll

def properMkDir(path,newDir):
    newPath = os.path.join(path,newDir)
    if not os.path.exists(newPath):
        os.mkdir(newPath)
    return newPath

data_dir = 'MAPS/AkPnBcht/MUS'
output_dir = 'preprocess/'
def preprocess(args):
    framecnt = 0

    inputs,outputs = [],[]
    addCnt, errCnt = 0,0
    for dp, dn, filenames in os.walk(data_dir):
        for f in filenames:

            if f.endswith('.wav'):
                audiofile = f
                fprefix = audiofile.split('.wav')[0]
                pre_midifile = fprefix + '.mid'
                if pre_midifile in filenames:
                    wavfile = os.path.join(dp,audiofile)
                    midifile = os.path.join(dp,pre_midifile)

                    inputnp = preprocess_wav(wavfile)
                    times = librosa.frames_to_time(np.arange(inputnp.shape[0]),sr=rate,hop_length=hop_size)
                    outputnp = preprocess_midi(midifile,times)

                    # check that num onsets is equal
                    if inputnp.shape[0] == outputnp.shape[0]:
                        print("adding to dataset fprefix {}".format(fprefix))
                        addCnt += 1
                        framecnt += inputnp.shape[0]
                        print("framecnt is {}".format(framecnt))
                        inputs.append(inputnp)
                        outputs.append(outputnp)
                    else:
                        print("error for fprefix {}".format(fprefix))
                        errCnt += 1
                        print(inputnp.shape)
                        print(outputnp.shape)

        print("{} examples in dataset".format(addCnt))
        print("{} examples couldnt be processed".format(errCnt))


        # concatenate dynamic list to numpy list of example
        if addCnt:
            inputs = np.concatenate(inputs)
            outputs = np.concatenate(outputs)

            sub_folder = properMkDir(output_dir,data_dir.split('/')[-2])
            folder = properMkDir(sub_folder,data_dir.split('/')[-1])

            mmi = np.memmap(filename=os.path.join(folder,'input.dat'), mode='w+',shape=inputs.shape)
            mmi[:] = inputs[:]
            mmo = np.memmap(filename=os.path.join(folder,'output.dat'), mode='w+',shape=outputs.shape)
            mmo[:] = outputs[:]
            del mmi
            del mmo


if __name__ == '__main__':
    prsr = argparse.ArgumentParser()

    #prsr.add_argument('data_dir')
    #prsr.add_argument('output_dir')

    args = vars(prsr.parse_args(  ))

    # x = preprocess_wav('MAPS/ENSTDkCl/MUS/MAPS_MUS-alb_se2_ENSTDkCl.wav')
    # times = librosa.frames_to_time(np.arange(x.shape[0]),sr=rate,hop_length=hop_size)
    # mid = preprocess_midi('MAPS/ENSTDkCl/MUS/MAPS_MUS-alb_se2_ENSTDkCl.mid', times=times)
    # preprocess(args)