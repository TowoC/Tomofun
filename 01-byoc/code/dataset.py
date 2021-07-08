import os
import pandas as pd
import numpy as np
import soundfile as sf
from tqdm import tqdm
import resampy
import librosa

import math, random
from IPython.display import Audio
import glob
from sklearn.preprocessing import label_binarize
from sklearn.utils import shuffle
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torchaudio
from torchaudio import transforms

try:
    def wav_read(wav_file):
        wav_data, sr = sf.read(wav_file, dtype='int16')
        return wav_data, sr

except ImportError:
    def wav_read(wav_file):
        raise NotImplementedError('WAV file reading requires soundfile package.')

def rechannel_(aud, new_channel):
    sig, sr = aud

    if (sig.shape[0] == new_channel):
        # Nothing to do
        return aud

    if (new_channel == 1):
        # Convert from stereo to mono by selecting only the first channel
        resig = sig[:1, :]
    elif (new_channel == 2):
        # Convert from mono to stereo by duplicating the first channel
        resig = torch.cat([sig, sig])
    elif (new_channel == 3):
        resig = torch.cat([sig, sig, sig])
    else:
        print('ERROR')

    return ((resig, sr))

# ----------------------------
# Since Resample applies to a single channel, we resample one channel at a time
# ----------------------------

def resample_(aud, newsr):
    sig, sr = aud

    if (sr == newsr):
        # Nothing to do
        return aud

    num_channels = sig.shape[0]
    # Resample first channel
    resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1,:])
    if (num_channels > 1):
        # Resample the second channel and merge both channels
        retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:,:])
        resig = torch.cat([resig, retwo])

    return ((resig, newsr))

# ----------------------------
# Pad (or truncate) the signal to a fixed length 'max_ms' in milliseconds
# ----------------------------

def pad_trunc_(aud, max_ms):
    sig, sr = aud
    num_rows, sig_len = sig.shape
    max_len = sr//1000 * max_ms

    if (sig_len > max_len):
        # Truncate the signal to the given length
        sig = sig[:,:max_len]

    elif (sig_len < max_len):
        # Length of padding to add at the beginning and end of the signal
        pad_begin_len = random.randint(0, max_len - sig_len)
        pad_end_len = max_len - sig_len - pad_begin_len

        # Pad with 0s
        pad_begin = torch.zeros((num_rows, pad_begin_len))
        pad_end = torch.zeros((num_rows, pad_end_len))

        sig = torch.cat((pad_begin, sig, pad_end), 1)

    return (sig, sr)

class AudioUtil():
    # ----------------------------
    # Load an audio file. Return the signal as a tensor and the sample rate
    # ----------------------------
    @staticmethod
    def open(audio_file):
        sig, sr = torchaudio.load(audio_file)
        return (sig, sr)
    
    
    # ----------------------------
    # Generate a Spectrogram
    # ----------------------------
    @staticmethod
    def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
        sig,sr = aud
        top_db = 80

        # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
        spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

        # Convert to decibels
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
        return (spec)


    
    
    # ----------------------------
    # Convert the given audio to the desired number of channels
    # ----------------------------
    @staticmethod
    def rechannel(aud, new_channel):
        sig, sr = aud

        if (sig.shape[0] == new_channel):
            # Nothing to do
            return aud

        if (new_channel == 1):
            # Convert from stereo to mono by selecting only the first channel
            resig = sig[:1, :]
        elif (new_channel == 2):
            # Convert from mono to stereo by duplicating the first channel
            resig = torch.cat([sig, sig])
        elif (new_channel == 3):
            resig = torch.cat([sig, sig, sig])
        else:
            print('ERROR')

        return ((resig, sr))
    
    # ----------------------------
    # Since Resample applies to a single channel, we resample one channel at a time
    # ----------------------------
    @staticmethod
    def resample(aud, newsr):
        sig, sr = aud

        if (sr == newsr):
            # Nothing to do
            return aud

        num_channels = sig.shape[0]
        # Resample first channel
        resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1,:])
        if (num_channels > 1):
            # Resample the second channel and merge both channels
            retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:,:])
            resig = torch.cat([resig, retwo])

        return ((resig, newsr))
    
    # ----------------------------
    # Pad (or truncate) the signal to a fixed length 'max_ms' in milliseconds
    # ----------------------------
    @staticmethod
    def pad_trunc(aud, max_ms):
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr//1000 * max_ms

        if (sig_len > max_len):
            # Truncate the signal to the given length
            sig = sig[:,:max_len]

        elif (sig_len < max_len):
            # Length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            # Pad with 0s
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))

            sig = torch.cat((pad_begin, sig, pad_end), 1)

        return (sig, sr)
    
    @staticmethod
    def background_soung_merge(ori_sig, aug_sig_paths, duration, add_alpha = 0.3, p = 0.5):
        if random.uniform(0.01, 1) >= p:
            sig, sr = ori_sig
            aug_sig_path = np.random.choice(aug_sig_paths, 1, replace=False)[0]
            aug_sig = torchaudio.load(aug_sig_path)

            aug_sig = rechannel_(aug_sig, sig.shape[0])
            aug_sig = resample_(aug_sig, sr)
            aug_sig = pad_trunc_(aug_sig, duration)

            aug_sig, aug_sr = aug_sig

            alpha = random.uniform(0, add_alpha)
            wave = (1 - alpha) * sig + alpha * aug_sig
            
            return (wave, sr)
        else:
            return ori_sig
        
        
    
    # ----------------------------
    # Shifts the signal to the left or right by some percent. Values at the end
    # are 'wrapped around' to the start of the transformed signal.
    # ----------------------------
    @staticmethod
    def time_shift(aud, shift_limit, p = 0.5):
        sig,sr = aud
        _, sig_len = sig.shape
        
        if random.uniform(0.01, 1) >= p:
            shift_amt = int(random.random() * shift_limit * sig_len)
            return (sig.roll(shift_amt), sr)
        else:
            return (sig, sr)
    
    # ----------------------------
    # Augment the Spectrogram by masking out some sections of it in both the frequency
    # dimension (ie. horizontal bars) and the time dimension (vertical bars) to prevent
    # overfitting and to help the model generalise better. The masked sections are
    # replaced with the mean value.
    # ----------------------------
    @staticmethod
    def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1, p = 0.5):
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec
        
        if random.uniform(0.01, 1) > (1 - p):
            freq_mask_param = max_mask_pct * n_mels
            for _ in range(n_freq_masks):
                aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

            time_mask_param = max_mask_pct * n_steps
            for _ in range(n_time_masks):
                aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

        return aug_spec
    
    def time_stretch(aud, rate):
        # rate：拉伸的尺寸，
        # rate > 1 加快速度
        # rate < 1 放慢速度
        sig,sr = aud
        
        if rate == 'random':
            rate = random.uniform(0.5, 1.5)
            
        return librosa.effects.time_stretch(np.array(sig[0]), rate)
    
    def pitch_shifting(aud, bins_per_octave=12, p=0.5):
        
        sig,sr = aud
        
        # sr: 音訊取樣率
        # n_steps: 要移動多少步
        # bins_per_octave: 每個八度音階(半音)多少步
        n_steps = random.randint(-5, 5)
        
        if random.uniform(0.01, 1) > (1 - p):
            return (torch.unsqueeze(torch.tensor(librosa.effects.pitch_shift(np.array(sig[0]), sr, n_steps, bins_per_octave=bins_per_octave)), dim = 0), sr)
        else:
            return aud


    
    def AddGaussianNoise(data, p=0.5):
        '''
        data : ndarray of audio timeseries
        ''' 
        
        if random.uniform(0.01, 1) > (1 - p):
            noise = np.random.randn(len(data[0][0]))
            data[0][0] = data[0][0] + 0.005*noise

        
        return data
    
    def PolarityInversion(data, p = 0.5):
        '''
        data : ndarray of audio timeseries
        '''
        
        if random.uniform(0.01, 1) > (1 - p):
            data[0][0] = -data[0][0]
            return data
        else:
            return data

    

    def Gain(data, min_gain_in_db=-6, max_gain_in_db=6, p=0.5):
        '''
        Multiply the audio by a random amplitude factor to reduce or increase the volume. This
        technique can help a model become somewhat invariant to the overall gain of the input audio.
        '''
        assert min_gain_in_db <= max_gain_in_db
        
        if random.uniform(0.01, 1) > (1 - p):
            amplitude_ratio = 10**(random.uniform(min_gain_in_db, max_gain_in_db)/20)
            data[0][0] = data[0][0] * amplitude_ratio
            
            return data
        else:
            return data

    def CutOut(data, p=0.5):
        '''
        data : ndarray of audio timeseries
        '''
        if random.uniform(0.01, 1) > (1 - p):
            start_ = np.random.randint(0,len(data[0][0]))
            end_ = np.random.randint(start_,len(data[0][0]))

            data[0][0][start_:end_] = 0

            return data
        else:
            return data


class SoundDataset(Dataset):
    def __init__(self, params, df, proba, label_flag = True):

        self.params = params
        self.csvfile = self.params.csv_path
        self.data_path = self.params.data_dir
        self.normalize = self.params.normalize
        # self.preload = self.params.preload
        self.p = proba
        self.aug_path = glob.glob(os.path.join(self.params.aug_dir, '*', '*.wav'))
        self.label_flag = label_flag
        self.df = df
        self.alpha = self.params.alpha

        self.duration = 5000
        self.sr = 44100
        self.channel = 3
        self.shift_pct = 0.4
        
  
            
    # ----------------------------
    # Number of items in dataset
    # ----------------------------
    def __len__(self):
        return len(self.df)    
    
    # ----------------------------
    # Get i'th item in dataset
    # ----------------------------
    def __getitem__(self, idx):
        # Absolute file path of the audio file - concatenate the audio directory with
        # the relative path
        audio_file = os.path.join(self.data_path, str(self.df.loc[idx, 'Filename'] + '.wav'))
        # Get the Class ID
        if self.label_flag == True:
            class_id = self.df.loc[idx, 'Label']

        aud = AudioUtil.open(audio_file)
        
        # normalize wav_data
        if self.normalize == 'peak':
            aud[0][0] = aud[0][0]/torch.max(aud[0][0])
        elif self.normalize == 'rms':
            rms_level = 0
            r = 10**(rms_level / 10.0)
            a = torch.sqrt((len(aud[0][0]) * r**2) / torch.sum(aud[0][0]**2))
            aud[0][0] = aud[0][0] * a
        else:
            # aud[0][0] = aud[0][0] / 32768.0
            pass
            
    
        # Some sounds have a higher sample rate, or fewer channels compared to the
        # majority. So make all sounds have the same number of channels and same 
        # sample rate. Unless the sample rate is the same, the pad_trunc will still
        # result in arrays of different lengths, even though the sound duration is
        # the same.
        reaud = AudioUtil.resample(aud, self.sr)
#         rechan = AudioUtil.rechannel(reaud, self.channel)
        dur_aud = AudioUtil.pad_trunc(reaud, self.duration)
        back_merge = AudioUtil.background_soung_merge(dur_aud, self.aug_path, self.duration, add_alpha = self.alpha, p = self.p)
        
#         stretch = AudioUtil.time_stretch(dur_aud, rate='random')
        pitch_shift = AudioUtil.pitch_shifting(back_merge, bins_per_octave=12, p = self.p)
        noiseadd = AudioUtil.AddGaussianNoise(pitch_shift, p = self.p)
        polarinverse = AudioUtil.PolarityInversion(noiseadd, p = self.p)
        gain = AudioUtil.Gain(polarinverse, min_gain_in_db=-12, max_gain_in_db=12, p = self.p)
        
        shift_aud = AudioUtil.time_shift(gain, self.shift_pct,p = self.p)
        shift_aud = AudioUtil.rechannel(shift_aud, self.channel)
        
        sgram = AudioUtil.spectro_gram(shift_aud, n_mels=128, n_fft=2048, hop_len=None)
        
        aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
        
        if self.label_flag == True:
            return aug_sgram, class_id
        else:
            return aug_sgram



# class SoundDataset(Dataset):
#     """Create Sound Dataset with loading wav files and labels from csv.

#     Attributes:
#         params: A class containing all the parameters.
#         data_type: A string indicating train or val.
#         csvfile: A string containing our wav files and labels.
#         normalize: A boolean indicating spectrogram is normalized to -1 to 1 or not.
#         mixup: A boolean indicating whether to do mixup augmentation or not.
#         preload: A boolean indicating whether to preload the spectrogram into memory or not.
#     """
#     def __init__(self, params):
#         """Init SoundDataset with params
#         Args:
#             params (class): all arguments parsed from argparse
#             train (bool): train or val dataset
#         """
#         self.params = params
#         self.csvfile = params.csv_path
#         self.data_dir = params.data_dir
#         self.normalize = self.params.normalize
#         self.preload = self.params.preload

#         self.X, self.Y, self.filenames = self.read_data(self.csvfile)
#         if self.preload:
#             self.X = self.convert_to_spec(self.X)
#             self.shape = self.get_shape(self.X[0])
#         else:
#             self.shape = self.get_shape(self.preprocessing(self.X[0][0], self.X[0][1]))

#     def read_data(self, csvfile):
#         """Read wav file from csv
#         Args:
#             csvfile: A string specifying the path of csvfile
#         Return:
#             data: A list of tuple (wav data in np.int16 data type, sampling rate of wav file)
#             label: A list of labels corresponding to the wav data
#             filenames: A list of filenames of the wav file
#         """
#         df = pd.read_csv(csvfile)
#         data, label, filenames = [], [], []
#         print("reading wav files...")
#         for i in tqdm(range(len(df))):
#             row = df.iloc[i]
#             path = os.path.join(self.data_dir, row.Filename + ".wav")
#             wav_data, sr = wav_read(path)
#             assert wav_data.dtype == np.int16
#             data.append((wav_data, sr))
#             label.append(row.Label)
#             filenames.append(path)
#         return data, label, filenames

#     def convert_to_spec(self, data):
#         """Convert wav_data into log mel spectrogram.
#         Args:
#             data: A list of tuple (wav data in np.int16 data type, sampling rate of wav file)
#         Return:
#             A list of log mel spectrogram
#         """
#         print("convert to log mel spectrogram...")
#         return [self.preprocessing(wav, sr) for wav, sr in tqdm(data)]

#     def spec_to_image(self, specs, eps=1e-6):
#         """ normalize the input
#         Args:
#             specs: A list of log mel spectrogram
#         Return:
#             A list of normalized log mel spectrogram
#         """
#         x = []
#         for spec in specs:
#             spec = np.squeeze(np.array(spec))
#             mean = spec.mean()
#             std = spec.std()
#             spec_norm = (spec - mean) / (std + eps)
#             spec_min, spec_max = spec_norm.min(), spec_norm.max()
#             spec_scaled = (spec_norm - spec_min) / (spec_max - spec_min)
#             spec_scaled = np.reshape(spec_scaled, (1, spec_scaled.shape[0], spec_scaled.shape[1]))
#             x.append(spec_scaled)
#         return np.array(x).astype('float32')

#     def get_shape(self, x):
#         """Get the shape of input data.
#         """
#         return x.shape

#     def preprocessing(self, wav_data, sr):
#         """Convert wav_data to log mel spectrogram.
#             1. normalize the wav_data
#             2. convert the wav_data into mono-channel
#             3. resample the wav_data to the sampling rate we want
#             4. compute the log mel spetrogram with librosa function
#         Args:
#             wav_data: An np.array indicating wav data in np.int16 datatype
#             sr: An integer specifying the sampling rate of this wav data
#         Return:
#             inpt: An np.array indicating the log mel spectrogram of data
#         """
#         # normalize wav_data
#         if self.normalize == 'peak':
#             samples = wav_data/np.max(wav_data)
#         elif self.normalize == 'rms':
#             rms_level = 0
#             r = 10**(rms_level / 10.0)
#             a = np.sqrt((len(wav_data) * r**2) / np.sum(wav_data**2))
#             samples = wav_data * a
#         else:
#             samples = wav_data / 32768.0

#         # convert samples to mono-channel file
#         if len(samples.shape) > 1:
#             samples = np.mean(samples, axis=1)

#         # resample samples to 8k
#         if sr != self.params.sr:
#             samples = resampy.resample(samples, sr, self.params.sr)

#         # transform samples to mel spectrogram
#         inpt_x = 500
#         spec = librosa.feature.melspectrogram(samples, sr=self.params.sr, n_fft=self.params.nfft, hop_length=self.params.hop, n_mels=self.params.mel)
#         spec_db = librosa.power_to_db(spec).T
#         spec_db = np.concatenate((spec_db, np.zeros((inpt_x - spec_db.shape[0], self.params.mel))), axis=0) if spec_db.shape[0] < inpt_x else spec_db[:inpt_x]
#         inpt = np.reshape(spec_db, (1, spec_db.shape[0], spec_db.shape[1]))
#         # inpt = np.reshape(spec_db, (spec_db.shape[0], spec_db.shape[1]))

#         return inpt.astype('float32')

#     def __len__(self):
#         return len(self.X)

#     def __getitem__(self, idx):
#         spec = self.X[idx] if self.params.preload else self.preprocessing(self.X[idx][0], self.X[idx][1])
#         label = self.Y[idx]

#         return spec.astype('float32'), label



class SoundDataset_test(SoundDataset):
    def __init__(self, params):
        self.params = params
        self.csvfile = params.csv_path
        self.data_dir = params.data_dir
        self.normalize = self.params.normalize
        self.preload = self.params.preload

        self.X, self.Y, self.filenames = self.read_data(self.csvfile)
        if self.preload:
            self.X = self.convert_to_spec(self.X)
            self.shape = self.get_shape(self.X[0])
        else:
            self.shape = self.get_shape(self.preprocessing(self.X[0][0], self.X[0][1]))

    def read_data(self, csvfile):
        df = pd.read_csv(csvfile)
        data, label, filenames = [], [], []
        print("reading wav files...")
        for i in tqdm(range(len(df))):
            row = df.iloc[i]
            path = os.path.join(self.data_dir, row.Filename + ".wav")
            wav_data, sr = wav_read(path)
            assert wav_data.dtype == np.int16
            data.append((wav_data, sr))
            lb = None
            if row.Barking == 1:
                lb = 0
            elif row.Howling == 1:
                lb = 1
            elif row.Crying == 1:
                lb = 2
            elif row.COSmoke == 1:
                lb = 3
            elif row.GlassBreaking == 1:
                lb = 4
            elif row.Other == 1:
                lb = 5
            label.append(lb)
            filenames.append(path)
        return data, label, filenames


