import torch
from torch.utils import data
import numpy as np
import pickle
import librosa
from SpecAugment import spec_augment_pytorch

class UrbanSound8KDataset(data.Dataset):
    # s = set()
    def __init__(self, dataset_path, mode, spec_augment=False):
        self.dataset = pickle.load(open(dataset_path, 'rb'))
        self.spec_augment=spec_augment
        self.mode = mode
        if spec_augment:
            self.transform=lambda x :spec_augment_pytorch.spec_augment(x,frequency_masking_para=60,time_masking_para=41)
        else:
            self.transform=lambda x:x

    def __getitem__(self, index):

        if self.mode == 'LMC':
            # Log-mel spectrogram, chroma, spectral contrast and toonetz
            # Edit here to load and concatenate the neccessary features to
            # create the LMC feature

            log_mel = self.dataset[index]['features']['logmelspec']
            chroma = self.dataset[index]['features']['chroma']
            spectral_contrast = self.dataset[index]['features']['spectral_contrast']
            tonnetz = self.dataset[index]['features']['tonnetz']

            if self.spec_augment:
                mel = librosa.core.db_to_power(log_mel)
                shape=mel.shape
                mel = np.reshape(mel, (-1, shape[0], shape[1]))
                warped_masked_mel=self.transform(torch.from_numpy(mel))[0,:,:].numpy()
                log_mel=librosa.core.power_to_db(warped_masked_mel)

            feature = np.concatenate((log_mel, chroma, spectral_contrast, tonnetz), axis=0)
            feature = torch.from_numpy(feature.astype(np.float32)).unsqueeze(0)

        elif self.mode == 'MC':
            # Edit here to load and concatenate the neccessary features to
            # MFCC, chroma, spectral contrast and toonetz
            # create the MC feature

            log_mel = self.dataset[index]['features']['logmelspec']
            mfcc = self.dataset[index]['features']['mfcc']
            chroma = self.dataset[index]['features']['chroma']
            spectral_contrast = self.dataset[index]['features']['spectral_contrast']
            tonnetz = self.dataset[index]['features']['tonnetz']

            if self.spec_augment:
                mel = librosa.core.db_to_power(log_mel)
                shape=mel.shape
                mel = np.reshape(mel, (-1, shape[0], shape[1]))
                warped_masked_mel=self.transform(torch.from_numpy(mel))[0,:,:].numpy()

                log_mel=librosa.core.power_to_db(warped_masked_mel)

                mfcc=librosa.feature.mfcc(S=log_mel)
                mfcc_delta = librosa.feature.delta(mfcc)
                mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
                mfcc=np.concatenate((mfcc, mfcc_delta, mfcc_delta2), axis=0)


            feature = np.concatenate((mfcc, chroma, spectral_contrast, tonnetz), axis=0)
            feature = torch.from_numpy(feature.astype(np.float32)).unsqueeze(0)

        elif self.mode == 'MLMC':
            # Edit here to load and concatenate the neccessary features to
            # create the MLMC feature

            mfcc = self.dataset[index]['features']['mfcc']
            log_mel = self.dataset[index]['features']['logmelspec']
            chroma = self.dataset[index]['features']['chroma']
            spectral_contrast = self.dataset[index]['features']['spectral_contrast']
            tonnetz = self.dataset[index]['features']['tonnetz']

            if self.spec_augment:
                mel = librosa.core.db_to_power(log_mel)
                shape=mel.shape
                mel = np.reshape(mel, (-1, shape[0], shape[1]))
                warped_masked_mel=self.transform(torch.from_numpy(mel))[0,:,:].numpy()
                log_mel=librosa.core.power_to_db(warped_masked_mel)
                mfcc=librosa.feature.mfcc(S=log_mel)

            feature = np.concatenate((mfcc, log_mel, chroma, spectral_contrast, tonnetz), axis=0)
            feature = torch.from_numpy(feature.astype(np.float32)).unsqueeze(0)

        label = self.dataset[index]['classID']
        classname = self.dataset[index]['class']
        fname = self.dataset[index]['filename']
        return feature, label, fname, classname

    def __len__(self):
        return len(self.dataset)


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)
