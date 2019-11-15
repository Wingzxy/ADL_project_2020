import torch
from torch.utils import data
import numpy as np
import pickle


class UrbanSound8KDataset(data.Dataset):
    # s = set()
    def __init__(self, dataset_path, mode):
        self.dataset = pickle.load(open(dataset_path, 'rb'))
        # print(self.dataset)
        self.mode = mode

    def __getitem__(self, index):

        if self.mode == 'LMC':
            # Log-mel spectrogram, chroma, spectral contrast and toonetz
            # Edit here to load and concatenate the neccessary features to
            # print("Undefined")
            # create the LMC feature

            log_mel = self.dataset[index]['features']['logmelspec']
            chroma = self.dataset[index]['features']['chroma']
            spectral_contrast = self.dataset[index]['features']['spectral_contrast']
            tonnetz = self.dataset[index]['features']['tonnetz']

            feature = np.concatenate((log_mel, chroma, spectral_contrast, tonnetz), axis=0)
            feature = torch.from_numpy(feature.astype(np.float32)).unsqueeze(0)

        elif self.mode == 'MC':
            # Edit here to load and concatenate the neccessary features to
            # MFCC, chroma, spectral contrast and toonetz
            # print("Undefined")
            # create the MC feature
            mfcc = self.dataset[index]['features']['mfcc']
            chroma = self.dataset[index]['features']['chroma']
            spectral_contrast = self.dataset[index]['features']['spectral_contrast']
            tonnetz = self.dataset[index]['features']['tonnetz']

            feature = np.concatenate((mfcc, chroma, spectral_contrast, tonnetz), axis=0)
            feature = torch.from_numpy(feature.astype(np.float32)).unsqueeze(0)

        elif self.mode == 'MLMC':
            # Edit here to load and concatenate the neccessary features to
            print("Undefined")
            # create the MLMC feature
            mfcc = self.dataset[index]['features']['mfcc']
            tonnetz = self.dataset[index]['features']['tonnetz']
            log_mel = self.dataset[index]['features']['logmelspec']
            feature = np.concatenate((log_mel, mfcc, tonnetz), axis=0)
            feature = torch.from_numpy(feature.astype(np.float32)).unsqueeze(0)
       
        label = self.dataset[index]['classID']
        classname = self.dataset[index]['class']
        fname = self.dataset[index]['filename']
        return feature, label, fname, classname

    def __len__(self):
        return len(self.dataset)
