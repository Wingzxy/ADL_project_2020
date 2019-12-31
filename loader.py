import torch
from torch.utils import data
from dataset import UrbanSound8KDataset
from SpecAugment import spec_augment_pytorch
import matplotlib.pyplot as plt


print("start...")

train_loader = torch.utils.data.DataLoader(UrbanSound8KDataset('UrbanSound8K_train.pkl', "LMC"),
                                          batch_size=32,
                                          shuffle=True,
                                          num_workers=8,
                                          pin_memory=True)

print("first finished...")
#
#val_loader = torch.utils.data.DataLoader(UrbanSound8KDataset('UrbanSound8K_test.pkl', "MC"),
#                                         batch_size=32,
#                                         shuffle=False,
#                                         num_workers=8,
#                                         pin_memory=True)

print("all finished...")

train_class_counts=[0,0,0,0,0,0,0,0,0,0]
val_class_counts=[0,0,0,0,0,0,0,0,0,0]


for i, (input,target,filename,label) in enumerate(train_loader):
    if i==0:
        print(input.size())
        spec_augment_pytorch.visualization_spectrogram(mel_spectrogram=input[0],
                                                      title="Mel Spectrogram")
        break

#    for c in range(0,10):
#        train_class_counts[c]+=target.tolist().count(c)

#for i, (input,target,filename,label) in enumerate(val_loader):

#    for c in range(0,10):
#        val_class_counts[c]+=target.tolist().count(c)

#print("Train dataset class counts")
#print(train_class_counts)
#print("Test dataset class counts")
#print(val_class_counts)

