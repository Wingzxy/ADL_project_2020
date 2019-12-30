import torch
from torch.utils import data
from dataset import UrbanSound8KDataset

print("start...")

train_loader = torch.utils.data.DataLoader(UrbanSound8KDataset('UrbanSound8K_train.pkl', "LMC"),
                                          batch_size=32,
                                          shuffle=True,
                                          num_workers=8,
                                          pin_memory=True)

print("first finished...")

val_loader = torch.utils.data.DataLoader(UrbanSound8KDataset('UrbanSound8K_test.pkl', "MC"),
                                         batch_size=32,
                                         shuffle=False,
                                         num_workers=8,
                                         pin_memory=True)

print("all finished...")

train_class_counts=[0,0,0,0,0,0,0,0,0,0]
val_class_counts=[0,0,0,0,0,0,0,0,0,0]


for i, (input,target,filename,label) in enumerate(train_loader):

    # target are their corresponding label
    for c in range(0,10):
        train_class_counts[c]+=target.tolist().count(c)
        # print(train_loader.dataset[int(i)]['classID'])
# print(len(class_set))
    # class_set.add(train_loader.dataset[i]['classID'])

# x = torch.randn(2,3,4,5)
# print(x)
for i, (input,target,filename,label) in enumerate(val_loader):
    for c in range(0,10):
        val_class_counts[c]+=target.tolist().count(c)

print("Train dataset class counts")
print(train_class_counts)
print("Test dataset class counts")
print(val_class_counts)
