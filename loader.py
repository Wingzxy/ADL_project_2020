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
class_set = list()


for i, (input,target,filename,label) in enumerate(train_loader):
    # print(input)
    # target are their corresponding label
    if i == 0:
        print(i)
        print('##################')
        print(input.shape)

        print('--------------------------------------------------')
        # print(train_loader.dataset[int(i)]['classID'])
    class_set.append(set(label))

print(class_set[0])
# print(len(class_set))
    # class_set.add(train_loader.dataset[i]['classID'])

# x = torch.randn(2,3,4,5)
# print(x)
# for i, (input,target,filename) in enumerate(val_loader):
    # validation code