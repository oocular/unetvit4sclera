"""
python src/unetvit/apis/inference.py 
"""
import numpy as np
import matplotlib.pyplot as plt
from src.unetvit.utils.datasets import segDataset
from src.unetvit.models.unetvit import UNet

import torch
import torchvision.transforms as transforms

from pathlib import Path

MAIN_DATASET_PATH = str(Path.home())+"/datasets/unetvit"
DATASET_PATH = MAIN_DATASET_PATH +"/semantic-segmentation-dataset"
MODELS_PATH = MAIN_DATASET_PATH +"/models"


color_shift = transforms.ColorJitter(.1,.1,.1,.1)
blurriness = transforms.GaussianBlur(3, sigma=(0.1, 2.0))

t = transforms.Compose([color_shift, blurriness])
dataset = segDataset(DATASET_PATH, training = True, transform= t)

# print(len(dataset))#72
# d = dataset[1]
# print(d[0].shape, d[1].shape)#torch.Size([3, 512, 512]) torch.Size([512, 512])
# plt.figure(figsize=(15,15))
# plt.subplot(1,2,1)
# plt.imshow(np.moveaxis(d[0].numpy(),0,-1))
# plt.subplot(1,2,2)
# plt.imshow(d[1].numpy())
# plt.show()

test_num = int(0.1 * len(dataset))
# print(f'test data : {test_num}')#7

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset)-test_num, test_num], generator=torch.Generator().manual_seed(101))

BACH_SIZE = 4
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BACH_SIZE, shuffle=True, num_workers=0)

test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BACH_SIZE, shuffle=False, num_workers=0)


def get_default_device():
    # pick the gpu if available
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data,device):
    #move tensors to choosen device
    if isinstance(data,(list,tuple)):
        return [to_device(x,device) for x in data]
    return data.to(device,non_blocking = True)


class DeviceDataLoader():
    # move the batches of the data to our selected device
    def __init__(self,dl,device):
        self.dl = dl
        self.device = device
    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)
    def __len__(self):
        return len(self.dl)

device = get_default_device()
# print(f"Device : {device}") #cuda

train_dataloader = DeviceDataLoader(train_dataloader, device)
test_dataloader = DeviceDataLoader(test_dataloader, device)


model = UNet(n_channels=3, n_classes=6, bilinear=True).to(device)
model.load_state_dict(torch.load(MODELS_PATH+'/unetvit_epoch_5_0.59060.pt'))
model.eval()


for batch_i, (x, y) in enumerate(test_dataloader):
    for j in range(len(x)):
        result = model(x[j:j+1])
        mask = torch.argmax(result, axis=1).cpu().detach().numpy()[0]
        im = np.moveaxis(x[j].cpu().detach().numpy(), 0, -1).copy()*255
        im = im.astype(int)
        gt_mask = y[j].cpu()

        plt.figure(figsize=(12,12))

        plt.subplot(1,3,1)
        im = np.moveaxis(x[j].cpu().detach().numpy(), 0, -1).copy()*255
        im = im.astype(int)
        plt.imshow(im)
        plt.title('image')
        
        plt.subplot(1,3,2)
        plt.imshow(gt_mask)
        plt.title('gt_mask')

        plt.subplot(1,3,3)
        plt.imshow(mask)
        plt.title('predicted mask')

        plt.show()


def precision(y, pred_mask, classes = 6):
    precision_list = [];
    for i in range(classes):
        actual_num = y.cpu() == i
        predicted_num = i == torch.argmax(pred_mask, axis=1).cpu()
        prec = torch.logical_and(actual_num,predicted_num).sum()/predicted_num.sum()
        precision_list.append(prec.numpy().tolist())
    return precision_list

def recall(y, pred_mask, classes = 6):
    recall_list = []
    for i in range(classes):
        actual_num = y.cpu() == i
        predicted_num = i == torch.argmax(pred_mask, axis=1).cpu()
        recall_val = torch.logical_and(actual_num, predicted_num).sum() / actual_num.sum()
        recall_list.append(recall_val.numpy().tolist())
    return recall_list



pred_list = []
gt_list = []
precision_list = []
recall_list = []
for batch_i, (x, y) in enumerate(test_dataloader):
    for j in range(len(x)):
        result = model(x.to(device)[j:j+1])
        precision_list.append(precision(y[j],result))
        recall_list.append(recall(y[j],result))


print(f"nanmean of precision_list : {np.nanmean(precision_list,axis = 0)}")
print(f"nanmean of recall_list : {np.nanmean(recall_list,axis = 0)}")

final_precision = np.nanmean(precision_list,axis = 0)
print(f"Final precision : {sum(final_precision[:-1])/5}")


final_recall = np.nanmean(recall_list,axis = 0)
print(f"Final recall : {sum(final_recall)/5}")