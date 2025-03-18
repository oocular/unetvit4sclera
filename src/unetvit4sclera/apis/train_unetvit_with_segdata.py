"""
https://www.kaggle.com/code/ganaianimesh/unet-with-vit
python src/unetvit/apis/train.py
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as transforms
from unetvit4sclera.models.unetvit import UNetViT
from unetvit4sclera.utils.datasets import segDataset
from torch.autograd import Variable

MAIN_DATASET_PATH = str(Path.home()) + "/datasets/unetvit"
DATASET_PATH = MAIN_DATASET_PATH + "/semantic-segmentation-dataset"

print(f"****** Start time ******* : {datetime.now()}")

color_shift = transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)
blurriness = transforms.GaussianBlur(3, sigma=(0.1, 2.0))

t = transforms.Compose([color_shift, blurriness])
dataset = segDataset(DATASET_PATH, training=True, transform=t)

test_num = int(0.1 * len(dataset))
# print(f'test data : {test_num}')#7

train_dataset, test_dataset = torch.utils.data.random_split(
    dataset,
    [len(dataset) - test_num, test_num],
    generator=torch.Generator().manual_seed(101),
)

BACH_SIZE = 4
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BACH_SIZE, shuffle=True, num_workers=0
)

test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BACH_SIZE, shuffle=False, num_workers=0
)


##################################### for GPU ###########################


def get_default_device():
    # pick the gpu if available
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def to_device(data, device):
    """
    move tensors to chosen device
    """
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
    # move the batches of the data to our selected device
    def __init__(self, dl, device):
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


#########################################################################
## Loss Function
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


criterion = FocalLoss(gamma=3 / 4).to(device)


#########################################################################
## Training
def acc(label, predicted):
    seg_acc = (y.cpu() == torch.argmax(pred_mask, axis=1).cpu()).sum() / torch.numel(
        y.cpu()
    )
    return seg_acc


def precision(y, pred_mask, classes=6):
    precision_list = []
    for i in range(classes):
        actual_num = y.cpu() == i
        predicted_num = i == torch.argmax(pred_mask, axis=1).cpu()
        prec = torch.logical_and(actual_num, predicted_num).sum() / predicted_num.sum()
        precision_list.append(prec.numpy().tolist())
    return precision_list


def recall(y, pred_mask, classes=6):
    recall_list = []
    for i in range(classes):
        actual_num = y.cpu() == i
        predicted_num = i == torch.argmax(pred_mask, axis=1).cpu()
        recall_val = (
            torch.logical_and(actual_num, predicted_num).sum() / actual_num.sum()
        )
        recall_list.append(recall_val.numpy().tolist())
    return recall_list


min_loss = torch.tensor(float("inf"))

model = UNetViT(n_channels=3, n_classes=6, bilinear=True).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)


os.makedirs(MAIN_DATASET_PATH + "/models", exist_ok=True)


N_EPOCHS = 1
# N_EPOCHS = 20
# N_EPOCHS = 100
N_DATA = len(train_dataset)
N_TEST = len(test_dataset)

plot_losses = []
scheduler_counter = 0


starttime = time.time()  # print(f'Starting training loop at {startt}')


for epoch in range(N_EPOCHS):
    # training
    model.train()
    loss_list = []
    acc_list = []
    for batch_i, (x, y) in enumerate(train_dataloader):

        pred_mask = model(x.to(device))
        loss = criterion(pred_mask, y.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.cpu().detach().numpy())
        acc_list.append(acc(y, pred_mask).numpy())

        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [Loss: %f (%f)]"
            % (
                epoch,
                N_EPOCHS,
                batch_i,
                len(train_dataloader),
                loss.cpu().detach().numpy(),
                np.mean(loss_list),
            )
        )
    scheduler_counter += 1

    # testing
    model.eval()
    val_loss_list = []
    val_acc_list = []
    for batch_i, (x, y) in enumerate(test_dataloader):
        with torch.no_grad():
            pred_mask = model(x.to(device))
        val_loss = criterion(pred_mask, y.to(device))
        val_loss_list.append(val_loss.cpu().detach().numpy())
        val_acc_list.append(acc(y, pred_mask).numpy())

    print(
        " epoch {} - loss : {:.5f} - acc : {:.2f} - val loss : {:.5f} - val acc : {:.2f}".format(
            epoch,
            np.mean(loss_list),
            np.mean(acc_list),
            np.mean(val_loss_list),
            np.mean(val_acc_list),
        )
    )

    plot_losses.append([epoch, np.mean(loss_list), np.mean(val_loss_list)])

    compare_loss = np.mean(val_loss_list)
    is_best = compare_loss < min_loss
    if is_best == True:
        scheduler_counter = 0
        min_loss = min(compare_loss, min_loss)
        torch.save(
            model.state_dict(),
            MAIN_DATASET_PATH
            + "/models/unetvit_epoch_{}_{:.5f}.pth".format(
                epoch, np.mean(val_loss_list)
            ),
        )

    if scheduler_counter > 5:
        lr_scheduler.step()
        print(f"lowering learning rate to {optimizer.param_groups[0]['lr']}")
        scheduler_counter = 0


## plot loss
plot_losses = np.array(plot_losses)
losses_csv = (
    MAIN_DATASET_PATH
    + "/models/losses_"
    + str(epoch)
    + str(np.mean(val_loss_list))
    + ".cvs"
)
np.savetxt(losses_csv, plot_losses, delimiter=",")

# plt.plot(plot_losses[:,0], plot_losses[:,1], color='b', linewidth=4)
# plt.plot(plot_losses[:,0], plot_losses[:,2], color='r', linewidth=4)
# plt.title('FocalLoss', fontsize=20)
# plt.xlabel('epoch',fontsize=20)
# plt.ylabel('loss',fontsize=20)
# plt.grid()
# plt.legend(['training', 'validation']) # using a named size
# plt.show()


endtime = time.time()
elapsedtime = endtime - starttime
print(f"Elapsed time for the training loop: {elapsedtime/60} (mins)")
print(f"****** End time ******* : {datetime.now()}")
