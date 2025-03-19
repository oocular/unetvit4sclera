"""
datasets.py
"""

import os
import random
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from scipy import ndimage
from torch.utils.data import Dataset
from torchvision.io import read_image


class segDataset(torch.utils.data.Dataset):
    def __init__(self, root, training, transform=None):
        super(segDataset, self).__init__()
        self.root = root
        self.training = training
        self.transform = transform
        self.IMG_NAMES = sorted(glob(self.root + "/*/images/*.jpg"))
        self.BGR_classes = {
            "Water": [41, 169, 226],
            "Land": [246, 41, 132],
            "Road": [228, 193, 110],
            "Building": [152, 16, 60],
            "Vegetation": [58, 221, 254],
            "Unlabeled": [155, 155, 155],
        }  # in BGR

        self.bin_classes = [
            "Water",
            "Land",
            "Road",
            "Building",
            "Vegetation",
            "Unlabeled",
        ]

    def __getitem__(self, idx):
        img_path = self.IMG_NAMES[idx]
        mask_path = img_path.replace("images", "masks").replace(".jpg", ".png")

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path)
        cls_mask = np.zeros(mask.shape)
        cls_mask[mask == self.BGR_classes["Water"]] = self.bin_classes.index("Water")
        cls_mask[mask == self.BGR_classes["Land"]] = self.bin_classes.index("Land")
        cls_mask[mask == self.BGR_classes["Road"]] = self.bin_classes.index("Road")
        cls_mask[mask == self.BGR_classes["Building"]] = self.bin_classes.index(
            "Building"
        )
        cls_mask[mask == self.BGR_classes["Vegetation"]] = self.bin_classes.index(
            "Vegetation"
        )
        cls_mask[mask == self.BGR_classes["Unlabeled"]] = self.bin_classes.index(
            "Unlabeled"
        )
        cls_mask = cls_mask[:, :, 0]

        if self.training == True:
            if self.transform:
                image = transforms.functional.to_pil_image(image)
                image = self.transform(image)
                image = np.array(image)

            # 90 degree rotation
            if np.random.rand() < 0.5:
                angle = np.random.randint(4) * 90
                image = ndimage.rotate(image, angle, reshape=True)
                cls_mask = ndimage.rotate(cls_mask, angle, reshape=True)

            # vertical flip
            if np.random.rand() < 0.5:
                image = np.flip(image, 0)
                cls_mask = np.flip(cls_mask, 0)

            # horizontal flip
            if np.random.rand() < 0.5:
                image = np.flip(image, 1)
                cls_mask = np.flip(cls_mask, 1)

        image = cv2.resize(image, (512, 512)) / 255.0
        cls_mask = cv2.resize(cls_mask, (512, 512))
        image = np.moveaxis(image, -1, 0)

        return torch.tensor(image).float(), torch.tensor(cls_mask, dtype=torch.int64)

    def __len__(self):
        return len(self.IMG_NAMES)


class MobiousDataset(Dataset):
    """
    MobiousDataset
    """

    def __init__(self, f_dir, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.f_dir = f_dir

        self.img_path = list(os.listdir(os.path.join(self.f_dir, "images")))
        self.masks_path = [i.replace(".jpg", ".png") for i in self.img_path]
        # self.labels_path = [i.replace(".jpg", ".npy") for i in self.img_path]

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img_path = os.path.join(self.f_dir, "images", self.img_path[idx])
        masks_path = os.path.join(self.f_dir, "masks", self.masks_path[idx])
        # TODO check when there is no numpy lalbels
        # labels_path = os.path.join(self.f_dir, "labels", self.labels_path[idx])

        image = read_image(img_path).type(
            torch.float
        )  # / 255 #torch.Size([1, 3, 400, 640])
        # image = np.asarray(Image.open( img_path ).convert("RGB")) #torch.Size([1, 400, 640, 3])

        # label = np.load(labels_path)
        # label = torch.tensor(label, dtype=torch.float) #.permute(2,0,1) #.unsqueeze(0)
        # print(label.size())

        ## For torch.Size([batch_size_, 400, 640]); <class 'torch.Tensor'>; torch.cuda.LongTensor`
        # mask_np = np.asarray(Image.open( masks_path ).convert("L"))
        # ?mask =  np.asarray( Image.fromarray( Image.open( masks_path )  )  )
        # ?mask = Image.fromarray(masks_path) #np.asarray(Image.open( masks_path ).convert("L"))
        # mask_t = torch.tensor(mask_np, dtype=torch.long) #uint8

        ##################
        # mask_to_class: Creating non-overlapping masks
        # https://discuss.pytorch.org/t/how-to-combine-separate-annotations-for-multiclass-semantic-segmentation/121232/3
        # This works for me
        # https://discuss.pytorch.org/t/how-to-combine-separate-annotations-for-multiclass-semantic-segmentation/121232/3
        # https://discuss.pytorch.org/t/multiclass-segmentation-u-net-masks-format/70979/14
        # https://github.com/gujingxiao/Lane-Segmentation-Solution-For-BaiduAI-Autonomous-Driving-Competition/blob/master/utils/process_labels.py
        ##################

        # # print(f"x.size() {mask.shape}; type(x): {type(mask)}; x.type: {mask.type()} ")

        mask = Image.open(masks_path).convert("P")
        # For the “P” mode, this method translates pixels through the palette.

        mask = np.array(mask)
        mask = torch.tensor(mask, dtype=torch.long)

        # print(f"************************")
        # print(mask.unique())
        # torch.set_printoptions(threshold=torch.inf)
        # print(mask)

        # TODO add sanity check for plotting encoded masks
        # plt.subplot(2,5,1), plt.imshow(mask), plt.colorbar()
        # plt.subplot(2,5,2), plt.imshow(mask>0 ), plt.colorbar()
        # plt.subplot(2,5,3), plt.imshow(mask>30), plt.colorbar()
        # plt.subplot(2,5,4), plt.imshow(mask>180), plt.colorbar()
        # plt.subplot(2,5,5), plt.imshow(mask>200), plt.colorbar()
        # plt.show()

        encode_mask = torch.tensor(
            np.zeros((mask.shape[0], mask.shape[1])), dtype=torch.long
        )
        # 0: sclera
        encode_mask[mask > 0] = 1  # sclera (0 to 10)
        # 1: pupil
        encode_mask[mask > 30] = 2  # pupil (20 to 30)
        # 2: iris
        encode_mask[mask > 180] = 3  # iris (40 to 180)
        # 3: background
        # encode_mask[mask>200] = 3 #background (196 to 255)

        # print(f"************************")
        # print(encode_mask.type())
        # print(encode_mask.unique())

        # TODO add sanity check for plotting encoded masks
        # plt.subplot(2,5,6), plt.imshow(encode_mask), plt.colorbar()
        # plt.subplot(2,5,7), plt.imshow(encode_mask>0 ), plt.colorbar()
        # plt.subplot(2,5,8), plt.imshow(encode_mask>1), plt.colorbar()
        # plt.subplot(2,5,9), plt.imshow(encode_mask>2), plt.colorbar()
        # plt.subplot(2,5,10), plt.imshow(encode_mask>3), plt.colorbar()
        # plt.show()

        # I tried putpalette
        # https://stackoverflow.com/questions/76131649/convert-black-and-white-image-with-color-palette-using-python
        # https://bo-li.medium.com/how-to-convert-mask-image-to-colour-map-for-mmsegmentation-52e20496192
        # https://github.com/albumentations-team/albumentations/issues/1294
        # https://stackoverflow.com/questions/67642262/python-image-types-shapes-and-channels-for-segmentation

        # label =label.clone().detach()?
        # TO_TEST/TO_REMOVE
        # label = np.load(os.path.join(self.f_dir, "labels", self.labels_path[idx]))
        # label = torch.tensor(label, dtype=torch.long)  # .unsqueeze(0)
        # label = F.one_hot(label, 4).type(torch.float)
        # print(label)
        # label = label.reshape([4, 400, 640])
        # print(label)

        seed = np.random.randint(2147483647)  # make a seed with numpy generator
        random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)  # needed for torchvision 0.7
        if self.transform:
            image = self.transform(image)  #

        random.seed(seed)  # apply this seed to target tranfsorms
        torch.manual_seed(seed)  # needed for torchvision 0.7
        if self.target_transform:
            encode_mask = self.target_transform(encode_mask)

        encode_mask = encode_mask.squeeze(
            0
        )  # from torch.Size([1, 400, 640]) to #torch.Size([400, 640])

        # return image, label
        return image, encode_mask
