import torch
import numpy as np
import matplotlib.pyplot as plt
from src.unetvit.models.unetvit import UNet
from src.unetvit.utils.datasets import segDataset
from conftest import DATASET_PATH, MODELS_PATH
from src.unetvit.utils.helpers import get_default_device, DeviceDataLoader, precision, recall
import torchvision.transforms as transforms
from loguru import logger

def test_segDataset():
    """
    Test segDataset class
    pytest -vs tests/test_inference.py::test_segDataset
    """
    color_shift = transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)
    blurriness = transforms.GaussianBlur(3, sigma=(0.1, 2.0))
    t = transforms.Compose([color_shift, blurriness])
    dataset = segDataset(DATASET_PATH, training=True, transform=t)

    assert len(dataset) == 72, f"Expected 72 images, but got {len(dataset)}"
    
    d = dataset[1]
    assert d[0].shape == (3, 512, 512), f"Expected image shape (3, 512, 512), but got {d[0].shape}"
    assert d[1].shape == (512, 512), f"Expected mask shape (512, 512), but got {d[1].shape}"

    logger.info(f"len(dataset) : {len(dataset)}")
    logger.info(f"len(d) : {len(d)}")
    logger.info(f"d[0].shape : {d[0].shape}")
    logger.info(f"d[1].shape : {d[1].shape}")

    #plot image and mask
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.imshow(np.moveaxis(d[0].numpy(),0,-1))
    plt.title("image")
    plt.subplot(1,2,2)
    plt.imshow(d[1].numpy())
    plt.title("mask")
    plt.show()

def test_inference():
    """
    Test inference
    pytest -vs tests/test_inference.py::test_inference
    """

    device = get_default_device()

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


    train_dataloader = DeviceDataLoader(train_dataloader, device)
    test_dataloader = DeviceDataLoader(test_dataloader, device)

    assert len(test_dataloader) == 2, f"Expected 2 batches, but got {len(test_dataloader)}"
    assert len(train_dataloader) == 17, f"Expected 17 batches, but got {len(train_dataloader)}"


    model = UNet(n_channels=3, n_classes=6, bilinear=True).to(device)
    model.load_state_dict(torch.load(MODELS_PATH + "/unetvit_epoch_5_0.59060.pth"))
    model.eval()

    #TODO: Add condition to plot this section
    for batch_i, (x, y) in enumerate(test_dataloader):
        for j in range(len(x)):
            result = model(x[j : j + 1])
            mask = torch.argmax(result, axis=1).cpu().detach().numpy()[0]
            im = np.moveaxis(x[j].cpu().detach().numpy(), 0, -1).copy() * 255
            im = im.astype(int)
            gt_mask = y[j].cpu()

            plt.figure(figsize=(12, 12))

            plt.subplot(1, 3, 1)
            im = np.moveaxis(x[j].cpu().detach().numpy(), 0, -1).copy() * 255
            im = im.astype(int)
            plt.imshow(im)
            plt.title("image")

            plt.subplot(1, 3, 2)
            plt.imshow(gt_mask)
            plt.title("gt_mask")

            plt.subplot(1, 3, 3)
            plt.imshow(mask)
            plt.title("predicted mask")

            plt.show()

    pred_list = []
    gt_list = []
    precision_list = []
    recall_list = []
    for batch_i, (x, y) in enumerate(test_dataloader):
        for j in range(len(x)):
            result = model(x.to(device)[j : j + 1])
            precision_list.append(precision(y[j], result))
            recall_list.append(recall(y[j], result))


    final_precision = np.nanmean(precision_list, axis=0)
    final_recall = np.nanmean(recall_list, axis=0)
    f1_score = 2 * (sum(final_precision[:-1])/5 * sum(final_recall)/5) / (sum(final_precision[:-1])/5 + sum(final_recall)/5)
    
    logger.info(f"nanmean of precision_list : {final_precision}")
    logger.info(f"nanmean of recall_list : {final_recall}")
    logger.info(f"Final precision : {sum(final_precision[:-1])/5}")
    logger.info(f"Final recall : {sum(final_recall)/5}")
    logger.info(f"Final f1_score : {f1_score}")

