from pathlib import Path

import torch

MAIN_DATASET_PATH = str(Path.home()) + "/datasets/unetvit"
DATASET_PATH = MAIN_DATASET_PATH + "/semantic-segmentation-dataset"
MODELS_PATH = MAIN_DATASET_PATH + "/models"


def get_default_device():
    """
    Pick the gpu if available
    """
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
    """
    Move the batches of the data to our selected device
    """

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)


def precision(y, pred_mask, classes=6):
    """
    Precssion is the ratio of ...
    """
    precision_list = []
    for i in range(classes):
        actual_num = y.cpu() == i
        predicted_num = i == torch.argmax(pred_mask, axis=1).cpu()
        prec = torch.logical_and(actual_num, predicted_num).sum() / predicted_num.sum()
        precision_list.append(prec.numpy().tolist())
    return precision_list


def recall(y, pred_mask, classes=6):
    """
    Recall is the ratio of ...
    """
    recall_list = []
    for i in range(classes):
        actual_num = y.cpu() == i
        predicted_num = i == torch.argmax(pred_mask, axis=1).cpu()
        recall_val = (
            torch.logical_and(actual_num, predicted_num).sum() / actual_num.sum()
        )
        recall_list.append(recall_val.numpy().tolist())
    return recall_list
