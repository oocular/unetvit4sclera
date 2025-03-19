import matplotlib.pyplot as plt
import numpy as np
import onnxruntime
import torch
import torchvision.transforms as transforms
from loguru import logger

from  unetvit4sclera.models.unet import UNet


def test_gpu_availability():
    torch.cuda.is_available()
    logger.info(f"GPU availability: {torch.cuda.is_available()}")
    logger.info(f"GPU count: {torch.cuda.device_count()}")
    logger.info(f"GPU name: {torch.cuda.get_device_name(0)}")
    assert torch.cuda.is_available() == True, f"Expected GPU availability"
