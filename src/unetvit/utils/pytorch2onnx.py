"""
convert pytorch to onnx model
"""

import os
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.onnx

from src.unetvit.models.unetvit import UNet
from src.unetvit.utils.helpers import get_default_device
from src.unetvit.utils.helpers import MODELS_PATH
from loguru import logger

def export_model(model, device, path_name, dummy_input):
    """
    Export model to onnx
    """
    torch.onnx.export(
        model,  # model being run
        dummy_input,  # model input (or a tuple for multiple inputs)
        path_name,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=16,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input": {0: "batch_size"},  # variable length axes
            "output": {0: "batch_size"},
        },
    )

    print(f"Saved ONNX model: {path_name}")


def main(input_model_name):
    """
    Convert pytorch model to onnx
    From the command line:
    python src/unetvit/utils/pytorch2onnx.py -i unetvit_epoch_5_0.59060.pth

    IN: input_model_name with pth extension
        The input size of data to the model is [batch, channel, height, width]
        It is definted by 
        dummy_input = torch.randn(1, 1, 400, 640, requires_grad=False).to(device)

    OUT: onnx model with onnx extension

    ISSUES:
        torch.onnx.errors.UnsupportedOperatorError:
        Exporting the operator 'aten::max_unpool2d' to ONNX opset version 14 is not supported.

    TODO validate_method?
        print(model)
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.data)
    """
    device = get_default_device()
    logger.info(f"device : {device}")
    
    number_of_channels = 3

    model_name = input_model_name[:-4]
    models_path_input_name = MODELS_PATH + "/" + model_name + ".pth" 
    models_path_output_name = MODELS_PATH + "/" + model_name + ".onnx"

    model = UNet(n_channels=number_of_channels, n_classes=6, bilinear=True).to(device)
    model.load_state_dict(
        torch.load(models_path_input_name, map_location=torch.device(device))
    )
    model = model.eval().to(device)

    batch_size = 1  # just a random number
    dummy_input = torch.randn((batch_size, number_of_channels, 512, 512)).to(device)
    
    export_model(model, device, models_path_output_name, dummy_input)
    logger.info(f"ONNX conversion has been scussecful to create: {models_path_output_name}")


if __name__ == "__main__":
    """
    USAGE
    python src/unetvit/utils/pytorch2onnx.py -i <model_name>.pth
    """

    # Parse args
    parser = ArgumentParser(description="Convert models to ONNX.")
    parser.add_argument(
        "-i",
        "--input_model_name",
        default="none",
        help=("Set model name"),
    )

    args = parser.parse_args()
    main(
        input_model_name=args.input_model_name,
    )
