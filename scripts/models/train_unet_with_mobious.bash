#!/bin/bash
#set -Eeuxo pipefail

SCRIPT_PATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_PATH/../../
source .venv/bin/activate #To activate the virtual environment

python src/unetvit4sclera/apis/train.py 
#EXAMPLE: python src/ready/apis/train_mobious.py -c configs/models/unet/config_train_unet_with_mobious.yaml
#EXAMPLE: python src/unetvit4sclera/utils/pytorch2onnx.py -i <model_name>.pth
