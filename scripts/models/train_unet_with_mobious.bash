#!/bin/bash
#set -Eeuxo pipefail

SCRIPT_PATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_PATH/../../
source .venv/bin/activate #To activate the virtual environment

python src/unetvit4sclera/apis/train_unet_with_mobious.py -c configs/models/config_train_unet_with_mobious.yaml
