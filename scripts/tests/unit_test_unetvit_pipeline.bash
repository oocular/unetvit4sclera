#!/bin/bash
set -Eeuxo pipefail

SCRIPT_PATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_PATH/../../
source .venv/bin/activate #To activate the virtual environment:

pytest -vs tests/test_unetvit_pipeline.py::test_gpu_availability
#pytest -vv tests/test_unetvit_pipeline.py
#pytest -vs tests/test_unetvit_pipeline.py::test_segDataset
#pytest -vs tests/test_unetvit_pipeline.py::test_inference

