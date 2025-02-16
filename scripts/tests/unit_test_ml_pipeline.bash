#!/bin/bash
set -Eeuxo pipefail

source .venv/bin/activate #To activate the virtual environment:

pytest -vs tests/ml_pipeline.py::test_gpu_availability
#pytest -vv tests/ml_pipeline.py
#pytest -vs tests/ml_pipeline.py::test_segDataset
#pytest -vs tests/ml_pipeline.py::test_inference

