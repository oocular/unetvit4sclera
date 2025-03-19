# unetvit4sclera

## :eyeglasses: Overview
This repository contains documentation and code for Open-Source UNet and UNet-ViT models for debugging, training, and evaluating sclera segmentation workflows.

## :school_satchel: Getting started
* :page_facing_up: [Docs](docs/README.md) Getting started, debugging, testing, demos.
* :floppy_disk: [Data](data/): [SBVPI](data/sbvpi); [mobious](data/mobious);
* :brain: [unet](src/unetvit4sclera/models/unet.py) and [unetvit](src/unetvit4sclera/models/unetvit.py)

## :nut_and_bolt: Dev installation
Commands to start your dev installation.
See further details for installation [here](docs).
```
uv venv --python 3.12 # Create a virtual environment at .venv.
source .venv/bin/activate #To activate the virtual environment
uv pip install -e ".[test,learning,model_optimisation]" # Install the package in editable mode
uv pip list --verbose #check versions
pre-commit run -a #pre-commit hooks
```

## :recycle: Model development
Run the following commands from the root directory of your cloned repository to execute the Bash scripts.
We encourage contributions for any enhancements you identify. See our guidelines [here](CONTRIBUTING.md).
* Pre-commit
```bash
bash scripts/activate_pre_commit.bash
```
* Test models
```bash
bash scripts/tests/unit_test_unet_pipeline.bash #TO TEST unet; please edit bash to test other modules
# bash scripts/tests/unit_test_unetvit_pipeline.bash #TO TEST unetvit; please edit bash to test other modules
```
* Train models
```bash
bash scripts/models/train_unet_with_mobious.bash
bash scripts/models/train_unetvit_with_segdataset.bash
```

## :octocat: Cloning repository
* Generate your SSH keys as suggested [here](https://docs.github.com/en/github/authenticating-to-github/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)
* Clone the repository by typing (or copying) the following lines in a terminal
```
mkdir -p ~/repositories/oocular && cd ~/repositories/oocular
git clone git@github.com:oocular/unetvit4sclera.git
```

