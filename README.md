# unetvit4sclera
Open-Source UNet-ViT Workflow for Debugging, Training, and Evaluating Sclera Segmentation

## :nut_and_bolt: Dev installation
```
uv venv --python 3.12 # Create a virtual environment at .venv.
source .venv/bin/activate #To activate the virtual environment
uv pip install -e ".[test,learning,model_optimisation]" # Install the package in editable mode
uv pip list --verbose #check versions
pre-commit run -a #pre-commit hooks
```
See further details for installation [here](docs).

## :recycle: Model development
* Pre-commit
```bash
bash scripts/activate_pre_commit.bash
```
* Test model train and optimisation
```bash
bash scripts/tests/unit_test_ml_pipeline.bash
```

## :octocat: Cloning repository
* Generate your SSH keys as suggested [here](https://docs.github.com/en/github/authenticating-to-github/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)
* Clone the repository by typing (or copying) the following lines in a terminal
```
mkdir -p ~/repositories/oocular && cd ~/repositories/oocular
git clone git@github.com:oocular/unetvit4sclera.git
```

