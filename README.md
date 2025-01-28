# unetvit4sclera
Open-Source UNet-ViT Workflow for Sclera Segmentation

## Source venv and pre-commit
```
source .venv/bin/activate
pre-commit run -a
```

## Model train and conversion
```
# Train
python src/unetvit4sclera/apis/train.py 
# Convert
python src/unetvit4sclera/utils/pytorch2onnx.py -i <model_name>.pth
```

## Test inference
See [tests](tests)


## :octocat: Cloning repository
* Generate your SSH keys as suggested [here](https://docs.github.com/en/github/authenticating-to-github/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)
* Clone the repository by typing (or copying) the following lines in a terminal
```
mkdir -p ~/repositories && cd ~/repositories
git clone git@github.com:mxochicale/unetvit4sclera.git
```

## Reference
https://www.kaggle.com/code/ganaianimesh/unet-with-vit
