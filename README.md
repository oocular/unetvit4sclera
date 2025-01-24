# unetvit
UNet-ViT model with train and inference workflows!

## Source venv and pre-commit
```
source .venv/bin/activate
pre-commit run -a
```

## Model train and conversion
```
# Train
python src/unetvit/apis/train.py 
# Convert
python src/unetvit/utils/pytorch2onnx.py -i <model_name>.pth
```

## Test inference
See [tests](tests)


## :octocat: Cloning repository
* Generate your SSH keys as suggested [here](https://docs.github.com/en/github/authenticating-to-github/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)
* Clone the repository by typing (or copying) the following lines in a terminal
```
mkdir -p ~/repositories && cd ~/repositories
git clone git@github.com:mxochicale/unetvit.git
```

## Reference
https://www.kaggle.com/code/ganaianimesh/unet-with-vit
