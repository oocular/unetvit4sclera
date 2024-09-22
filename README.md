# unetvit
UNet-ViT model with train and inference workflows!

## Source venv and pre-commit
```
source .venv/bin/activate
pre-commit run -a
```

## Model
### Train
```
python src/unetvit/apis/train.py 
```
### Convert
```
python src/unetvit/utils/pytorch2onnx.py -i <model_name>.pth
```


## Test
See [tests](tests)


## Clone repo
```
git clone git@github.com:mxochicale/unetvit.git
```

## Reference
https://www.kaggle.com/code/ganaianimesh/unet-with-vit
