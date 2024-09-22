# unetvit
UNet-ViT model with train and inference workflows!

## Source venv and pre-commit
```
source .venv/bin/activate
pre-commit run -a
```

## Tests
```
pytest -vv tests/test_inference.py
pytest -vs tests/test_inference.py::test_segDataset
pytest -vs tests/test_inference.py::test_inference
```

## Clone repo
```
git clone git@github.com:mxochicale/unetvit.git
```

## Reference
https://www.kaggle.com/code/ganaianimesh/unet-with-vit
