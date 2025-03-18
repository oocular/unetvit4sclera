# Aerial imagery

## Download 
Download dataset https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery?resource=download

* Using terminal
```bash
mkdir -p ~/datasets/unetvit && cd ~/datasets/unetvit
wget -O semantic-segmentation-of-aerial-imagery.zip https://www.kaggle.com/api/v1/datasets/download/humansintheloop/semantic-segmentation-of-aerial-imagery
unzip semantic-segmentation-of-aerial-imagery.zip
mv Semantic\ segmentation\ dataset/ semantic-segmentation-dataset
rm semantic-segmentation-of-aerial-imagery.zip
```

## Dataset
```
~/datasets/unetvit/semantic-segmentation-dataset$ tree -d
.
├── Tile 1
│   ├── images
│   └── masks
├── Tile 2
│   ├── images
│   └── masks
├── Tile 3
│   ├── images
│   └── masks
├── Tile 4
│   ├── images
│   └── masks
├── Tile 5
│   ├── images
│   └── masks
├── Tile 6
│   ├── images
│   └── masks
├── Tile 7
│   ├── images
│   └── masks
└── Tile 8
    ├── images
    └── masks

24 directories, 145 files
```

## Models
```
~/datasets/unetvit/models$ tree -h
[4.0K]  .
├── [1.5K]  losses_190.6092257.cvs
├── [387M]  unetvit_epoch_12_0.55486.pth
└── [387M]  unetvit_epoch_5_0.59060.pth

0 directories, 3 files
```
