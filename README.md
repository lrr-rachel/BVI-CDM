# BVI-CDM

## Dataset

### BVI-RLV

## Dataset Preparation

### 1. Download the dataset

Download the video dataset BVI-RLV from [here](https://dx.doi.org/10.21227/mzny-8c77).

Data Structure
```
.
└── BVI-RLV dataset
    ├── input
    │   ├── S01
    │   │   ├── low_light_10
    │   │   └── low_light_20
    │   ├── S02
    │   │   ├── low_light_10
    │   │   └── low_light_20
    │   └── ...
    └── gt
        ├── S01
        │   ├── normal_light_10
        │   └── normal_light_20
        ├── S02
        │   ├── normal_light_10
        │   └── normal_light_20
        └── ...
```

### 2. Modify the dataset path

Modify the dataset path in the config file `LowLightVideo.yml`. 

Please generate and replace ```data/video_input.txt``` and ```data/video_test.txt``` with your own paths of low-light training and testing images before continue.

### Train and test on other dataset  
Please modify `dataset.py` to train and test on other datasets.

## Train
### Option 1: recompile DCNv2 PyTorch C++ extensions from [BasicSR](https://github.com/XPixelGroup/BasicSR) during installation
```
BASICSR_EXT=True python setup.py develop
```
```
python train.py
```

### Option 2: run from scratch and load the DCNv2 PyTorch C++ extensions just-in-time (JIT)
``` 
BASICSR_JIT=True python train.py
```

## Test
``` 
python evaluate.py
```


## Citation
If you use the BVI-CDM code or BVI-RLV dataset in your research and find this useful, please consider citing our work:
```
@article{Lin:BVI-RLV:2024,
  title={{BVI-RLV: A} Fully Registered Dataset and Benchmarks for Low-Light Video Enhancement},
  author={R Lin and N Anantrasirichai and G Huang and J Lin and Q Sun and A Malyugina and DR Bull},
  journal={arXiv preprint arXiv:2407.03535},
  year={2024}
}
```