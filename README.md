# OPERA: Omni-Supervised Representation Learning with Hierarchical Supervisions

This repository is the official implementation of **"OPERA: Omni-Supervised Representation Learning with Hierarchical Supervisions"**. 

![framework](figures/framework.pdf)

![framework](figures/result.pdf)

## Updates

Under construction.

- [x] Pretraining code
- [ ] Pretraining details
- [ ] Pretrained models
- [ ] Downstream tasks code

  

## Preparation

### Dataset

#### ImageNet

- Download from [here](https://www.image-net.org/)

Organize ImageNet as follows:

```
- dataset
    |- train
    |   |- class1
    |   |   |- image1
    |   |   |- ...
    |   |- ...
    |- test
        |- class1
        |   |- image1
        |   |- ...
        |- ...
```

### Pretrained models



### Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

### Device 

We conducted most of the experiments with 8 Nvidia RTX 3090 GPU cards. 

## Training Models

### Baseline method

To pretrain a ResNet50 on ImageNet-1K using MoCo-v3, please run the command as follows:
```python
python train.py
```

### OPERA

To pretrain a ResNet50 on ImageNet-1K using OPERA, please run the command as follows:
```python
python train.py 
```


### More options

There are more options to train various models:
| Args | Options |
| - | - |
| --dataset | mnist / cifar10 / imagenet |
| --model | simple / allconv12 / lenet |
| --optim | sgd / adamw / |



## Acknowledgments

Our code is based on 



## Citation

If you find this project useful in your research, please cite:

````
@article{wang2022opera,
    title={OPERA: Omni-Supervised Representation Learning with Hierarchical Supervisions},
    author={Wang, Chengkun and Zheng, Wenzhao and Zhu, Zheng and Zhou, Jie and Lu, Jiwen},
    year={2022}
}
````
