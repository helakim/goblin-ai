---

| **`Status`** | **`Pyhon version`** | **`Language`** | **`Windows`** | **`Crated`** | **`Description`** | **`build`** |
|---------------------|------------------|-------------------|---------------|---------------|---------------|---------------|
|![Status](https://img.shields.io/pypi/status/Django.svg) |![Python Version](https://img.shields.io/pypi/pyversions/Django.svg)|![Code Language](https://img.shields.io/badge/python3.6-100%25-red.svg)| ![Window Supported](https://img.shields.io/badge/supported-not-orange.svg) |![Created by](https://img.shields.io/badge/Sunday%2029%2C%20Sep%202019-hela.kim-ff69b4.svg)|![Description](https://img.shields.io/badge/FaceDetection-Model-yellowgreen.svg)|![build](https://img.shields.io/circleci/token/YOURTOKEN/project/github/RedSparr0w/node-csgo-parser/master.svg)

---

### detector backbone (Only-U) 
![backbone_1](fig/backbone-1.png)

# **Only-U (Sunday 29, sep 2019)**

1. **Only-U** is implemented by PyTorch and CuDA and does not use external deep learning models.

2. The backbone model is a model that borrows the idea of the existing deep residual network and adds a transition block and is called **Block Wise Network**.

3. Although the detector model is similar to the U-NET, the 2-path way strategy can be used to communicate the amount of pixel information to the convolution unit more than the U-NET.

4. We use FP16 to reduce training times by 3 to 4 times compared to FP32.

## Installation (Denpendencies)

#### Only-U requires:
    1. pytorch 1.2.0
    2. torchvision >= 0.3.0
    3. opencv-contrib >= 3.3.49
    4. CUDA >= 10.0 and CuDnn >= 8.0
    
#### Todo
 - Todo


#### For more information
 - Todo

#### Author
 - kyung tae kim (firefoxdev0619@gmail.com)

#### References