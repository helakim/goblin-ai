## Expansion Pyramid Network for high resolution face detection (Only-U)
![backbone_1](fig/backbone-1.png)

---

| **`Status`** | **`Pyhon version`** | **`Language`** | **`Windows`** | **`Crated`** | **`Description`** | **`build`** |
|---------------------|------------------|-------------------|---------------|---------------|---------------|---------------|
|![Status](https://img.shields.io/pypi/status/Django.svg) |![Python Version](https://img.shields.io/pypi/pyversions/Django.svg)|![Code Language](https://img.shields.io/badge/python3.6-100%25-red.svg)| ![Window Supported](https://img.shields.io/badge/supported-not-orange.svg) |![Created by](https://img.shields.io/badge/Sunday%2029%2C%20Sep%202019-hela.kim-ff69b4.svg)|![Description](https://img.shields.io/badge/FaceDetection-Model-yellowgreen.svg)|![build](https://img.shields.io/circleci/token/YOURTOKEN/project/github/RedSparr0w/node-csgo-parser/master.svg)

---

# **Only-U (Sunday 29, sep 2019)**

1. **Only-U** is implemented by PyTorch and CuDA and does not use external deep learning models.

2. The backbone model is a model that borrows the idea of the existing deep residual network and adds a transition block and is called **Block Wise Network**.

3. Although the detector model is similar to the U-NET, the 2-path way strategy can be used to communicate the amount of pixel information to the convolution unit more than the U-NET.

4. We use FP16 to reduce training times by 3 to 4 times compared to FP32.

## Installation (Dependencies)
    Todo 

## Only-U requires:
    1. pytorch 1.2.0
    2. torchvision >= 0.3.0
    3. opencv-contrib >= 3.3.49
    4. CUDA >= 10.0 and CuDnn >= 8.0

## Training DataSet
 - Deep Learning model learning data used Korean face image on ai-hub
 - [Ai-Hub Korean Face DataSet download link](http://www.aihub.or.kr/content/606)
 
## **Only-U** Training (Coming Soon)
```
python model_train.py -gpu=0,1 \  # multi-gpu 
                      -data_set=samples/train/annotation.json \
                      -val_set=samples/val/annotation.json \
                      -backbone=default # choose [atrous50, atrous101, atrous152, res50, res101, res152] \
                      -save_path=train_results \
                      -visdom_use=True
```

## **Only-U** Single Inference (Coming Soon)
 - Todo

## Inference Results
- multi-person facial detection (by google.com)
![test_case_1](results/test_case_1_inference_result.jpg)


- basic facial detection (by google.com)
![test_case_2](results/test_case_2_inference_result.jpg)


- 19th Century Person Face detection (by google.com)
![face_case_3](results/test_case_3_inference_result.jpg)


- validation image detection (by ai-hub.co.kr)
![face_case_4](results/test_case_4_inference_result.jpg)


## Todo
 - [ ] Deep Learning Model(Only-U) training code
 - [ ] Deep Learning Model(Only-U) inference code
 - [ ] Add reference

## Author
 - kyung tae kim (firefoxdev0619@gmail.com)

## References

```
@misc{Only-U 2019,
  author =       {kyung tae kim},
  title =        {Only-U},
  howpublished = {\url{https://github.com/helakim/goblin-ai/tree/master/pytorch_version/face_detection}},
  year =         {2019}
}
```

## Contact
For any question, feel free to contact :)
```
kyung tae kim     : firefoxdev0619@gmail.com
```