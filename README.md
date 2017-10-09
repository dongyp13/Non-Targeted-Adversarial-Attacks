# Non-Targeted-Adversarial-Attacks

## Method
We propose an iterative fast gradient sign approach with momentum. We noticed that by using momentum, transferability (black-box attack accuracy) is also improved when increasing the iterations. The previous conclusion about "iterative attacks reduce transferability" is not true, at least in our method.

## Models
We use the ensemble of eight models in our implementation. Many of them are adversarially trained models. The models can be downloaded [here](https://ml.cs.tsinghua.edu.cn/~yinpeng/nips17/nontargeted/models.zip).
