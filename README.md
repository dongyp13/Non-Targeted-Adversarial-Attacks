# Non-Targeted-Adversarial-Attacks

## Introduction
This repository contains the code for the top-1 submission to [NIPS 2017: Non-targeted Adversarial Attacks Competition](https://www.kaggle.com/c/nips-2017-non-targeted-adversarial-attack).

## Method
We propose a momentum iterative method to generate more transferable adversarial examples. We summarize our algorithm in [Boosting Adversarial Attacks with Momentum](https://arxiv.org/pdf/1710.06081.pdf).

Besically, the update rule of momentum iterative method is

<img src="https://latex.codecogs.com/svg.latex?\Large&space;g_{t+1} = \mu \cdot g_{t} + \frac{\nabla_{x}J(x_{t}^{*},y)}{\|\nabla_{x}J(x_{t}^{*},y)\|_1}, x_{t+1}^{*} = \mathrm{clip}(x_{t}^{*} + \alpha\cdot\mathrm{sign}(g_{t+1})),"/>


### Citation
If you use momentum iterative method for attacks in your research, please consider citing

@article{dong2017boosting,
  title={Boosting Adversarial Attacks with Momentum},
  author={Dong, Yinpeng and Liao, Fangzhou and Pang, Tianyu and Su, Hang and Hu, Xiaolin and Li, Jianguo and Zhu, Jun},
  journal={arXiv preprint arXiv:1710.06081},
  year={2017}
}

## Implementation

### Models
We use the ensemble of eight models in our submission, many of which are adversarially trained models. The models can be downloaded [here](http://ml.cs.tsinghua.edu.cn/~yinpeng/nips17/nontargeted/models.zip).

If you want to attack other models, you can replace the model definition part to your own models.

### Cleverhans
We also implement this method in [Cleverhans](https://github.com/tensorflow/cleverhans/blob/master/cleverhans/attacks.py#L454-L605).

## Targeted Attacks
Please find the targeted attacks at [https://github.com/dongyp13/Targeted-Adversarial-Attack](https://github.com/dongyp13/Targeted-Adversarial-Attack).
