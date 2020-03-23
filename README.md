# [MutualNet: Adaptive ConvNet via Mutual Learning from Network Width and Resolution](https://arxiv.org/abs/1909.12978)
This work proposes a method to train a network that is executable at dynamic resource constraints. The proposed mutual learning scheme significantly improves the accuracy-efficiency tradeoffs over [Slimmable Networks](https://github.com/JiahuiYu/slimmable_networks). The method is also promising to serve as a plug-and-play strategy to boost a single network.

The code is based on the implementation of [Slimmable Networks](https://github.com/JiahuiYu/slimmable_networks).
# Install
- PyTorch 1.0.1, torchvision 0.2.2, Numpy, pyyaml 5.1.
- Follow the PyTorch [example](https://github.com/pytorch/examples/tree/master/imagenet) to prepare ImageNet dataset.
# Run
## Training
```
python train_mutualnet.py app:apps/mobilenet_v1.yml
```
Training hyperparameters are in the .yml files.
## Testing

Modify '*test_only: False*' to '*test_only: True*' in .yml file to enable testing.

Modify '*pretrained: /PATH/TO/YOUR/WEIGHTS*' to assign trained weights.
```
python train_mutualnet.py app:apps/mobilenet_v1.yml
```
# Results
## Performance over the whole FLOPs specturm
Comparison with US-Net \[1\] under different backbones.

![Results compared with US-Net](imgs/result1.JPG)
## Scaling up model compared with EfficienNet
The best model scaling on MobileNet v1 compared with EfficientNet \[2\]
|Model|Best Model Scaling|FLOPs|Top-1 Acc|
|-----|------------------|-----|---------|
EfficienNet \[2\]|*d=1.4, w=1.2, r=1.3*|2.3B|75.6%|
MutualNet|*w=1.6, r=1.3*|2.3B|**77.1%**|
## Boosting performance of a single network
Top-1 accuracy on Cifar-10 and Cifar-100
|WideResNet-28-10|GPU search hours|Cifar-10|Cifar-100|
|----------------|:--------------:|:------:|:-------:|
|Baseline|0|96.1%|81.2%|
|Cutout|0|96.9%|81.6|
|AutoAug|5000|**97.4%**|82.9|
|Fast AutoAug|3.5|97.3%|82.7%|
|MutualNet|0|97.2%|**83.8%**|

Compared with popular performance boosting methods on ImageNet
|ResNet-50|Additional Cost|Top-1 Acc|
|---------|:-------------:|:-------:|
|Baseline| \ |76.5%|
|Cutout| \ |77.1%|
|KD|Teacher Network|76.5%|
|SENet|SE Block|77.6%|
|AutoAug|15000 GPU search hours|77.6%|
|Fast AutoAug|450 GPU search hours|77.6%|
|MutualNet| \ |**78.6%**|
# Reference
\[1\] Yu, Jiahui, and Thomas S. Huang. "Universally slimmable networks and improved training techniques." Proceedings of the IEEE International Conference on Computer Vision. 2019.

\[2\] Tan, Mingxing, and Quoc Le. "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." International Conference on Machine Learning. 2019.
