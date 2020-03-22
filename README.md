# [MutualNet: Adaptive ConvNet via Mutual Learning from Network Width and Resolution](https://arxiv.org/abs/1909.12978)
This work proposes a method to train a network that is executable at dynamic resource constraints. The proposed mutual learning scheme significantly improves the accuracy-efficiency tradeoffs over [Slimmable Networks](https://github.com/JiahuiYu/slimmable_networks). The method is also promising to serve as a regularization method to boost a single network.

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
![Results compared with US-Net](imgs/result1.JPG)
## Scale up model compared woth EfficienNet

|Model|Best Model Scaling|FLOPs|Top-1 Acc|
EfficienNet|*d=1.4, w=1.2, r=1.3*|2.3B|75.6%|
MutualNet|*w=1.6, r=1.3*|2.3B|**77.1|
## Boost performance of a single network
