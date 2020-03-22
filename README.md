# [MutualNet](https://arxiv.org/abs/1909.12978)
This work proposes a method to train a network that is executable at dynamic resource constraints. The proposed mutual learning scheme significantly improves the accuracy-efficiency tradeoffs over [Slimmable Networks](https://github.com/JiahuiYu/slimmable_networks). The method is also promising to serve as a regularization method to boost a single network.

The code is based on the implementation of [Slimmable Networks](https://github.com/JiahuiYu/slimmable_networks).
# Install
- PyTorch 1.0.1, torchvision 0.2.2, Numpy, pyyaml 5.1.
- Follow the PyTorch [example](https://github.com/pytorch/examples/tree/master/imagenet) to prepare ImageNet dataset.
# Run
- **Training**
```
python train_mutualnet.py app:apps/mobilenet_v1.yml
```
Training hyperparameters are in the .yml files.
- **Testing**

Modify *test_only: False* to *test_only: True* in the .yml file.

Modify *pretrained: /PATH/TO/YOUR/WEIGHTS* to assign trained weights.
```
python train_mutualnet.py app:apps/mobilenet_v1.yml
```
# Results
TODO
