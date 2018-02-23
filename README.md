## Grad-CAM implementation in Pytorch ##

### What makes the network think the image label is 'pug, pug-dog' and 'tabby, tabby cat':
![Dog](https://github.com/ducminhkhoi/pytorch-grad-cam/blob/master/examples/dog.jpg?raw=true) ![Cat](https://github.com/ducminhkhoi/pytorch-grad-cam/blob/master/examples/cat.jpg?raw=true)

Gradient class activation maps are a visualization technique for deep learning networks.

See the paper: https://arxiv.org/pdf/1610.02391v1.pdf

The paper authors torch implementation: https://github.com/ramprs/grad-cam


----------


This uses VGG19 from torchvision. It will be downloaded when used for the first time.

The code can be modified to work with any model.
However the VGG models in torchvision have features/classifier methods for the convolutional part of the network, and the fully connected part.
This code assumes that the model passed supports these two methods.


----------

Differences with original code:
* Rewrite all classes which are more efficient that original ones
* Write new class GuidedGradCam, which combines GradCam and GuidedBackpropReLUModel, is fully differentiable so you can use for another job like feeding the output to another network
* We just need one backpropagation for both CAM and GuidedBackpropagation


----------

Usage: `python grad-cam.py --image-path <path_to_image>`

To use with CUDA:
`python grad-cam.py --image-path <path_to_image> --use-cuda`

To use with VGG19:
`python grad-cam.py --image-path <path_to_image> --model-name=vgg19`
