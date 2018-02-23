import torch
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import numpy as np
import argparse
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from torch import nn

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])


class ModelOutputs:
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers."""

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers

    def __call__(self, x):
        self.model.zero_grad()
        target_activations = []
        for name, module in self.model.features._modules.items():
            x = module(x)
            if name in self.target_layers:
                target_activations += [x]

        x = x.view(x.size(0), -1)
        output = self.model.classifier(x)
        return target_activations, output


def show_cam_on_image(img, mask, file_name):
    mask = mask.data.cpu().numpy().transpose((1, 2, 0))
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    img *= torch.FloatTensor(std)[:, None, None]
    img += torch.FloatTensor(mean)[:, None, None]
    img = img.permute(1, 2, 0).numpy()

    cam = heatmap + img
    cam = cam / np.max(cam)
    cv2.imwrite(file_name, np.uint8(255 * cam))


class GradCam(nn.Module):
    def __init__(self, model, target_layer_names):
        super(GradCam, self).__init__()
        self.model = ModelOutputs(model, target_layer_names)

    def forward(self, input, index=None):
        features, output = self.model(input)

        if index is None:
            one_hot = output.max()  # choose the target output, in this case, choose the maximum activation
        else:
            one_hot = output[index]

        feature = features[-1]
        feature.retain_grad()

        one_hot.backward(torch.ones_like(one_hot), retain_graph=True)

        grads_val, target = feature.grad, feature
        weights = grads_val.mean(-1).mean(-1)[:, :, None, None]

        cam = ((weights * target).sum(1) + 1)
        cam = torch.clamp(cam, min=0)

        min = cam.min(-1)[0].min(-1)[0][:, None, None]
        max = cam.max(-1)[0].max(-1)[0][:, None, None]
        cam = (cam - min) / (max - min)

        mask = F.upsample(cam[:, None, :, :], (224, 224), mode='bilinear')
        return mask


class GuidedBackpropReLU(Function):

    def forward(self, input):
        output = torch.clamp(input, min=0)
        self.save_for_backward(input, output)
        return output

    def backward(self, grad_output):
        input, _ = self.saved_tensors
        grad_input = grad_output.clone()

        grad_input[(input < 0) | (grad_output < 0)] = 0

        return grad_input

    def _apply(self, fn):
        pass

    def named_parameters(self, memo=None, prefix=''):
        return []


class GuidedBackpropReLUModel(nn.Module):
    def __init__(self, model):
        super(GuidedBackpropReLUModel, self).__init__()
        self.model = model

        # replace ReLU with GuidedBackpropReLU
        for idx, module in self.model.features._modules.items():
            if module.__class__.__name__ == 'ReLU':
                self.model.features._modules[idx] = GuidedBackpropReLU()

    def forward(self, input, index=None):
        output = self.model(input)
        input.retain_grad()

        if index is None:
            one_hot = output.max()  # choose the target output, in this case, choose the maximum activation
        else:
            one_hot = output[index]

        one_hot.backward(torch.ones_like(one_hot), retain_graph=True)

        output = input.grad[0]

        return output


class GuidedGradCam(nn.Module):

    def __init__(self, args):
        super(GuidedGradCam, self).__init__()
        model = getattr(models, args.model_name)(pretrained=True)
        self.grad_cam = GradCam(model=model, target_layer_names=target_layer_name)
        self.gbp = GuidedBackpropReLUModel(model=model)

    def forward(self, input, target_index):
        mask = self.grad_cam(input, target_index)
        gb = self.gbp(input, target_index)
        cam_gb = mask * gb[None, ...]
        return mask, gb, cam_gb


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./examples/both.png',
                        help='Input image path')
    parser.add_argument('--model-name', type=str, default='vgg19',
                        help='Input image path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args


if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    args = get_args()

    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    target_index = None
    target_layer_name = ['35']

    img = Image.open(args.image_path)
    img = transform(img)[None, :, :, :]
    input = Variable(img, requires_grad=True)

    model = GuidedGradCam(args)

    if args.use_cuda:
        model = model.cuda()
        input = input.cuda()

    cam, gb, cam_gb = model(input, target_index)  # fully differentiable

    # visualize and save example
    show_cam_on_image(img[0], cam[0], 'cam.jpg')
    utils.save_image(gb.data, 'gb.jpg')
    utils.save_image(cam_gb.data, 'cam_gb.jpg')
