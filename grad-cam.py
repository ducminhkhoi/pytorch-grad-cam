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


class MyModel(nn.Module):
    """ Define your model here"""

    def __init__(self, args):
        super(MyModel, self).__init__()
        model = getattr(models, args.model_name)(pretrained=True)
        self.base_model = nn.Sequential(*list(model.features.children())[:-1])
        self.classifier = model.classifier

        # replace ReLU with GuidedBackpropReLU
        modules = {**self.base_model._modules, **self.classifier._modules}
        for idx, module in modules.items():
            if isinstance(module, nn.ReLU):
                modules[idx] = GuidedBackpropReLU()

    def __call__(self, x):

        x1 = self.base_model(x)
        x = F.max_pool2d(x1, 2)
        x = x.view(x.size(0), -1)
        output = self.classifier(x)

        return x1, output


def show_cam_on_image(img, mask, file_name):
    mask = mask.permute(1, 2, 0).cpu().numpy()
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    img *= torch.FloatTensor(std)[:, None, None]
    img += torch.FloatTensor(mean)[:, None, None]
    img = img.permute(1, 2, 0).numpy()

    cam = heatmap + img
    cam = cam / np.max(cam)
    cv2.imwrite(file_name, np.uint8(255 * cam))


class GuidedGradCam(nn.Module):
    def __init__(self, args):
        super(GuidedGradCam, self).__init__()
        self.model = MyModel(args)
        self.model.eval()

    def forward(self, input, indices=None):
        feature, output = self.model(input)

        feature.retain_grad()
        input.retain_grad()

        if indices:
            indices = Variable(torch.LongTensor(indices)).long().view(-1, 1).cuda()
            one_hot = output.gather(1, indices)
        else:
            one_hot = output.max(-1)[0]  # choose the target output, in this case, choose the maximum activations

        self.model.zero_grad()

        one_hot.backward(torch.ones_like(one_hot))

        grads_val, target = feature.grad, feature
        grads_val = F.relu(grads_val)
        weights = F.adapative_avg_pool2d(grads_val, 1)

        cam = ((weights * target).sum(1) + 1)
        cam = F.relu(cam)

        min = -F.max_pool2d(-cam, cam.size(-1))
        max = F.max_pool2d(cam, cam.size(-1))
        cam = (cam - min) / (max - min)

        mask = F.upsample(cam[:, None, :, :], (224, 224), mode='bilinear')

        gb = input.grad
        cam_gb = mask * gb
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
    target_index = [243, 281]

    img = Image.open(args.image_path)
    img = transform(img)[None, :, :, :].repeat(2, 1, 1, 1)
    input = Variable(img, requires_grad=True)

    model = GuidedGradCam(args)

    if args.use_cuda:
        model = model.cuda()
        input = input.cuda()

    cam, gb, cam_gb = model(input, target_index)  # fully differentiable

    # visualize and save example
    for i in range(cam.size(0)):
        show_cam_on_image(img[i], cam[i].data, 'examples/cam_{}.jpg'.format(i))
        utils.save_image(gb[i].data, 'examples/gb_{}.jpg'.format(i))
        utils.save_image(cam_gb[i].data, 'examples/cam_gb_{}.jpg'.format(i))
