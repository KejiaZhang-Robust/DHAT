import torch
import numpy as np
from typing import Callable, List, Tuple, Optional
import ttach as tta
import cv2
import os
from torchvision import datasets, models, transforms
from PIL import Image
from __init__ import *
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, deprocess_image, preprocess_image
)

def scale_cam_image(cam, target_size=None):
    result = []
    for img in cam:
        img = img - np.min(img)
        img = img / (1e-7 + np.max(img))
        if target_size is not None:
            img = cv2.resize(img, target_size)
        result.append(img)
    result = np.float32(result)

    return result

def get_2d_projection(activation_batch):
    activation_batch[np.isnan(activation_batch)] = 0
    projections = []
    for activations in activation_batch:
        reshaped_activations = (activations).reshape(
            activations.shape[0], -1).transpose()
        reshaped_activations = reshaped_activations - \
            reshaped_activations.mean(axis=0)
        U, S, VT = np.linalg.svd(reshaped_activations, full_matrices=True)
        projection = reshaped_activations @ VT[0, :]
        projection = projection.reshape(activations.shape[1:])
        projections.append(projection)
    return np.float32(projections)

class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))
            self.handles.append(
                target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output

        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            return
        
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()

class ClassifierOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]

class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layers: List[torch.nn.Module]):
        super(GradCAM, self).__init__()
        self.model = model.eval()
        self.target_layers = target_layers
        self.device = next(self.model.parameters()).device

        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, None)
        self.tta_transforms = tta.Compose(
                [
                    tta.HorizontalFlip(),
                    tta.Multiply(factors=[0.9, 1, 1.1]),
                ]
            )
    
    def get_cam_image(self,
                      input_tensor: torch.Tensor,
                      target_layer: torch.nn.Module,
                      targets: List[torch.nn.Module],
                      activations: torch.Tensor,
                      grads: torch.Tensor,
                      eigen_smooth: bool = False) -> np.ndarray:

        weights = self.get_cam_weights(grads)
        weighted_activations = weights[:, :, None, None] * activations
        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
        else:
            cam = weighted_activations.sum(axis=1)
        return cam

    def get_cam_weights(self, grads):
        return np.mean(grads, axis=(2, 3))
    
    def forward(self,
                input_tensor: torch.Tensor,
                targets: List[torch.nn.Module],
                eigen_smooth: bool = False) -> np.ndarray:

        input_tensor = input_tensor.to(self.device)

        #TODO:if compute_input_gradient:
        # input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)

        self.outputs = outputs = self.activations_and_grads(input_tensor)

        if targets is None:
            target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
            targets = [ClassifierOutputTarget(
                category) for category in target_categories]

        self.model.zero_grad()
        loss = sum([target(output) for target, output in zip(targets, outputs)])
        loss.backward(retain_graph=True)

        cam_per_layer = self.compute_cam_per_layer(input_tensor,
                                                   targets,
                                                   eigen_smooth)
        return self.aggregate_multi_layers(cam_per_layer)
    
    def compute_cam_per_layer(
            self,
            input_tensor: torch.Tensor,
            targets: List[torch.nn.Module],
            eigen_smooth: bool) -> np.ndarray:
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]

            cam = self.get_cam_image(input_tensor,
                                     target_layer,
                                     targets,
                                     layer_activations,
                                     layer_grads,
                                     eigen_smooth)
            cam = np.maximum(cam, 0)
            scaled = scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer
    
    def get_target_width_height(self,
                                input_tensor: torch.Tensor) -> Tuple[int, int]:
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height
    
    def aggregate_multi_layers(self, cam_per_target_layer: np.ndarray) -> np.ndarray:
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return scale_cam_image(result)
    
    def __call__(self,
                 input_tensor: torch.Tensor,
                 targets: List[torch.nn.Module] = None,
                 aug_smooth: bool = False,
                 eigen_smooth: bool = False) -> np.ndarray:

        # Smooth the CAM result with test time augmentation
        if aug_smooth is True:
            return self.forward_augmentation_smoothing(
                input_tensor, targets, eigen_smooth)

        return self.forward(input_tensor,
                            targets, eigen_smooth)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True


aug_smooth = True
eigen_smooth = True
#TODO:zkj Load the image
# 文件夹路径
folder_path = "Tiny_Imagenet_Val"
input_category = "clean"
model_check = 'ResNet18_ST'

id = 0
image_name = f"{id}.JPEG"
image_path = os.path.join(folder_path, input_category, image_name)
label_path = os.path.join(folder_path, "targets.txt")
with open(label_path, 'r') as file:
    labels = [int(label.strip()) for label in file.readlines()]
targets = [ClassifierOutputTarget(labels[id])]
# targets = None

device = 'cuda' if torch.cuda.is_available() else 'cpu'
check_path = os.path.join('./checkpoint','Tiny-Imagenet', model_check)
net = ResNet18(Num_class=200, Norm=True, norm_mean=torch.tensor([0.485, 0.456, 0.406]), norm_std=torch.tensor([0.229, 0.224, 0.225]))
net = net.to(device)
net = torch.nn.DataParallel(net)  # parallel GPU
checkpoint = torch.load(os.path.join(check_path, 'checkpoint.pth.tar'), map_location=device)
net.load_state_dict(checkpoint['state_dict'])

target_layers = [net.module.layer4]
# 读取图像文件名和标签
image_rgb = Image.open(image_path).convert('RGB')
rgb_show_img = np.float32(image_rgb) / 255
image = transforms.ToTensor()(image_rgb)
input_tensor = image.unsqueeze(0)

cam = GradCAM(model=net, target_layers=target_layers)
cam.batch_size = 32
grayscale_cam = cam(input_tensor=input_tensor,
                    targets=targets,
                    eigen_smooth=eigen_smooth)

grayscale_cam = grayscale_cam[0, :]

cam_image = show_cam_on_image(rgb_show_img, grayscale_cam, use_rgb=True)
cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)


cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])

show_cam = cam_mask
show_cam = (show_cam>0.5).astype(np.float32)
show_cam = cv2.resize(show_cam, (0,0), fx=5, fy=5)
cv2.imshow('CAM Image', show_cam)
cv2.waitKey(0)
cv2.destroyAllWindows()

show_cam = (cam_mask<0.5).astype(np.float32)
show_cam = show_cam * rgb_show_img
show_cam = cv2.cvtColor(show_cam, cv2.COLOR_RGB2BGR)
show_cam = cv2.resize(show_cam, (0,0), fx=5, fy=5)
cv2.imshow('CAM Image', show_cam)
cv2.waitKey(0)
cv2.destroyAllWindows()


show_cam = (cam_mask>0.5).astype(np.float32)
show_cam = show_cam * rgb_show_img
show_cam = cv2.cvtColor(show_cam, cv2.COLOR_RGB2BGR)
show_cam = cv2.resize(show_cam, (0,0), fx=5, fy=5)
cv2.imshow('CAM Image', show_cam)
cv2.waitKey(0)
cv2.destroyAllWindows()

heatmap = cv2.applyColorMap(np.uint8(255 * cam_mask), cv2.COLORMAP_JET)
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
heatmap = np.float32(heatmap) / 255
show_cam = heatmap
show_cam = cv2.resize(show_cam, (0,0), fx=5, fy=5)
cv2.imshow('CAM Image', show_cam)
cv2.waitKey(0)
cv2.destroyAllWindows()
exit()