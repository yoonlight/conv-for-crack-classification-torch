"""
# Visualization

## Reference

- [GradCAM code example](https://github.com/jacobgil/pytorch-grad-cam)
- [Convert image to Tensor](https://www.geeksforgeeks.org/converting-an-image-to-a-torch-tensor-in-python/)
- [Input shape in no batch](https://stackoverflow.com/questions/75747275/running-pytorch-model-at-inference-i-e-with-batch-size-1-and-not-the-batch-si)

"""
import os
from pathlib import Path

import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms
from PIL import Image

from run import CrackClassifier
from models.alexnet import AlexNet
from models.lenet import LeNet5
from models.shallow import ShallowCNN

# AlexNet 54ruy0xl
# LeNet zaxsq5qd
# Shaollow CNN wgizk72m
# epoch=29-step=5010.ckpt
MODEL_PATH = "./lightning_logs/wgizk72m/checkpoints/epoch=29-step=5010.ckpt"

IMAGE_PATH = Path("./datasets/test/Negative")

image_list = os.listdir(IMAGE_PATH)
image_name = image_list[1]
image = Image.open(IMAGE_PATH / image_name)

transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
])

# Load model

model = ShallowCNN()
model = CrackClassifier.load_from_checkpoint(
    MODEL_PATH, model=model, learning_rate=1e-4)

model: ShallowCNN = model.model

model.eval()
# layer = 
target_layers = [model.conv2]

# Load GradCAM

cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
targets = [ClassifierOutputTarget(0)]
input_tensor = transform(image)
grayscale_cam = cam(input_tensor=input_tensor.unsqueeze(0), targets=targets)

grayscale_cam = grayscale_cam[0, :]
image = image.resize((224, 224))
rgb_img = np.float32(image) / 255
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

# save image

result = Image.fromarray(visualization)
result.save(f"outputs/shallow/negative/{image_name}-result-conv2.jpg")
