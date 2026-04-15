---
library_name: transformers
pipeline_tag: mask-generation
license: apache-2.0
tags:
- vision
---

# Model Card for Segment Anything Model in High Quality (SAM-HQ)

<p align="center">
	<img src="https://huggingface.co/sushmanth/sam_hq_vit_b/resolve/main/assets/arc.png" alt="SAM-HQ Architecture">
	<em> Architecture of SAM-HQ compared to the original SAM model, showing the HQ-Output Token and Global-local Feature Fusion components.</em>
</p>


#  Table of Contents

0. [TL;DR](#TL;DR)
1. [Model Details](#model-details)
2. [Usage](#usage)
3. [Citation](#citation)

# TL;DR

SAM-HQ (Segment Anything in High Quality) is an enhanced version of the Segment Anything Model (SAM) that produces higher quality object masks from input prompts such as points or boxes. While SAM was trained on a dataset of 11 million images and 1.1 billion masks, its mask prediction quality falls short in many cases, particularly when dealing with objects that have intricate structures. SAM-HQ addresses these limitations with minimal additional parameters and computation cost.

The model excels at generating high-quality segmentation masks, even for objects with complex boundaries and thin structures where the original SAM model often struggles. SAM-HQ maintains SAM's original promptable design, efficiency, and zero-shot generalizability while significantly improving mask quality.

# Model Details

SAM-HQ builds upon the original SAM architecture with two key innovations while preserving SAM's pretrained weights:

1. **High-Quality Output Token**: A learnable token injected into SAM's mask decoder that is responsible for predicting high-quality masks. Unlike SAM's original output tokens, this token and its associated MLP layers are specifically trained to produce highly accurate segmentation masks.

2. **Global-local Feature Fusion**: Instead of only applying the HQ-Output Token on mask-decoder features, SAM-HQ first fuses these features with early and final ViT features for improved mask details. This combines both the high-level semantic context and low-level boundary information for more accurate segmentation.

SAM-HQ was trained on a carefully curated dataset of 44K fine-grained masks (HQSeg-44K) compiled from several sources with extremely accurate annotations. The training process takes only 4 hours on 8 GPUs, introducing less than 0.5% additional parameters compared to the original SAM model.

The model has been evaluated on a suite of 10 diverse segmentation datasets across different downstream tasks, with 8 of them evaluated in a zero-shot transfer protocol. Results demonstrate that SAM-HQ can produce significantly better masks than the original SAM model while maintaining its zero-shot generalization capabilities.

SAM-HQ addresses two key problems with the original SAM model:
1. Coarse mask boundaries, often neglecting thin object structures
2. Incorrect predictions, broken masks, or large errors in challenging cases

These improvements make SAM-HQ particularly valuable for applications requiring highly accurate image masks, such as automated annotation and image/video editing tasks.

# Usage

## Prompted-Mask-Generation

```python
from PIL import Image
import requests
from transformers import SamHQModel, SamHQProcessor

model = SamHQModel.from_pretrained("syscv-community/sam-hq-vit-base")
processor = SamHQProcessor.from_pretrained("syscv-community/sam-hq-vit-base")

img_url = "https://raw.githubusercontent.com/SysCV/sam-hq/refs/heads/main/demo/input_imgs/example1.png"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
input_boxes = [[[306, 132, 925, 893]]]  # Bounding box for the image
```

```python
inputs = processor(raw_image, input_boxes=input_boxes, return_tensors="pt").to("cuda")
outputs = model(**inputs)
masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
scores = outputs.iou_scores
```

Among other arguments to generate masks, you can pass 2D locations on the approximate position of your object of interest, a bounding box wrapping the object of interest (the format should be x, y coordinate of the top right and bottom left point of the bounding box), a segmentation mask. At this time of writing, passing a text as input is not supported by the official model according to the official repository. For more details, refer to this notebook, which shows a walkthrough of how to use the model, with a visual example!

## Automatic-Mask-Generation

The model can be used for generating segmentation masks in a "zero-shot" fashion, given an input image. The model is automatically prompted with a grid of `1024` points which are all fed to the model.

The pipeline is made for automatic mask generation. The following snippet demonstrates how easy you can run it (on any device! Simply feed the appropriate `points_per_batch` argument)

```python
from transformers import pipeline
generator = pipeline("mask-generation", model="syscv-community/sam-hq-vit-base", device=0, points_per_batch=256)
image_url = "https://raw.githubusercontent.com/SysCV/sam-hq/refs/heads/main/demo/input_imgs/example1.png"
outputs = generator(image_url, points_per_batch=256)
```

Now to display the image:

```python
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
plt.imshow(np.array(raw_image))
ax = plt.gca()
for mask in outputs["masks"]:
    show_mask(mask, ax=ax, random_color=True)
plt.axis("off")
plt.show()
```

## Complete Example with Visualization

```python
import numpy as np
import matplotlib.pyplot as plt
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  
def show_boxes_on_image(raw_image, boxes):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    for box in boxes:
      show_box(box, plt.gca())
    plt.axis('on')
    plt.show()
def show_points_on_image(raw_image, input_points, input_labels=None):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
      labels = np.ones_like(input_points[:, 0])
    else:
      labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    plt.axis('on')
    plt.show()
def show_points_and_boxes_on_image(raw_image, boxes, input_points, input_labels=None):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
      labels = np.ones_like(input_points[:, 0])
    else:
      labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    for box in boxes:
      show_box(box, plt.gca())
    plt.axis('on')
    plt.show()
def show_points_and_boxes_on_image(raw_image, boxes, input_points, input_labels=None):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
      labels = np.ones_like(input_points[:, 0])
    else:
      labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    for box in boxes:
      show_box(box, plt.gca())
    plt.axis('on')
    plt.show()
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
def show_masks_on_image(raw_image, masks, scores):
    if len(masks.shape) == 4:
      masks = masks.squeeze()
    if scores.shape[0] == 1:
      scores = scores.squeeze()
    nb_predictions = scores.shape[-1]
    fig, axes = plt.subplots(1, nb_predictions, figsize=(15, 15))
    for i, (mask, score) in enumerate(zip(masks, scores)):
      mask = mask.cpu().detach()
      axes[i].imshow(np.array(raw_image))
      show_mask(mask, axes[i])
      axes[i].title.set_text(f"Mask {i+1}, Score: {score.item():.3f}")
      axes[i].axis("off")
    plt.show()
def show_masks_on_single_image(raw_image, masks, scores):
    if len(masks.shape) == 4:
        masks = masks.squeeze()
    if scores.shape[0] == 1:
        scores = scores.squeeze()
    # Convert image to numpy array if it's not already
    image_np = np.array(raw_image)
    # Create a figure
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image_np)
    # Overlay all masks on the same image
    for i, (mask, score) in enumerate(zip(masks, scores)):
        mask = mask.cpu().detach().numpy()  # Convert to NumPy
        show_mask(mask, ax)  # Assuming `show_mask` properly overlays the mask
    ax.set_title(f"Overlayed Masks with Scores")
    ax.axis("off")
    plt.show()

import torch
from transformers import SamHQModel, SamHQProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SamHQModel.from_pretrained("syscv-community/sam-hq-vit-base").to(device)
processor = SamHQProcessor.from_pretrained("syscv-community/sam-hq-vit-base")

from PIL import Image
import requests
img_url = "https://raw.githubusercontent.com/SysCV/sam-hq/refs/heads/main/demo/input_imgs/example1.png"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
plt.imshow(raw_image)

inputs = processor(raw_image, return_tensors="pt").to(device)
image_embeddings, intermediate_embeddings = model.get_image_embeddings(inputs["pixel_values"])

input_boxes = [[[306, 132, 925, 893]]]
show_boxes_on_image(raw_image, input_boxes[0]) 

inputs.pop("pixel_values", None)
inputs.update({"image_embeddings": image_embeddings})
inputs.update({"intermediate_embeddings": intermediate_embeddings})
with torch.no_grad():
    outputs = model(**inputs)
masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
scores = outputs.iou_scores

show_masks_on_single_image(raw_image, masks[0], scores)

show_masks_on_image(raw_image, masks[0], scores)
```

# Citation

```
@misc{ke2023segmenthighquality,
      title={Segment Anything in High Quality}, 
      author={Lei Ke and Mingqiao Ye and Martin Danelljan and Yifan Liu and Yu-Wing Tai and Chi-Keung Tang and Fisher Yu},
      year={2023},
      eprint={2306.01567},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2306.01567}, 
}
```