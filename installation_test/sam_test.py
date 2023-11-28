import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

import sys
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

path = "third_party/segment-anything/"
notebook_path = path + "notebooks/"
example_path = "installation_test/"

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


image = cv2.imread(notebook_path + 'images/dog.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



plt.figure(figsize=(20,20))
plt.imshow(image)
plt.axis('off')
plt.savefig(example_path + "test.png")




sam_checkpoint = example_path + "sam_vit_b_01ec64.pth"
model_type = "vit_b"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)

masks = mask_generator.generate(image)

print(len(masks))
print(masks[0].keys())

plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.savefig(example_path + "masked_image.png")