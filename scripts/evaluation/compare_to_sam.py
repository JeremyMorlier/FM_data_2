import torch
import cv2
import glob
import json
import random
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from segment_anything.modeling import *  
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator



images = glob.glob("/users2/local/sam_dataset_validation/*.jpg")
np.random.shuffle(images)
print(len(images))

sam_checkpoint = "/users2/local/r17bensa/sam_vit_b_01ec64.pth"
model_type = "vit_b"

device = "cuda"

sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
sam.to(device=device)

#sam_neighbor = sam_model_registry["vit_b_n"](checkpoint=sam_checkpoint)
#sam_neighbor.to(device=device)

tiny_sam_checkpoint = "/users2/local/r17bensa/mobile_sam.pt"
sam_tmp = sam_model_registry["vit_t"](checkpoint=tiny_sam_checkpoint)
sam_tmp.to(device=device)
sam_tmp.image_encoder.target_img_size = 1024

#sam_neighbor.image_encoder(torch.randn(1,3,1024,1024).to(device = device))
#reda = torch.load("/users2/local/r17bensa/SAM_9_attention_epoch_9.pth")
#dict = sam_neighbor.image_encoder.state_dict()
#for k in sam_neighbor.image_encoder.state_dict() :
#    sam_neighbor.image_encoder.state_dict()[k] = 0

#sam_neighbor.image_encoder.load_state_dict(reda.state_dict())


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    


def calculate_iou(segmentation1, segmentation2):
    intersection = np.sum(segmentation1 * segmentation2)
    union = np.sum(segmentation1) + np.sum(segmentation2) - intersection
    iou = (intersection / union) * 100
    return iou


iou_liste = []
times_normal = []
times_neighbor = []

print(device)
for i, image_path in enumerate(images) : 
    print(image_path)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    f = open(image_path.replace('.jpg','.json'))
    label = json.load(f)
    index = random.randint(0,len(label['annotations'])-1)
    input_point = np.array(label['annotations'][index]['point_coords'])
    input_label = np.array([1])

    predictor = SamPredictor(sam)
    predictor.set_image(image, upscale = False)
    mask, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )

    predictor_neighbor = SamPredictor(sam_tmp)
    predictor_neighbor.set_image(image, upscale = False)
    mask_neighbor, _, _ = predictor_neighbor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )

    print(i)
    print(calculate_iou(mask*1,mask_neighbor*1))
    iou_liste.append(calculate_iou(mask*1,mask_neighbor*1))
    print(np.mean(iou_liste))

    print("#####################################################")