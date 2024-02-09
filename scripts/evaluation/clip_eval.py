import os
import sys
sys.path.append(sys.path[0] + "/../../third_party/wise_ft/")
sys.path.append(sys.path[0] + "/../../")
print(sys.path)
import numpy as np

import torch
import torch.utils as utils
import open_clip
from torchvision.transforms import v2

from src.models.eval import evaluate
from src.models.finetune import finetune
from src.models.modeling import ClassificationHead, ImageEncoder, ImageClassifier
from src.models.utils import fisher_load
from src.models.zeroshot import get_zeroshot_classifier
from src.args import parse_arguments

import time
from scripts.common import transform_pilTensor

def clip_eval(args):
    print(args)

    # Use GPU if available otherwise CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args.save = "/users/local/j20morli/mobile_clip/result_eval/"
    args.data_location="/nasbrain/j20morli/eval_clip/"
    args.load = "/users/local/j20morli/mobile_clip/data/result_distillation/RN50-quickgelu_ViT-S-32-alt_yfcc_8.pth"
    args.eval_datasets = ["ImageNet", "ImageNetV2", "ImageNetR", "ImageNetA", "ImageNetSketch"]
    args.train_dataset = "ImageNet"
    args.model = "ViT-S-32-alt"
    args.result_db = "/users/local/j20morli/mobile_clip/results/results.json1"
    args.device = device
    args.template = "openai_imagenet_template"
    args.transform = transform_pilTensor

    teacher_name = 'RN50-quickgelu'
    models = ["ViT-S-32-alt", "RN50", "convnext_tiny", "ViT-S-16-alt"]
    checkpoints = [10, 100, 1000, 5000, 10000, 20000, 30000, 40000, 10000]
    data_names = ["yfcc", "random", "white"]
    assert args.save is not None, 'Please provide a path to store models'
    
    if args.load is not None :
        # Zero shot model and Classification Head
        image_encoder = ImageEncoder(args, keep_lang=True)
        classification_head = get_zeroshot_classifier(args, image_encoder.model)
        
        #image_encoder.load(args.load)

        delattr(image_encoder.model, 'transformer')
        
        classifier = ImageClassifier(image_encoder, classification_head, process_images=False)
        zeroshot_checkpoint = os.path.join(args.save, 'zeroshot.pt')
        classifier.save(zeroshot_checkpoint)

        args.load = zeroshot_checkpoint
        args.save = os.path.join(args.save, 'finetuned.pt')
        finetuned_checkpoint = finetune(args)
    else :
        print("Error please provide initial weights")
        return None

    # Load models
    zeroshot = ImageClassifier.load(zeroshot_checkpoint)
    finetuned = ImageClassifier.load(finetuned_checkpoint)

    evaluate(zeroshot, args)
    evaluate(finetuned, args)

if __name__ == '__main__':

    args = parse_arguments()

    teacher_name = 'RN50-quickgelu'
    models = ["ViT-S-32-alt", "RN50", "convnext_tiny", "ViT-S-16-alt"]
    checkpoints = [10, 100, 1000, 5000, 10000, 20000, 30000, 40000, 10000]
    data_names = ["yfcc", "random", "white"]

    # Use GPU if available otherwise CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # args.save = "/users/local/j20morli/mobile_clip/result_eval/"
    # args.data_location = "/nasbrain/j20morli/"
    # args.eval_datasets = ["ImageNet", "ImageNetV2", "ImageNetR", "ImageNetA", "ImageNetSketch"]
    # args.train_dataset = "ImageNet"
    # args.result_db = "/users/local/j20morli/mobile_clip/results/results.json1"
    # args.device = device
    # args.template = "openai_imagenet_template"

    # for model in models :
    #     for data_name in data_names :
    #         for checkpoint in checkpoints :
    #             name = teacher_name + "_" + model + "_" + data_name + "_" + str(checkpoint)
    #             args.load = "/users/local/j20morli/mobile_clip/result_distillation/" + name + ".pth"
    #             args.model = model
    #             clip_eval(args)
                
    clip_eval(args)