import os

import numpy as np

import torch

import sys
sys.path.append("/home/j20morli/Documents/Projects/02_mobile_CLIP/third_party/wise_ft")
from src.models.eval import evaluate
from src.models.finetune import finetune
from src.models.modeling import ClassificationHead, ImageEncoder, ImageClassifier
from src.models.utils import fisher_load
from src.models.zeroshot import get_zeroshot_classifier
from src.args import parse_arguments


def clip_eval(args):
    assert args.save is not None, 'Please provide a path to store models'
    
    if args.load is not None :
        # Zero shot model and Classification Head
        image_encoder = ImageEncoder(args, keep_lang=True)
        classification_head = get_zeroshot_classifier(args, image_encoder.model)
        
        image_encoder.load(args.load)

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
    # alphas = args.alpha
    # for alpha in alphas:
    #     args.alpha = alpha

    #     theta = _merge(alpha, theta_0, theta_1, fishers, args.fisher_floor)

    #     # update the model (in-place) acccording to the new weights
    #     finetuned.load_state_dict(theta)

    #     # save model
    #     finetuned.save(os.path.join(args.save, f'wise_ft_alpha={alpha:.3f}.pt'))

    #     # evaluate
    #     evaluate(finetuned, args)


if __name__ == '__main__':
    args = parse_arguments()
    clip_eval(args)