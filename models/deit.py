from transformers import DeiTForImageClassification
from utils.utils import ModelWrapper

def deit_tiny(num_classes=10):
    return ModelWrapper(DeiTForImageClassification.from_pretrained(
        "facebook/deit-tiny-distilled-patch16-224",
        num_labels=num_classes,
    ))

def deit_small(num_classes=10):
    return ModelWrapper(DeiTForImageClassification.from_pretrained(
        "facebook/deit-small-distilled-patch16-224",
        num_labels=num_classes,
    ))

def deit_base(num_classes=10):
    return ModelWrapper(DeiTForImageClassification.from_pretrained(
        "facebook/deit-base-distilled-patch16-224",
        num_labels=num_classes,
    ))
