from transformers import ViTForImageClassification

def vit_tiny(num_classes=10):
    return ViTForImageClassification.from_pretrained(
        "google/vit-tiny-patch16-224",
        num_labels=num_classes,
    )

def vit_small(num_classes=10):
    return ViTForImageClassification.from_pretrained(
        "google/vit-small-patch16-224",
        num_labels=num_classes,
    )

def vit_base(num_classes=10):
    return ViTForImageClassification.from_pretrained(
        "google/vit-base-patch32-224",
        num_labels=num_classes,
    )

def vit_large(num_classes=10):
    return ViTForImageClassification.from_pretrained(
        "google/vit-large-patch16-224",
        num_labels=num_classes,
    )

def vit_large_32(num_classes=10):
    return ViTForImageClassification.from_pretrained(
        "google/vit-large-patch32-224",
        num_labels=num_classes,
    )

def vit_huge(num_classes=10):
    return ViTForImageClassification.from_pretrained(
        "google/vit-huge-patch16-224",
        num_labels=num_classes,
    )

def vit_huge_32(num_classes=10):
    return ViTForImageClassification.from_pretrained(
        "google/vit-huge-patch32-224",
        num_labels=num_classes,
    )
