from transformers import SwinForImageClassification

# 定义不同 Swin Transformer 模型的返回函数
def swin_tiny(num_classes=10):
    return SwinForImageClassification.from_pretrained(
        "microsoft/swin-tiny-patch4-window7-224",
        num_labels=num_classes,
    )

def swin_base(num_classes=10):
    return SwinForImageClassification.from_pretrained(
        "microsoft/swin-base-patch4-window7-224",
        num_labels=num_classes,
    )

def swin_large(num_classes=10):
    return SwinForImageClassification.from_pretrained(
        "microsoft/swin-large-patch4-window7-224",
        num_labels=num_classes,
    )
