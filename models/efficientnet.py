from transformers import EfficientNetForImageClassification

# 定义不同 EfficientNet 模型的返回函数
def efficientnet_b0(num_classes=10):
    return EfficientNetForImageClassification.from_pretrained(
        "efficientnet-b0",
        num_labels=num_classes,
    )

def efficientnet_b1(num_classes=10):
    return EfficientNetForImageClassification.from_pretrained(
        "efficientnet-b1",
        num_labels=num_classes,
    )

def efficientnet_b2(num_classes=10):
    return EfficientNetForImageClassification.from_pretrained(
        "efficientnet-b2",
        num_labels=num_classes,
    )
