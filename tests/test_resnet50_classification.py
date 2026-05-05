# Test whether resnet50 implementation for classification matches PyTorch
import os
import sys
from PIL import Image
import numpy as np
import urllib.request

parent_dir = os.path.dirname(os.path.dirname(__file__))

sys.path.append(parent_dir)

import resnet50mat
import torch


def my_preprocess(img, crop_size=224, small_edge_size=256):
    # Match torchvision ImageClassification(crop_size=224): resize shortest edge,
    # center crop, convert uint8 image to float [0, 1], then normalize.
    width, height = img.size
    short, long = (width, height) if width <= height else (height, width)
    new_short = small_edge_size
    new_long = int(new_short * long / short)
    new_width, new_height = (new_short, new_long) if width <= height else (new_long, new_short)
    img = img.resize((new_width, new_height), Image.BILINEAR)

    x = int(round((new_width - crop_size) / 2.0))
    y = int(round((new_height - crop_size) / 2.0))
    img = img.crop((x, y, x + crop_size, y + crop_size))

    tensor = torch.from_numpy(np.array(img, copy=True)).permute(2, 0, 1).to(torch.float32).div_(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


def test_resnet50_classification():
    model = resnet50mat._resnet50()
    checkpoint_path = f"{parent_dir}/resnet50-0676ba61.pth"

    if not os.path.exists(checkpoint_path):
        # https://docs.pytorch.org/vision/0.12/_modules/torchvision/models/resnet.html
        url = "https://download.pytorch.org/models/resnet50-0676ba61.pth"
        print(f"Downloading {checkpoint_path}...")
        urllib.request.urlretrieve(url, checkpoint_path)

    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict)
    model.eval()

    data_dir = os.path.join(parent_dir, "data")

    path = f"{data_dir}/alpaca.png"

    if not os.path.exists(path):
        url = "https://media.githubusercontent.com/media/99991/testing/refs/heads/master/misc/images/alpaca.png"
        print(f"Downloading {path}...")
        os.makedirs(data_dir, exist_ok=True)
        urllib.request.urlretrieve(url, path)

    img = Image.open(path).convert("RGB")
    inputs = my_preprocess(img).unsqueeze(0)
    prediction = model(inputs).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id]

    expected_class_id = 355 # llama
    expected_score = 0.9899351

    assert class_id == expected_class_id, f"Expected {expected_class_id=}, got {class_id=}"
    assert abs(score - expected_score) < 1e-5, f"Expected {expected_score=}, got {score=}"

    print("Test passed. Correctly classified subject as llama.")

def test_torchvision_model():
    from torchvision.models import resnet50, ResNet50_Weights

    weights = ResNet50_Weights.IMAGENET1K_V1
    model = resnet50(weights=weights)
    model.eval()

    preprocess = weights.transforms()

    category_names = weights.meta["categories"]

    path = f"{parent_dir}/data/alpaca.png"

    img = Image.open(path).convert("RGB")

    # Forward torchvision model
    with torch.no_grad():
        inputs = preprocess(img).unsqueeze(0)
        prediction = model(inputs).squeeze(0).softmax(0)

    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = category_names[class_id]
    print(f"{class_id=} ({category_name}): {100 * score:.5f}%")

if __name__ == "__main__":
    test_resnet50_classification()
    # uncomment to obtain expected_class_id
    # currently not in use to reduce dependencies
    #test_torchvision_model()
