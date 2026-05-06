# ResNet-50 Image Matting Baseline

This repository provides a robust image matting baseline leveraging a ResNet-50 backbone, inspired by the training philosophies of [**AEMatter**](https://github.com/aipixel/AEMatter).

| **Input image** | **Input trimap** | **Output alpha matte** |
|--------------------------|-----------------|------------------|
| ![image](https://raw.githubusercontent.com/frcs/alternative-matting-laplacian/master/GT04.png) | ![trimap](https://raw.githubusercontent.com/frcs/alternative-matting-laplacian/master/trimap-GT04.png) | ![alpha](https://github.com/user-attachments/assets/2e16a9dd-babc-4a1b-9a6c-113a65096d02) |

## [Example](/example_single_image.py)

```python
import resnet50mat
from PIL import Image
import util
util.download_test_images()

# Downloads model from https://huggingface.co/Coldswamp/ResNet50-Matting
model = resnet50mat.load("model_better.ckpt")

image = Image.open("data/GT04.png")
trimap = Image.open("data/GT04_trimap.png")

alpha = model.predict(image, trimap)

alpha.save("data/predicted_alpha.png")

print("Predicted alpha matte saved to data/predicted_alpha.png")
```

## Usage

```bash
git clone https://github.com/Windaway/ResNet50-Matting.git
cd ResNet50-Matting
pip install -r requirements.txt
python example_single_image.py
```

## Results on Composition-1K Test Set

| Checkpoint | SAD ↓ | MSE ↓ | Grad ↓ | Conn ↓ |
| :--- | :--- | :--- | :--- | :--- |
| `model.ckpt` | 23.82677 | 0.00427 | 8.08990 | 19.02270 |
| **`model_better.ckpt`** | **23.47848** | **0.00396** | **7.80796** | **18.89092** |

Note: `model_better.ckpt` represents the state-of-the-art performance for this baseline configuration.

To reproduce the test results, run [`test_composition_1k_dataset.py`](/test_composition_1k_dataset.py):

1. Ask [Brain Price](https://arxiv.org/pdf/1703.03872) to send you `Adobe_Deep_Matting_Dataset.zip`, place it in this directory and unzip it. The name of the resulting directory will be `Combined_Dataset`.
2. Download and extract the [Pascal VOC2012 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit) dataset.
4. Run `python test_composition_1k_dataset.py --adobe_dir Combined_Dataset/ --pascal_dir VOCdevkit/VOC2012/JPEGImages/`
    * This takes about 11 minutes and 8147MiB of VRAM on an RTX 3060 (12GB) GPU.

## Pretrained Model

Pre-trained checkpoints are available on Hugging Face. They will be downloaded automatically when instantiating the model.

* [https://huggingface.co/Coldswamp/ResNet50-Matting](https://huggingface.co/Coldswamp/ResNet50-Matting)

## Key Optimization: Eval-Mode Fine-Tuning

The superior performance of `model_better.ckpt` is attributed to a specific optimization strategy: Fine-tuning the entire network in `eval()` mode for a final epoch.

### Technical Intuition

In many image restoration tasks (such as Super-Resolution and Matting), batch normalization layers can introduce noise if the mini-batch statistics fluctuate significantly during the final stages of convergence.

To achieve a more stable result:

1. After standard training, we perform one additional fine-tuning epoch.
2. The model is switched to `.eval()` mode, but gradients are not frozen.
3. By keeping the batch normalization layers in evaluation mode, we use the stable, pre-calculated global running statistics instead of batch-specific statistics. This allows the weights to strictly optimize for the loss objective without being affected by batch-level variance.

To replicate the performance of `model_better.ckpt`, ensure you run the final fine-tuning step as follows:

```python
# Final Fine-tuning in Eval Mode
model.eval()  # Use global BN stats, but keep gradients active
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # Use a smaller LR

for images, trimaps, gt in final_refinement_loader:
    optimizer.zero_grad()
    output = model(images, trimaps)
    loss = criterion(output, gt)
    loss.backward()
    optimizer.step()

# Save the refined model
save_checkpoint(model, "model_better.ckpt")
```
