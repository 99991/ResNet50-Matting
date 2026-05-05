# ResNet-50 Image Matting Baseline

This repository provides a robust image matting baseline leveraging a **ResNet-50** backbone, inspired by the training philosophies of **AEMatter**.

## 📥 Model Download

Pre-trained checkpoints are available on Hugging Face:

**🔗 [https://huggingface.co/Coldswamp/ResNet50-Matting](https://huggingface.co/Coldswamp/ResNet50-Matting)**

You can download the checkpoints using the `huggingface_hub` library:

```python
from huggingface_hub import hf_hub_download

# Download the better-performing checkpoint
hf_hub_download(repo_id="Coldswamp/ResNet50-Matting", filename="model_better.ckpt")

# Or download the standard checkpoint
hf_hub_download(repo_id="Coldswamp/ResNet50-Matting", filename="model.ckpt")
```

Or via the Hugging Face CLI:

```bash
huggingface-cli download Coldswamp/ResNet50-Matting
```

---

## 📊 Experimental Results

We evaluate our model using standard matting metrics. The results below demonstrate the performance of the provided checkpoints.

| Checkpoint | SAD ↓ | MSE ↓ | Grad ↓ | Conn ↓ |
| :--- | :--- | :--- | :--- | :--- |
| `model.ckpt` | 23.82677 | 0.00427 | 8.08990 | 19.02270 |
| **`model_better.ckpt`** | **23.47848** | **0.00396** | **7.80796** | **18.89092** |

> **Note:** `model_better.ckpt` represents the state-of-the-art (SOTA) performance for this baseline configuration.

---

## 💡 Key Optimization: Eval-Mode Fine-Tuning

The superior performance of `model_better.ckpt` is attributed to a specific optimization strategy: **Fine-tuning the entire network in `eval()` mode** for a final epoch.

### Technical Intuition
In many image restoration tasks (such as Super-Resolution and Matting), Batch Normalization (BN) layers can introduce noise if the mini-batch statistics fluctuate significantly during the final stages of convergence.

To achieve a more stable and high-performance result:
1. **The Strategy**: After standard training, we perform one additional fine-tuning epoch.
2. **The Mechanism**: The model is switched to `.eval()` mode, but gradients are **not** frozen.
3. **The Benefit**: By keeping the BN layers in evaluation mode, we use the stable, pre-calculated global running statistics instead of batch-specific statistics. This allows the weights to strictly optimize for the loss objective without being affected by batch-level variance.
4. **Validation**: This approach has been proven effective in various low-level vision benchmarks to refine details and improve global consistency.

---

## 🛠 Reproducing the Result

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
