import math
import torch
import argparse
import numpy as np
from PIL import Image
import util
import resnet50mat

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="model_better.ckpt")
    parser.add_argument("--adobe_dir", default="Combined_Dataset")
    parser.add_argument("--pascal_dir", default="PascalVOC2012")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Results for model.ckpt:
    # |     1 |  85.54571 |  10.07789 |  14.36688 |  88.38642 |  85.54571 |  10.07789 |  14.36688 |  88.38642 | 2007_000027.jpg | 16452523375_08591714cf_o_0.png |
    # ...
    # |  1000 |   3.80314 |   1.53890 |   1.35462 |   2.33773 |  23.75879 |   4.24140 |   7.99427 |  18.96106 | 2008_000491.jpg | woman-morning-bathrobe-bathroom_19.png |

    # Results for model_better.ckpt:
    # |     1 |  87.79555 |  10.14639 |  13.73451 |  92.78490 |  87.79555 |  10.14639 |  13.73451 |  92.78490 | 2007_000027.jpg | 16452523375_08591714cf_o_0.png |
    # ...
    # |  1000 |   3.70049 |   1.46659 |   1.27914 |   2.24583 |  23.41398 |   3.92533 |   7.71221 |  18.82480 | 2008_000491.jpg | woman-morning-bathrobe-bathroom_19.png |

    model = resnet50mat.load(args.checkpoint)
    model.to(args.device)

    # Load lists of foreground and background image names
    with open(f"{args.adobe_dir}/Test_set/test_fg_names.txt") as f:
        fg_names = f.read().strip().replace("\r", "").split("\n")

    with open(f"{args.adobe_dir}/Test_set/test_bg_names.txt") as f:
        bg_names = f.read().strip().replace("\r", "").split("\n")

    bgs_per_fg = len(bg_names) // len(fg_names)

    sads = []
    mses = []
    grads = []
    conns = []

    print("# Composition-1K Test Results\n")
    print("| INDEX | SAD ÷1K | MSE ×1K | Grad ÷1K | Conn ÷1K | AVG SAD | AVG MSE | AVG Grad | AVG Conn | bg_name | trimap_name |")
    print("| ----- | ------- | ------- | -------- | -------- | ------- | ------- | -------- | -------- | ------- | ----------- |")

    # One foreground image is composited onto 20 different background images
    for i_fg in range(len(fg_names)):
        fg_name = fg_names[i_fg]
        fg_path = f"{args.adobe_dir}/Test_set/Adobe-licensed images/fg/{fg_name}"
        alpha_path = f"{args.adobe_dir}/Test_set/Adobe-licensed images/alpha/{fg_name}"

        gt_alpha_pil = Image.open(alpha_path).convert("L")
        fg_pil = Image.open(fg_path).convert("RGB")

        gt_alpha_np = np.array(gt_alpha_pil) / 255.0
        fg_np = np.array(fg_pil) / 255.0

        for i_bg in range(bgs_per_fg):
            i = i_fg * bgs_per_fg + i_bg

            bg_name = bg_names[i]

            trimap_name = fg_name.replace(".png", f"_{i_bg}.png")
            trimap_path = f"{args.adobe_dir}/Test_set/Adobe-licensed images/trimaps/{trimap_name}"
            bg_path = f"{args.pascal_dir}/{bg_name}"

            trimap_pil = Image.open(trimap_path).convert("L")
            bg_pil = Image.open(bg_path).convert("RGB")

            # Resize background to cover foreground if background is smaller
            w, h = fg_pil.size
            bw, bh = bg_pil.size
            scale = max(w / bw, h / bh)
            if scale > 1:
                bw = math.ceil(scale * bw)
                bh = math.ceil(scale * bh)
                bg_pil = bg_pil.resize((bw, bh), Image.BICUBIC)

            trimap_np = np.array(trimap_pil)
            bg_np = np.array(bg_pil) / 255.0

            # Unknown region is neither foreground nor background
            is_fg_np = trimap_np == 255
            is_bg_np = trimap_np == 0
            is_unknown_np = ~(is_fg_np | is_bg_np)

            # Blend foreground and background to create image
            a_np = gt_alpha_np[:, :, np.newaxis]
            image_np = a_np * fg_np + (1 - a_np) * bg_np[:h, :w]

            # Quantize to uint8 and predict alpha matte
            image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
            image_pil = Image.fromarray(image_np)

            alpha_pil = model.predict(image_pil, trimap_pil)

            alpha_np = np.array(alpha_pil) / 255.0

            # Compute and print metrics
            sad = util.compute_sad(alpha_np, gt_alpha_np, is_unknown_np)
            mse = util.compute_mse(alpha_np, gt_alpha_np, is_unknown_np)
            grad = util.compute_grad(alpha_np, gt_alpha_np, is_unknown_np)
            conn = util.compute_conn(alpha_np, gt_alpha_np, is_unknown_np)

            sads.append(sad)
            mses.append(mse)
            grads.append(grad)
            conns.append(conn)

            print(f"| {i + 1:5d} | {sad:9.5f} | {mse:9.5f} | {grad:9.5f} | {conn:9.5f} | {np.mean(sads):9.5f} | {np.mean(mses):9.5f} | {np.mean(grads):9.5f} | {np.mean(conns):9.5f} | {bg_name} | {trimap_name} |")

if __name__ == "__main__":
    main()
