# Ensure that mean squared error for GT04.png is small.
# Dataset: https://alphamatting.com/datasets.php
import os
import sys
from PIL import Image
import numpy as np

parent_dir = os.path.dirname(os.path.dirname(__file__))

sys.path.append(parent_dir)

import util
import resnet50mat

def test():
    data_dir = os.path.join(parent_dir, "data")

    util.download_test_images(data_dir)

    model = resnet50mat.load(f"{parent_dir}/model.ckpt")

    image = Image.open(f"{data_dir}/GT04.png").convert("RGB")
    trimap = Image.open(f"{data_dir}/GT04_trimap.png").convert("L")

    alpha = model.predict(image, trimap)

    alpha.save(f"{data_dir}/predicted_alpha.png")

    expected_alpha = Image.open(f"{data_dir}/GT04_alpha.png").convert("L")

    alpha = np.array(alpha) / 255.0
    expected_alpha = np.array(expected_alpha) / 255.0
    is_unknown = np.array(trimap) == 128

    sad = util.compute_sad(alpha, expected_alpha, is_unknown)
    mse = util.compute_mse(alpha, expected_alpha, is_unknown)
    grad = util.compute_grad(alpha, expected_alpha, is_unknown)
    conn = util.compute_conn(alpha, expected_alpha, is_unknown)

    print(f"SAD: {sad:.10f}")
    print(f"MSE: {mse:.10f}")
    print(f"Grad: {grad:.10f}")
    print(f"Conn: {conn:.10f}")

    expected_sad = 6.7441
    expected_mse = 4.0378
    expected_grad = 2.3806
    expected_conn = 5.5687

    # 1e-4 might still be better, but using 1e-3 to allow CUDA slop
    max_error = 1e-3

    assert abs(sad - expected_sad) < max_error
    assert abs(mse - expected_mse) < max_error
    assert abs(grad - expected_grad) < max_error
    assert abs(conn - expected_conn) < max_error

    print("Tests passed.")

if __name__ == "__main__":
    test()
