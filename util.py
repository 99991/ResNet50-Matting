import os
import zlib
import struct
import urllib.request
import scipy.ndimage
import numpy as np
from PIL import Image
from io import BytesIO

def decode_png_idat(idat_data, width, height, header=[8, 2, 0, 0, 0]):
    def write_chunk(chunk_type, chunk_data):
        f.write(struct.pack(">I", len(chunk_data)))
        f.write(chunk_type)
        f.write(chunk_data)
        f.write(struct.pack(">I", zlib.crc32(chunk_type + chunk_data)))

    f = BytesIO()
    f.write(b"\x89PNG\r\n\x1a\n")
    write_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, *header))
    write_chunk(b"IDAT", idat_data)
    write_chunk(b"IEND", b"")
    f.seek(0)
    return Image.open(f)

def download_test_images(directory="data"):
    # backup URL in case the other URL is down
    # https://web.archive.org/web/20211023132308/https://sjtrny.com/files/10.1109_DICTA.2012.6411686.pdf
    url = "https://sjtrny.com/files/10.1109_DICTA.2012.6411686.pdf"
    filename = url.split("/")[-1]

    if os.path.exists(os.path.join(directory, "GT04.png")):
        return

    print(f"Downloading test images...")
    urllib.request.urlretrieve(url, filename)

    with open(filename, "rb") as f:
        data = f.read()

    os.remove("10.1109_DICTA.2012.6411686.pdf")

    # Three example images with hardcoded offsets into the PDF
    example_image_data = [
        ("GT04_trimap.png", data[2828574:2828574+13271]),
        ("GT04_alpha.png", data[5275869:5275869+211014]),
        ("GT04.png", data[6939898:6939898+840881]),
    ]
    width = 800
    height = 563

    os.makedirs(directory, exist_ok=True)

    for filename, image_data in example_image_data:
        image = decode_png_idat(image_data, width, height)
        image.save(os.path.join(directory, filename))

def compute_mse(pred, target, is_unknown):
    return np.mean(np.square(pred - target)[is_unknown]) * 1000

def compute_sad(pred, target, is_unknown):
    return np.sum(np.abs(pred - target)[is_unknown]) / 1000

def correlate_x(image, kernel):
    return scipy.ndimage.correlate(image, kernel.reshape(1, -1), mode="nearest")

def correlate_y(image, kernel):
    return correlate_x(image.T, kernel).T

def calculate_gradient(image, sigma):
    r = int(3 * sigma)

    # Compute 1D Gaussian kernel and its derivative
    x = np.linspace(-r, r, 2 * r + 1)
    g = np.exp(-0.5 * np.square(x) / (sigma * sigma)) / (sigma * np.sqrt(2 * np.pi))
    dg = -1.0 / sigma ** 2 * x * g

    # Normalize such that sum is 1
    g /= np.linalg.norm(g)
    dg /= np.linalg.norm(dg)

    dx = correlate_y(correlate_x(image, dg), g)
    dy = correlate_x(correlate_y(image, dg), g)

    return np.sqrt(dx * dx + dy * dy)

def compute_grad(pred, target, is_unknown, sigma=1.4):
    g1 = calculate_gradient(pred, sigma)
    g2 = calculate_gradient(target, sigma)

    return np.sum(np.square(g1 - g2)[is_unknown]) / 1000

def largest_connected_component(mask):
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    labeled, num = scipy.ndimage.label(mask, structure=structure)
    counts = np.bincount(labeled.ravel())
    counts[labeled[mask == 0][0]] = 0
    return labeled == np.argmax(counts)

def compute_conn(pred, target, is_unknown, steps=10):
    l_map = np.full(pred.shape, -1.0)

    for i in range(steps):
        thresh_lo = (i + 0) / steps
        thresh_hi = (i + 1) / steps

        pred_thresh = pred >= thresh_hi
        target_thresh = target >= thresh_hi

        omega = largest_connected_component(pred_thresh & target_thresh)

        l_map[(l_map == -1) & ~omega] = thresh_lo

    l_map[l_map == -1] = 1.0

    pred_d = pred - l_map
    target_d = target - l_map

    pred_phi = 1.0 - pred_d * (pred_d >= 0.15)
    target_phi = 1.0 - target_d * (target_d >= 0.15)

    return np.sum(np.abs(pred_phi - target_phi)[is_unknown]) / 1000

def test():
    download_test_images()

    pred = np.array(Image.open("data/predicted_alpha.png").convert("L")) / 255.0
    target = np.array(Image.open("data/GT04_alpha.png").convert("L")) / 255.0
    trimap = np.array(Image.open("data/GT04_trimap.png").convert("L"))
    is_unknown = trimap == 128

    sad = compute_sad(pred, target, is_unknown)
    mse = compute_mse(pred, target, is_unknown)
    grad = compute_grad(pred, target, is_unknown)
    conn = compute_conn(pred, target, is_unknown)

    print(f"SAD: {sad:.10f}")
    print(f"MSE: {mse:.10f}")
    print(f"Grad: {grad:.10f}")
    print(f"Conn: {conn:.10f}")

    expected_sad = 6.7441
    expected_mse = 4.0378
    expected_grad = 2.3806
    expected_conn = 5.5687

    assert abs(sad - expected_sad) < 1e-4
    assert abs(mse - expected_mse) < 1e-4
    assert abs(grad - expected_grad) < 1e-4
    assert abs(conn - expected_conn) < 1e-4

if __name__ == "__main__":
    test()
