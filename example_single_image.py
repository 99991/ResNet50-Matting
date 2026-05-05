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
