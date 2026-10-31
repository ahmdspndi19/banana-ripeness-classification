import os, glob, json
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

IMG = 224
model = tf.keras.models.load_model("models/banana_cnn.keras")
kelas = json.load(open("models/classes.json"))
DATA = os.path.join("data", "dataset0", "test")

rng = np.random.default_rng(0)
for i, k in enumerate(kelas):
    paths = glob.glob(os.path.join(DATA, k, "*"))
    pilih = rng.choice(paths, 30, replace=False)
    X = np.stack([
        np.array(ImageOps.exif_transpose(Image.open(p)).convert("RGB")
                 .resize((IMG, IMG), Image.BILINEAR), dtype=np.float32)
        for p in pilih])
    pred = model.predict(X, verbose=0).argmax(axis=1)
    print(f"{k}: {(pred == i).mean() * 100:.0f}% benar dari 30 foto")