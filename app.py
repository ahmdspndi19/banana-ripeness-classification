import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

IMG = 224  # harus sama dengan ukuran saat training
MODEL_PATH = "models/banana_mobilenet.keras"
CLASS_PATH = "models/classes.json"
AMBANG = 0.8  # di bawah ini muncul peringatan keyakinan rendah

# urutan tampilan dari mentah ke terlalu matang
URUTAN = ["unripe_banana", "ripe_banana", "very_ripe_banana"]
TAMPIL = {
    "unripe_banana": ("Mentah", "🟢"),
    "ripe_banana": ("Matang", "🟡"),
    "very_ripe_banana": ("Terlalu matang", "🟤"),
}

st.set_page_config(page_title="Banana Ripeness Classifier",
                   page_icon="🍌", layout="wide")
st.title("🍌 Klasifikasi Kematangan Pisang")
st.caption("Model CNN (MobileNetV2) untuk mengklasifikasikan pisang: "
           "mentah, matang, atau terlalu matang.")


@st.cache_resource
def muat_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_PATH) as f:
        kelas = json.load(f)
    return model, kelas


model, kelas = muat_model()


def siapkan(img: Image.Image) -> Image.Image:
    """Perbaiki rotasi HP dan ganti latar transparan jadi putih."""
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        putih = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(putih, img)
    return img.convert("RGB")


def prediksi(img: Image.Image):
    x = np.array(img.resize((IMG, IMG), Image.BILINEAR), dtype=np.float32)
    x = np.expand_dims(x, 0)  # tanpa dibagi 255: Rescaling sudah di dalam model
    return model.predict(x, verbose=0)[0]


kiri, kanan = st.columns(2, gap="large")

# ---------- KIRI: input ----------
with kiri:
    st.subheader("Input")
    sumber = st.radio("Sumber gambar", ["Upload foto", "Kamera"], horizontal=True)
    if sumber == "Upload foto":
        file = st.file_uploader("Pilih foto pisang", type=["jpg", "jpeg", "png"])
    else:
        file = st.camera_input("Ambil foto pisang")

    img = None
    if file:
        img = siapkan(Image.open(file))
        st.image(img, caption="Foto input", use_container_width=True)

# ---------- KANAN: hasil ----------
with kanan:
    st.subheader("Hasil")
    if img is None:
        st.info("Upload atau ambil foto pisang di sebelah kiri untuk melihat hasilnya.")
    else:
        prob = prediksi(img)
        idx = int(np.argmax(prob))
        nama, ikon = TAMPIL.get(kelas[idx], (kelas[idx], ""))

        st.markdown(f"## {ikon} {nama}")
        st.metric("Tingkat keyakinan", f"{prob[idx] * 100:.1f}%")

        if prob[idx] < AMBANG:
            st.warning("Keyakinan model rendah. Coba foto dengan cahaya lebih baik, "
                       "satu buah atau satu tandan saja, dan pastikan pisang terlihat jelas.")

        st.markdown("**Probabilitas per kelas**")
        for k in URUTAN:
            if k in kelas:
                p = float(prob[kelas.index(k)])
                label, ik = TAMPIL[k]
                st.write(f"{ik} {label}: **{p * 100:.1f}%**")
                st.progress(min(max(p, 0.0), 1.0))

st.divider()
st.caption("Catatan: model hanya mengenal tiga kelas pisang dan memberi satu label "
           "per foto. Foto non-pisang atau foto berisi beberapa tandan dengan "
           "kematangan berbeda tetap diprediksi sebagai salah satu dari tiga kelas.")