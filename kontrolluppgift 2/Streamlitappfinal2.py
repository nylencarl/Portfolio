import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage
import streamlit_drawable_canvas
import joblib

st.cache_resource.clear()
st.cache_data.clear()

@st.cache_resource
def load_model():
    return joblib.load("best_svc_mnist_final.pkl")

model = load_model()
siffror = [str(i) for i in range(10)]


# MNIST-style preprocessing

def preprocess_mnist_style(img):
    """
    Gör preprocessing som liknar MNIST:
    - Gråskala
    - Invertera (vit siffra på svart)
    - Bounding box
    - Skala till 20x20
    - Placera i 28x28
    - Center of Mass centrering
    """

    # Ta bort alpha
    img = Image.fromarray(np.array(img)[..., :3].astype("uint8"))
    img = img.convert("L")

    # Invertera
    img = ImageOps.invert(img)

    arr = np.array(img).astype(np.float32)

    #  Hitta bounding box 
    rows = np.any(arr > 20, axis=1)
    cols = np.any(arr > 20, axis=0)

    if not np.any(rows) or not np.any(cols):
        return Image.new("L", (28, 28), 0)

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    digit = arr[y_min:y_max+1, x_min:x_max+1]

    # Skala så längsta sida blir 20 pixlar 
    h, w = digit.shape
    scale = 20.0 / max(h, w)

    digit_resized = ndimage.zoom(digit, zoom=scale, order=1)

    # Placera i 28x28 
    new_img = np.zeros((28, 28), dtype=np.float32)

    h_new, w_new = digit_resized.shape
    y_offset = (28 - h_new) // 2
    x_offset = (28 - w_new) // 2

    new_img[y_offset:y_offset+h_new, x_offset:x_offset+w_new] = digit_resized

    # Center of Mass centrering för att efterlikna MNIST ännu mer
    com_y, com_x = ndimage.center_of_mass(new_img)

    if not np.isnan(com_y) and not np.isnan(com_x):
        shift_y = 14 - com_y
        shift_x = 14 - com_x

        new_img = ndimage.shift(
            new_img,
            shift=(shift_y, shift_x),
            mode="constant",
            cval=0,
            order=1
        )

    return Image.fromarray(np.clip(new_img, 0, 255).astype(np.uint8))


@st.cache_data
def predict_digit(img):

    img_28 = preprocess_mnist_style(img)

    # Debug
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Original", width=150)
    with col2:
        st.image(img_28, caption="MNIST-style 28x28", width=150)

    img_array = np.array(img_28).astype(np.float32) / 255.0
    img_array = img_array.flatten().reshape(1, -1)

    pred = model.predict(img_array)[0]
    prob = model.predict_proba(img_array)[0].max() * 100

    return siffror[int(pred)], prob


# UI 

st.title("🖊️ MNIST")

col1, col2 = st.columns(2)

with col1:
    st.subheader("✏️ Rita var du vill")

    canvas_result = streamlit_drawable_canvas.st_canvas(
        stroke_width=18,
        stroke_color="black",
        background_color="white",
        height=280,
        width=280,
        key="canvas"
    )

    if canvas_result.image_data is not None:
        img = Image.fromarray(canvas_result.image_data.astype("uint8"))
        pred = predict_digit(img)
        st.success(f"🎯 **{pred[0]}** ({pred[1]:.1f}%)")


with col2:
    st.subheader("🖼️ Upload")

    uploaded = st.file_uploader("PNG/JPG", type=["png", "jpg"])

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, width=200)

        if st.button("🔍 Predict"):
            pred = predict_digit(img)
            st.success(f"🎯 **{pred[0]}** ({pred[1]:.1f}%)")


if st.button("🗑️ Rensa"):
    st.rerun()
