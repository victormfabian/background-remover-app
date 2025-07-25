import streamlit as st 
from rembg import remove, new_session
from PIL import Image, ImageFilter, ImageEnhance
import io
import cv2
import numpy as np

st.set_page_config(page_title="🧼 Background Remover", layout="wide")
st.title("🧼 Smart Background Remover")

# Output resolution choices
res_options = {
    "Original Size": None,
    "HD (1920x1080)": (1920, 1080),
    "2K (2560x1440)": (2560, 1440),
    "4K (3840x2160)": (3840, 2160),
    "8K (7680x4320)": (7680, 4320),
}

# Auto-detect image type for best model

def detect_best_model(image: Image.Image) -> str:
    np_img = np.array(image.convert("RGB"))
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Only use human model if face is large and dominant
    if len(faces) > 0:
        h, w = gray.shape
        x, y, fw, fh = faces[0]
        if fw * fh > 0.1 * w * h:
            return "u2net_human_seg"

    # Heuristic fallback based on brightness/contrast
    brightness = np.mean(np_img)
    contrast = np.std(np_img)
    if contrast > 50 and brightness < 180:
        return "isnet-general-use"

    return "isnet-general-use"

# Improved edge cleaning with feathering

def clean_transparency_edges(img, feather_radius=2):
    np_img = np.array(img)
    alpha = np_img[:, :, 3].astype(np.float32) / 255.0
    alpha_blurred = cv2.GaussianBlur(alpha, (0, 0), sigmaX=feather_radius, sigmaY=feather_radius)
    alpha_blurred = np.clip(alpha_blurred, 0, 1)
    np_img[:, :, 3] = (alpha_blurred * 255).astype(np.uint8)
    return Image.fromarray(np_img)

# Background remover with enhancement

def remove_background_hd(image_data, hd_size=None, softness=0.0, contrast_boost=False, sharpen=False):
    output_image = remove(image_data, session=session)
    img = Image.open(io.BytesIO(output_image)).convert("RGBA")

    # Clean edge artifacts with feathering
    img = clean_transparency_edges(img, feather_radius=2)

    if contrast_boost:
        img = ImageEnhance.Contrast(img).enhance(1.25)
    if sharpen:
        img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
    if softness > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=softness))
    if hd_size:
        img.thumbnail(hd_size, Image.LANCZOS)
        base = Image.new("RGBA", hd_size, (255, 255, 255, 0))
        pos = ((hd_size[0] - img.width) // 2, (hd_size[1] - img.height) // 2)
        base.paste(img, pos, mask=img)
        return base
    return img

# Upload and options
uploaded_file = st.file_uploader("📄 Upload Image", type=["png", "jpg", "jpeg"])
softness = st.slider("🥶 Edge Softness", 0.0, 5.0, 0.0, 0.1)
contrast_boost = st.checkbox("🌆 Enhance Contrast for Better Edge Detection", value=True)
sharpen = st.checkbox("🗑️ Sharpen Details After Removal", value=True)
res_choice = st.selectbox("📏 Export Resolution", list(res_options.keys()), index=0)
background_color = st.color_picker("🎨 Preview Background Color", value="#ffffff")

if uploaded_file:
    image_bytes = uploaded_file.read()
    original_img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")

    with st.spinner("🧠 Detecting best model..."):
        model_choice = detect_best_model(original_img)
    session = new_session(model_choice)
    st.success(f"🧼 Auto-selected model: **{model_choice}**")

    with st.spinner("🧼 Removing background..."):
        result_img_display = remove_background_hd(
            image_bytes,
            hd_size=None,
            softness=softness,
            contrast_boost=contrast_boost,
            sharpen=sharpen,
        )
        result_img_export = remove_background_hd(
            image_bytes,
            hd_size=res_options[res_choice],
            softness=softness,
            contrast_boost=contrast_boost,
            sharpen=sharpen,
        )

    col1, spacer, col2 = st.columns([1.2, 0.1, 1.2])
    with col1:
        st.image(original_img, caption="🖞️ Original", use_container_width=True)
    with col2:
        preview_bg = Image.new("RGBA", result_img_display.size, background_color)
        preview_bg.paste(result_img_display, (0, 0), mask=result_img_display)
        st.image(preview_bg, caption=f"🌟 Background Removed ({res_choice})", use_container_width=True)

    output = io.BytesIO()
    result_img_export.save(output, format="PNG")
    output.seek(0)

    st.download_button(
        "📅 Download PNG",
        data=output,
        file_name=f"bg_removed_{res_choice.replace(' ', '_')}.png",
        mime="image/png"
    )
