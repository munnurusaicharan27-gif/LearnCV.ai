# Fixing string formatting issue and regenerating files

import json, os, datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

base_dir = "/mnt/data/ImageToolkit_Project"
os.makedirs(base_dir, exist_ok=True)

app_py = r'''# Image Processing & Analysis Toolkit (Streamlit + OpenCV)
# Author: Your Name | Roll No: ______
# Date: __DATE__
#
# How to run:
#   pip install streamlit opencv-python numpy pillow matplotlib
#   streamlit run app.py
#
# Notes:
# - Streamlit doesn't support a native top menu; we simulate a File menu with buttons.
# - Webcam "realtime" requires streamlit-webrtc (optional). Snapshot via st.camera_input works without extras.
import io
import time
import cv2
import numpy as np
from PIL import Image, ImageOps
import streamlit as st

st.set_page_config(page_title="Image Processing & Analysis Toolkit", layout="wide")

# ----------------- Helpers -----------------
def pil_to_cv(img_pil):
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def cv_to_pil(img_cv):
    if len(img_cv.shape)==2:
        return Image.fromarray(img_cv)
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

def ensure_uint8(img):
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

def get_image_info(img_pil, raw_bytes=None, filename="uploaded"):
    w, h = img_pil.size
    mode = img_pil.mode
    fmt = getattr(img_pil, "format", None) or (filename.split(".")[-1].upper() if "." in filename else "N/A")
    dpi = img_pil.info.get("dpi", (None, None))
    size_bytes = len(raw_bytes) if raw_bytes else None
    channels = len(img_pil.getbands())
    return {
        "width": w, "height": h, "channels": channels, "mode": mode,
        "format": fmt, "dpi": dpi, "file_size_bytes": size_bytes
    }

def contrast_stretch(img_cv):
    # per-channel linear stretching
    out = np.zeros_like(img_cv)
    if len(img_cv.shape)==2:
        lo, hi = np.percentile(img_cv, (1, 99))
        out = cv2.normalize(img_cv, None, 0, 255, cv2.NORM_MINMAX)
    else:
        for c in range(3):
            ch = img_cv[:,:,c]
            lo, hi = np.percentile(ch, (1, 99))
            ch = np.clip((ch - lo) * (255.0 / max(1, (hi - lo))), 0, 255)
            out[:,:,c] = ch
    return ensure_uint8(out)

def equalize_color(img_bgr):
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    return cv2.cvtColor(cv2.merge([y_eq, cr, cb]), cv2.COLOR_YCrCb2BGR)

def sharpen(img_bgr, amount=1.0):
    blur = cv2.GaussianBlur(img_bgr, (0,0), sigmaX=1.0)
    # Unsharp mask
    sharp = cv2.addWeighted(img_bgr, 1+amount, blur, -amount, 0)
    return sharp

def encode_image(img_cv, fmt="PNG", quality=95):
    fmt = fmt.upper()
    if fmt == "JPG" or fmt == "JPEG":
        success, enc = cv2.imencode(".jpg", img_cv, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    elif fmt == "PNG":
        success, enc = cv2.imencode(".png", img_cv, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    elif fmt == "BMP":
        success, enc = cv2.imencode(".bmp", img_cv)
    else:
        success, enc = cv2.imencode(".png", img_cv)
    if not success:
        raise RuntimeError("Encoding failed")
    return enc.tobytes()

def split_screen(original_bgr, processed_bgr):
    h, w = original_bgr.shape[:2]
    proc = cv2.resize(processed_bgr, (w, h), interpolation=cv2.INTER_AREA)
    half = w//2
    out = original_bgr.copy()
    out[:, half:] = proc[:, half:]
    return out

# ----------------- UI: Header / File menu -----------------
st.title("📸 Image Processing & Analysis Toolkit")
st.caption("GUI in Python • OpenCV • Streamlit")

col_menu1, col_menu2, col_menu3, col_menu4 = st.columns([1,1,1,6])
with col_menu1:
    open_btn = st.button("📂 Open")
with col_menu2:
    save_btn = st.button("💾 Save")
with col_menu3:
    exit_btn = st.button("🚪 Exit")

uploaded = st.file_uploader("Upload an image", type=["png","jpg","jpeg","bmp"])

# Optional camera snapshot (not real-time video)
cam_img = st.camera_input("Or capture from webcam (snapshot)", label_visibility="collapsed")

if exit_btn:
    st.info("App terminated by user.")
    st.stop()

# ----------------- Sidebar: Operations -----------------
st.sidebar.header("🧰 Operations")
op_category = st.sidebar.selectbox(
    "Category",
    ["Image Info","Color Conversions","Transformations","Filtering & Morphology",
     "Enhancement","Edge Detection","Compression", "Bitwise Ops", "Bonus: Split View"]
)

# Load image
src_pil = None
raw_bytes = None
filename = None

if cam_img is not None:
    src_pil = Image.open(cam_img)
    raw_bytes = cam_img.getvalue()
    filename = "camera.png"
elif uploaded is not None:
    src_pil = Image.open(uploaded)
    raw_bytes = uploaded.getvalue()
    filename = uploaded.name

if src_pil is None:
    st.warning("Please upload or capture an image to begin.")
    st.stop()

src_pil = ImageOps.exif_transpose(src_pil).convert("RGB")
src_bgr = pil_to_cv(src_pil)

# ----------------- Right Panel: Two columns -----------------
left_col, right_col = st.columns(2, gap="large")

with left_col:
    st.subheader("Original")
    st.image(src_pil, use_column_width=True)

processed_bgr = src_bgr.copy()
status_msgs = []

# ----------------- Operations -----------------
if op_category == "Image Info":
    info = get_image_info(src_pil, raw_bytes, filename)
    with right_col:
        st.subheader("Image Info")
        st.json(info)
elif op_category == "Color Conversions":
    choice = st.sidebar.radio("Convert to", ["BGR→RGB","RGB→HSV","RGB→YCbCr","RGB→Gray"])
    with right_col:
        st.subheader("Processed")
        if choice == "BGR→RGB":
            processed_bgr = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB)
        elif choice == "RGB→HSV":
            processed_bgr = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2HSV)  # src is BGR
        elif choice == "RGB→YCbCr":
            processed_bgr = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2YCrCb)
        elif choice == "RGB→Gray":
            gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)
            st.image(gray, use_column_width=True, clamp=True)
            processed_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        st.image(cv_to_pil(processed_bgr), use_column_width=True)

elif op_category == "Transformations":
    tchoice = st.sidebar.selectbox("Transform", ["Rotation","Scaling","Translation","Affine","Perspective"])
    if tchoice == "Rotation":
        angle = st.sidebar.slider("Angle (deg)", -180.0, 180.0, 15.0, 1.0)
        scale = st.sidebar.slider("Scale", 0.1, 3.0, 1.0, 0.1)
        h, w = src_bgr.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
        processed_bgr = cv2.warpAffine(src_bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    elif tchoice == "Scaling":
        fx = st.sidebar.slider("fx", 0.1, 3.0, 1.2, 0.1)
        fy = st.sidebar.slider("fy", 0.1, 3.0, 1.2, 0.1)
        processed_bgr = cv2.resize(src_bgr, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
    elif tchoice == "Translation":
        tx = st.sidebar.slider("Shift X (px)", -300, 300, 50, 1)
        ty = st.sidebar.slider("Shift Y (px)", -300, 300, 50, 1)
        h, w = src_bgr.shape[:2]
        M = np.float32([[1,0,tx],[0,1,ty]])
        processed_bgr = cv2.warpAffine(src_bgr, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    elif tchoice == "Affine":
        h, w = src_bgr.shape[:2]
        offset = st.sidebar.slider("Corner offset (px)", -100, 100, 30, 1)
        pts1 = np.float32([[0,0],[w-1,0],[0,h-1]])
        pts2 = np.float32([[0+offset,0],[w-1,0+offset],[0,h-1-offset]])
        M = cv2.getAffineTransform(pts1, pts2)
        processed_bgr = cv2.warpAffine(src_bgr, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    elif tchoice == "Perspective":
        h, w = src_bgr.shape[:2]
        off = st.sidebar.slider("Corner perspective (px)", -150, 150, 60, 1)
        src_pts = np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
        dst_pts = np.float32([[0+off,0+off],[w-1-off,0+off],[w-1-off,h-1-off],[0+off,h-1-off]])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        processed_bgr = cv2.warpPerspective(src_bgr, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    with right_col:
        st.subheader("Processed")
        st.image(cv_to_pil(processed_bgr), use_column_width=True)

elif op_category == "Filtering & Morphology":
    fchoice = st.sidebar.selectbox("Operation", ["Gaussian","Median","Mean","Sobel","Laplacian","Dilation","Erosion","Opening","Closing"])
    k = st.sidebar.slider("Kernel size (odd)", 3, 31, 5, 2)
    if fchoice in ["Dilation","Erosion","Opening","Closing"]:
        iterations = st.sidebar.slider("Iterations", 1, 5, 1, 1)
        kernel = np.ones((k,k), np.uint8)
        if fchoice == "Dilation":
            processed_bgr = cv2.dilate(src_bgr, kernel, iterations=iterations)
        elif fchoice == "Erosion":
            processed_bgr = cv2.erode(src_bgr, kernel, iterations=iterations)
        elif fchoice == "Opening":
            processed_bgr = cv2.morphologyEx(src_bgr, cv2.MORPH_OPEN, kernel, iterations=iterations)
        else:
            processed_bgr = cv2.morphologyEx(src_bgr, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    elif fchoice == "Gaussian":
        processed_bgr = cv2.GaussianBlur(src_bgr, (k,k), 0)
    elif fchoice == "Median":
        processed_bgr = cv2.medianBlur(src_bgr, k)
    elif fchoice == "Mean":
        processed_bgr = cv2.blur(src_bgr, (k,k))
    elif fchoice == "Sobel":
        dx = st.sidebar.selectbox("dx", [1,0])
        dy = st.sidebar.selectbox("dy", [0,1])
        gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)
        sob = cv2.Sobel(gray, cv2.CV_64F, dx, dy, ksize=k)
        processed_bgr = cv2.convertScaleAbs(sob)
        processed_bgr = cv2.cvtColor(processed_bgr, cv2.COLOR_GRAY2BGR)
    elif fchoice == "Laplacian":
        gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=k)
        processed_bgr = cv2.convertScaleAbs(lap)
        processed_bgr = cv2.cvtColor(processed_bgr, cv2.COLOR_GRAY2BGR)
    with right_col:
        st.subheader("Processed")
        st.image(cv_to_pil(processed_bgr), use_column_width=True)

elif op_category == "Enhancement":
    echoice = st.sidebar.selectbox("Enhance", ["Histogram Equalization","Contrast Stretching","Sharpening"])
    if echoice == "Histogram Equalization":
        gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)
        eq_gray = cv2.equalizeHist(gray)
        color_mode = st.sidebar.radio("Apply to", ["Grayscale (Y only)","Color (Y channel)"])
        if color_mode == "Color (Y channel)":
            processed_bgr = equalize_color(src_bgr)
        else:
            processed_bgr = cv2.cvtColor(eq_gray, cv2.COLOR_GRAY2BGR)
    elif echoice == "Contrast Stretching":
        processed_bgr = contrast_stretch(src_bgr)
    else:
        amount = st.sidebar.slider("Sharpen Amount", 0.0, 2.0, 1.0, 0.1)
        processed_bgr = sharpen(src_bgr, amount=amount)
    with right_col:
        st.subheader("Processed")
        st.image(cv_to_pil(processed_bgr), use_column_width=True)

elif op_category == "Edge Detection":
    echoice = st.sidebar.selectbox("Edge", ["Sobel","Canny","Laplacian"])
    gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)
    if echoice == "Sobel":
        k = st.sidebar.slider("Kernel (odd)", 3, 31, 5, 2)
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k)
        mag = cv2.magnitude(sx, sy)
        edges = cv2.convertScaleAbs(mag)
    elif echoice == "Canny":
        t1 = st.sidebar.slider("Threshold1", 0, 255, 100, 1)
        t2 = st.sidebar.slider("Threshold2", 0, 255, 200, 1)
        edges = cv2.Canny(gray, t1, t2)
    else:
        k = st.sidebar.slider("Kernel (odd)", 1, 31, 3, 2)
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=k)
        edges = cv2.convertScaleAbs(lap)
    processed_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    with right_col:
        st.subheader("Processed")
        st.image(edges, use_column_width=True, clamp=True)

elif op_category == "Compression":
    fmt = st.sidebar.selectbox("Save format", ["JPG","PNG","BMP"])
    quality = 95
    if fmt == "JPG":
        quality = st.sidebar.slider("JPEG quality", 10, 100, 90, 1)
    # Re-encode original and show size
    orig_bytes = encode_image(src_bgr, fmt=fmt, quality=quality)
    processed_bgr = src_bgr
    with right_col:
        st.subheader("Encoded Preview")
        st.image(cv_to_pil(processed_bgr), use_column_width=True)
        st.write(f"Encoded as **.{fmt.lower()}** | Size: **{len(orig_bytes)} bytes** | Quality: {quality if fmt=='JPG' else 'N/A'}")
        st.download_button(f"Download .{fmt.lower()}", data=orig_bytes, file_name=f"processed.{fmt.lower()}")

elif op_category == "Bitwise Ops":
    st.sidebar.info("Upload a second image (same size recommended)")
    up2 = st.sidebar.file_uploader("Second image", type=["png","jpg","jpeg","bmp"], key="second")
    if up2 is not None:
        img2 = Image.open(up2).convert("RGB")
        img2 = ImageOps.exif_transpose(img2)
        bgr2 = pil_to_cv(img2)
        bgr2 = cv2.resize(bgr2, (src_bgr.shape[1], src_bgr.shape[0]))
        op = st.sidebar.selectbox("Operation", ["AND","OR","XOR","NOT (on first)"])
        if op == "AND":
            processed_bgr = cv2.bitwise_and(src_bgr, bgr2)
        elif op == "OR":
            processed_bgr = cv2.bitwise_or(src_bgr, bgr2)
        elif op == "XOR":
            processed_bgr = cv2.bitwise_xor(src_bgr, bgr2)
        else:
            processed_bgr = cv2.bitwise_not(src_bgr)
    else:
        st.sidebar.warning("Upload second image for bitwise operations.")
    with right_col:
        st.subheader("Processed")
        st.image(cv_to_pil(processed_bgr), use_column_width=True)

elif op_category == "Bonus: Split View":
    st.sidebar.info("Half original + half processed (choose a base operation below)")
    base = st.sidebar.selectbox("Base effect", ["Sharpen","Gaussian Blur","Canny Edges"])
    if base == "Sharpen":
        amount = st.sidebar.slider("Sharpen Amount", 0.0, 2.0, 1.0, 0.1)
        proc = sharpen(src_bgr, amount)
    elif base == "Gaussian Blur":
        k = st.sidebar.slider("Kernel (odd)", 3, 31, 9, 2)
        proc = cv2.GaussianBlur(src_bgr, (k,k), 0)
    else:
        t1 = st.sidebar.slider("Canny t1", 0, 255, 80, 1)
        t2 = st.sidebar.slider("Canny t2", 0, 255, 160, 1)
        edges = cv2.Canny(cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY), t1, t2)
        proc = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    processed_bgr = split_screen(src_bgr, proc)
    with right_col:
        st.subheader("Split View")
        st.image(cv_to_pil(processed_bgr), use_column_width=True)

# ----------------- Save processed (right panel image) -----------------
if save_btn:
    # default save as PNG
    data = encode_image(processed_bgr, fmt="PNG")
    st.download_button("⬇️ Download Processed (PNG)", data=data, file_name="processed.png")

# ----------------- Status Bar -----------------
st.markdown("---")
info = get_image_info(src_pil, raw_bytes, filename)
st.text(f"Status • Dimensions: {info['height']}x{info['width']} • Channels: {info['channels']} • DPI: {info['dpi']} • Format: {info['format']} • File Size: {info['file_size_bytes']} bytes")
'''

app_py = app_py.replace("__DATE__", datetime.date.today().isoformat())

with open(os.path.join(base_dir, "app.py"), "w", encoding="utf-8") as f:
    f.write(app_py)

# Notebook
nb = {
 "cells": [
  {"cell_type":"markdown","metadata":{},"source":[
    "# Image Processing & Analysis Toolkit — Notebook\n",
    "\n",
    "**Roll No:** ROLL123  \n",
    "**Module:** Image Processing Fundamentals & Computer Vision\n",
    "\n",
    "This notebook walks through core operations used in the Streamlit GUI, with explanations and runnable code."
  ]},
  {"cell_type":"markdown","metadata":{},"source":[
    "## Setup\n",
    "Install (in terminal):\n",
    "```\n",
    "pip install opencv-python numpy matplotlib pillow\n",
    "```\n"
  ]},
  {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":[
    "import cv2, numpy as np\n",
    "from matplotlib import pyplot as plt\n",
    "from PIL import Image\n",
    "print(cv2.__version__)"
  ]},
  {"cell_type":"markdown","metadata":{},"source":["## Load an image (update the path)"]},
  {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":[
    "path = 'your_image.jpg'  # change this\n",
    "img_bgr = cv2.imread(path)\n",
    "img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)\n",
    "plt.imshow(img_rgb); plt.axis('off')"
  ]},
  {"cell_type":"markdown","metadata":{},"source":[
    "## Color Conversions (RGB/HSV/YCbCr/Gray)\n",
    "- **HSV** is good for color-based segmentation.\n",
    "- **YCbCr** separates luminance (Y) from chroma (Cb/Cr) for compression and enhancement.\n",
    "- **Grayscale** reduces to intensity."
  ]},
  {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":[
    "hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)\n",
    "ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)\n",
    "gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)\n",
    "plt.figure(); plt.imshow(gray, cmap='gray'); plt.title('Gray'); plt.axis('off')"
  ]},
  {"cell_type":"markdown","metadata":{},"source":[
    "## Geometric Transformations\n",
    "- Rotation, Scaling, Translation, Affine, Perspective."
  ]},
  {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":[
    "h, w = img_bgr.shape[:2]\n",
    "M = cv2.getRotationMatrix2D((w/2, h/2), 30, 1.0)\n",
    "rot = cv2.warpAffine(img_bgr, M, (w, h))\n",
    "plt.figure(); plt.imshow(cv2.cvtColor(rot, cv2.COLOR_BGR2RGB)); plt.title('Rotated'); plt.axis('off')"
  ]},
  {"cell_type":"markdown","metadata":{},"source":[
    "## Filtering & Morphology\n",
    "- Gaussian/Median/Mean smoothing\n",
    "- Sobel/Laplacian edges\n",
    "- Dilation/Erosion/Opening/Closing"
  ]},
  {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":[
    "blur = cv2.GaussianBlur(img_bgr, (5,5), 0)\n",
    "edges = cv2.Canny(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), 100, 200)\n",
    "plt.figure(); plt.imshow(edges, cmap='gray'); plt.title('Canny'); plt.axis('off')"
  ]},
  {"cell_type":"markdown","metadata":{},"source":[
    "## Enhancement\n",
    "- Histogram Equalization on luminance (Y channel)\n",
    "- Unsharp masking for sharpening\n"
  ]},
  {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":[
    "ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)\n",
    "y, cr, cb = cv2.split(ycrcb)\n",
    "y_eq = cv2.equalizeHist(y)\n",
    "eq = cv2.cvtColor(cv2.merge([y_eq, cr, cb]), cv2.COLOR_YCrCb2BGR)\n",
    "plt.figure(); plt.imshow(cv2.cvtColor(eq, cv2.COLOR_BGR2RGB)); plt.title('Equalized (Y)'); plt.axis('off')"
  ]},
  {"cell_type":"markdown","metadata":{},"source":[
    "## Compression\n",
    "Use `cv2.imencode` to estimate size without writing to disk."
  ]},
  {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":[
    "ok, enc_jpg = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])\n",
    "ok, enc_png = cv2.imencode('.png', img_bgr)\n",
    "print('JPG bytes:', len(enc_jpg.tobytes()))\n",
    "print('PNG bytes:', len(enc_png.tobytes()))"
  ]},
  {"cell_type":"markdown","metadata":{},"source":[
    "## Practice Tasks\n",
    "1. Implement contrast stretching using percentile clipping (1%–99%).\n",
    "2. Build a custom 3×3 sharpening kernel and compare vs unsharp masking.\n",
    "3. Try perspective transform to deskew a document image.\n",
    "4. Compare compression quality vs size for JPEG qualities 50–100."
  ]}
 ],
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.x"}
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

nb_name = "ImageToolkit_ROLL123.ipynb"
with open(os.path.join(base_dir, nb_name), "w", encoding="utf-8") as f:
    json.dump(nb, f)

# Report PDF template
pdf_path = os.path.join(base_dir, "Report_Template.pdf")
c = canvas.Canvas(pdf_path, pagesize=A4)
width, height = A4
margin = 2*cm
y = height - margin
c.setFont("Helvetica-Bold", 16)
c.drawString(margin, y, "Image Processing & Analysis Toolkit — Report")
y -= 24
c.setFont("Helvetica", 11)
lines = [
    "Name: ______________________    Roll No: ____________    Date: ____________",
    "Module: Image Processing Fundamentals & Computer Vision",
    "",
    "1) CMOS vs CCD (notes): _________________________________________________",
    "__________________________________________________________________________",
    "2) Sampling & Quantization: ______________________________________________",
    "__________________________________________________________________________",
    "3) Point Spread Functions (PSFs): ________________________________________",
    "__________________________________________________________________________",
    "",
    "Screenshots/Results: (attach and annotate):",
    " - Color conversions",
    " - Transformations",
    " - Filtering & Morphology",
    " - Enhancement & Edge Detection",
    " - Compression (sizes/quality)",
    "",
    "Algorithm Explanations:",
    " - Describe equations, kernels, and parameters used.",
    " - Discuss trade-offs and observations.",
    "",
    "Conclusion & Future Work: ________________________________________________",
    "__________________________________________________________________________",
    "",
    "Submission Checklist: app.py, notebook (.ipynb), report (.pdf)",
]
for txt in lines:
    c.drawString(margin, y, txt); y -= 14
c.showPage(); c.save()


import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.title("🖼️ Image Processing & Analysis Toolkit")

# File uploader
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    # Show original
    st.image(image, caption="Original Image", use_container_width=True)

    # Example: Grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    st.image(gray, caption="Grayscale Image", use_container_width=True, channels="GRAY")

    # Collect info
    info = {
        "height": img_array.shape[0],
        "width": img_array.shape[1],
        "channels": img_array.shape[2] if len(img_array.shape) == 3 else 1,
        "dpi": image.info.get("dpi", (72, 72))[0],
        "format": image.format,
        "file_size_bytes": uploaded_file.size
    }

    # ✅ Status bar (only shows when image exists)
    st.text(
        f"Status • Dimensions: {info['height']}x{info['width']} • "
        f"Channels: {info['channels']} • DPI: {info['dpi']} • "
        f"Format: {info['format']} • File Size: {info['file_size_bytes']} bytes"
    )
