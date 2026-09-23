import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import torchvision
from PIL import Image
from timeit import default_timer as timer


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FoodVision 101",
    page_icon="🍕",
    layout="wide",
    initial_sidebar_state="collapsed",
)

device = "cuda" if torch.cuda.is_available() else "cpu"

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --ink: #17211b;
        --muted: #68736b;
        --line: #dfe6df;
        --paper: #f5f7f2;
        --lime: #d4f36b;
        --coral: #ff765f;
    }

    .stApp {
        background: var(--paper);
        color: var(--ink);
        font-family: 'Manrope', sans-serif;
    }

    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { visibility: hidden; }
    .block-container { max-width: 1180px; padding: 3rem 3rem 4rem; }

    .hero {
        background: var(--ink);
        border-radius: 22px;
        padding: 2.8rem 3rem 2.6rem;
        position: relative;
        overflow: hidden;
        color: #f7faf4;
        margin-bottom: 1.15rem;
    }
    .hero::after {
        content: '101';
        position: absolute;
        right: 2rem;
        bottom: -3rem;
        color: rgba(212, 243, 107, 0.13);
        font-size: 13rem;
        font-weight: 800;
        line-height: 1;
        letter-spacing: -0.08em;
    }
    .eyebrow, .mono { font-family: 'DM Mono', monospace; letter-spacing: .08em; text-transform: uppercase; }
    .eyebrow { color: var(--lime); font-size: .72rem; margin-bottom: 1rem; }
    .hero h1 { font-size: clamp(2.7rem, 6vw, 5.4rem); line-height: .95; letter-spacing: -.07em; margin: 0; max-width: 650px; position: relative; z-index: 1; }
    .hero-copy { color: #b9c5bb; font-size: 1rem; line-height: 1.6; max-width: 470px; margin: 1.25rem 0 0; position: relative; z-index: 1; }

    .status-row { display: flex; gap: .65rem; flex-wrap: wrap; margin: .8rem 0 2rem; }
    .status { border: 1px solid var(--line); border-radius: 999px; padding: .48rem .8rem; color: var(--muted); background: rgba(255,255,255,.62); font-size: .78rem; }
    .status strong { color: var(--ink); }
    .status-dot { color: #4d9d67; font-size: .95rem; vertical-align: -1px; }

    [data-testid="stFileUploader"] { border: 1px dashed #aebcaf; border-radius: 16px; background: #fbfcf9; padding: .8rem; }
    [data-testid="stFileUploaderDropzone"] { background: transparent; border: 0; }
    [data-testid="stFileUploaderDropzoneInstructions"] p { font-size: .9rem; color: var(--muted); }
    [data-testid="stFileUploader"] button { background: var(--lime); color: var(--ink); border: 0; border-radius: 10px; font-weight: 700; }
    [data-testid="stFileUploader"] button:hover { background: #c1df58; color: var(--ink); border: 0; }
    .section-label { font-family: 'DM Mono', monospace; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); font-size: .72rem; margin-bottom: .7rem; }
    .upload-title { font-size: 1.75rem; font-weight: 800; letter-spacing: -.04em; margin: 0 0 .35rem; }
    .upload-copy { color: var(--muted); margin: 0 0 1.2rem; }
    .empty-state { min-height: 270px; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; border: 1px solid var(--line); border-radius: 16px; background: rgba(255,255,255,.4); color: var(--muted); padding: 2rem; }
    .empty-icon { font-size: 2.4rem; margin-bottom: .7rem; }

    .result-header { display: flex; align-items: end; justify-content: space-between; gap: 1rem; border-bottom: 1px solid var(--line); padding-bottom: .9rem; margin-bottom: 1.1rem; }
    .result-title { font-size: 1.9rem; font-weight: 800; letter-spacing: -.05em; margin: 0; }
    .winner { background: var(--lime); border-radius: 16px; padding: 1.25rem 1.35rem; margin-bottom: 1rem; }
    .winner-label { font-family: 'DM Mono', monospace; text-transform: uppercase; font-size: .68rem; letter-spacing: .08em; color: #52652d; }
    .winner-name { font-size: 1.75rem; font-weight: 800; letter-spacing: -.05em; margin: .25rem 0 .1rem; }
    .winner-meta { color: #52652d; font-size: .82rem; }
    .rank-row { display: flex; align-items: center; gap: .75rem; padding: .62rem 0; border-bottom: 1px solid var(--line); }
    .rank-number { color: var(--muted); font-family: 'DM Mono', monospace; font-size: .75rem; width: 1.5rem; }
    .rank-name { flex: 1; font-weight: 600; }
    .rank-score { color: var(--muted); font-family: 'DM Mono', monospace; font-size: .75rem; }
    .footnote { color: var(--muted); font-size: .75rem; margin-top: 1.6rem; }

    .stButton > button { background: var(--ink); color: white; border: 0; border-radius: 10px; min-height: 3rem; font-weight: 700; }
    .stButton > button:hover { background: #2b3a2f; color: var(--lime); border: 0; }
    @media (max-width: 700px) {
        .block-container { padding: 1.2rem 1rem 3rem; }
        .hero { padding: 2rem 1.4rem; border-radius: 16px; }
        .hero h1 { font-size: 3.3rem; }
        .hero::after { right: -.5rem; font-size: 9rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================
from effnet_model import create_model

model,transform = create_model(model=torchvision.models.efficientnet_b4,weights=torchvision.models.EfficientNet_B4_Weights.DEFAULT,num_classes=101,seed=42)
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    101
)
model = model.to(device)

checkpoint = torch.load(
    "models/food101_model.pth",
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

from effnet_model import class_names
from predict import predict


# ============================================================
# STREAMLIT UI
# ============================================================

st.markdown(
    """
    <section class="hero">
        <div class="eyebrow">Computer vision / food recognition</div>
        <h1>What’s on the plate?</h1>
        <p class="hero-copy">Drop in a food photo and let FoodVision identify it across 101 familiar dishes.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"<div class='status-row'><div class='status'><span class='status-dot'>●</span> Model <strong>ready</strong></div><div class='status'>Architecture <strong>EfficientNet-B4</strong></div><div class='status'>Runtime <strong>{device.upper()}</strong></div><div class='status'>Classes <strong>101</strong></div></div>",
    unsafe_allow_html=True,
)

upload_column, result_column = st.columns([0.92, 1.08], gap="large")
predict_clicked = False
image = None

with upload_column:
    st.markdown("<div class='section-label'>01 / Add an image</div><h2 class='upload-title'>Show us your meal.</h2><p class='upload-copy'>Use a clear JPG, JPEG, or PNG for the best read.</p>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload a food image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        st.markdown("<div class='empty-state'><div class='empty-icon'>⌁</div><strong>Your preview will appear here</strong><span>One image at a time · up to 200 MB</span></div>", unsafe_allow_html=True)
    else:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption=uploaded_file.name, use_container_width=True)
        predict_clicked = st.button("Identify this dish  →", type="primary", use_container_width=True)

with result_column:
    st.markdown("<div class='section-label'>02 / Results</div>", unsafe_allow_html=True)

    if uploaded_file is None:
        st.markdown("<div class='empty-state'><div class='empty-icon'>✦</div><strong>Waiting for a plate</strong><span>Your top five predictions will land here.</span></div>", unsafe_allow_html=True)
    elif predict_clicked and image is not None:
        predictions, prediction_time = predict(
            image=image,
            model=model,
            transform=transform,
            device=device,
            class_names=class_names,
        )

        predicted_class, confidence = next(iter(predictions.items()))
        st.markdown("<div class='result-header'><h2 class='result-title'>Best match</h2><span class='mono'>Top 05</span></div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='winner'><div class='winner-label'>Most likely dish</div><div class='winner-name'>{predicted_class.replace('_', ' ').title()}</div><div class='winner-meta'>{confidence * 100:.2f}% confidence · analyzed in {prediction_time:.3f}s</div></div>",
            unsafe_allow_html=True,
        )

        for rank, (class_name, probability) in enumerate(predictions.items(), start=1):
            st.markdown(
                f"<div class='rank-row'><span class='rank-number'>0{rank}</span><span class='rank-name'>{class_name.replace('_', ' ').title()}</span><span class='rank-score'>{probability * 100:.1f}%</span></div>",
                unsafe_allow_html=True,
            )
        st.markdown("<div class='footnote'>Predictions are generated by a fine-tuned Food-101 classifier.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='empty-state'><div class='empty-icon'>↗</div><strong>Ready when you are</strong><span>Press “Identify this dish” to run the model.</span></div>", unsafe_allow_html=True)