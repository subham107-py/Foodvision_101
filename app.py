import json
import html
import pickle
from pathlib import Path
from typing import Any

import streamlit as st
import torch
import torch.nn as nn
import torchvision
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parent
CATALOG_PATH = ROOT_DIR / "models" / "model_catalog.json"
CLASSES_PATH = ROOT_DIR / "data" / "food-101" / "meta" / "classes.txt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ARCHITECTURES: dict[str, Any] = {
    "EfficientNet-B4": torchvision.models.efficientnet_b4,
    "Effnetb4": torchvision.models.efficientnet_b4,
    "ConvNeXt Tiny": torchvision.models.convnext_tiny,
    "ConvNeXt Base": torchvision.models.convnext_base,
}
WEIGHTS = {
    "EfficientNet-B4": torchvision.models.EfficientNet_B4_Weights.DEFAULT,
    "Effnetb4": torchvision.models.EfficientNet_B4_Weights.DEFAULT,
    "ConvNeXt Tiny": torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT,
    "ConvNeXt Base": torchvision.models.ConvNeXt_Base_Weights.DEFAULT,
}

st.set_page_config(
    page_title="FoodVision | Model Studio",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def read_food_classes() -> list[str]:
    if not CLASSES_PATH.is_file():
        raise FileNotFoundError(f"Food-101 class labels were not found: {CLASSES_PATH}")
    return [line.strip() for line in CLASSES_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_catalog() -> list[dict[str, Any]]:
    if not CATALOG_PATH.is_file():
        raise FileNotFoundError(f"Model configuration was not found: {CATALOG_PATH}")

    with CATALOG_PATH.open(encoding="utf-8") as catalog_file:
        catalog = json.load(catalog_file)
    if not isinstance(catalog, list) or not all(isinstance(model, dict) for model in catalog):
        raise ValueError(f"Model catalog must contain a list of models: {CATALOG_PATH}")
    for model in catalog:
        required_fields = ("id", "name", "description", "architecture", "checkpoint")
        if any(not isinstance(model.get(field), str) or not model[field].strip() for field in required_fields):
            raise ValueError(f"Every model in {CATALOG_PATH} must define {', '.join(required_fields)}.")
    return catalog


@st.cache_resource(show_spinner="Loading model…")
def load_model(
    architecture: str,
    checkpoint_path: str,
    class_count: int,
    checkpoint_modified_ns: int,
) -> tuple[nn.Module, Any]:
    del checkpoint_modified_ns
    if architecture not in ARCHITECTURES or architecture not in WEIGHTS:
        raise ValueError(f"Unsupported model architecture: {architecture}")

    model = ARCHITECTURES[architecture](weights=None)
    classifier = model.classifier[-1]
    if not isinstance(classifier, nn.Linear):
        raise ValueError(f"Unexpected classifier structure for {architecture}")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint must contain a model state dictionary.")

    classifier_name = next(
        name for name, module in model.named_modules() if module is model.classifier[-1]
    )
    classifier_weights = state_dict.get(f"{classifier_name}.weight")
    if not isinstance(classifier_weights, torch.Tensor) or classifier_weights.ndim != 2:
        raise ValueError("Checkpoint is missing a compatible classifier weight matrix.")
    checkpoint_class_count = classifier_weights.shape[0]
    if checkpoint_class_count < class_count:
        raise ValueError(
            f"Checkpoint supports {checkpoint_class_count} classes, but {class_count} labels are configured."
        )
    model.classifier[-1] = nn.Linear(classifier.in_features, checkpoint_class_count)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    transform = WEIGHTS[architecture].transforms()
    return model, transform


def resolve_classes(model_entry: dict[str, Any]) -> list[str]:
    classes = model_entry.get("classes")
    if classes is None:
        return read_food_classes()
    if not isinstance(classes, list) or not classes or not all(
        isinstance(class_name, str) and class_name.strip() for class_name in classes
    ):
        raise ValueError(f"Invalid class labels configured for {model_entry['name']}.")
    return [class_name.strip() for class_name in classes]


def classify_image(image: Image.Image, model: nn.Module, transform: Any, classes: list[str], top_k: int):
    input_tensor = transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        logits = model(input_tensor)[0]
        probabilities = torch.softmax(logits, dim=0)
    if probabilities.numel() < len(classes):
        raise ValueError(
            f"Model returned {probabilities.numel()} scores but {len(classes)} class labels are configured."
        )
    probabilities = probabilities[: len(classes)]
    scores, indices = torch.topk(probabilities, k=min(top_k, len(classes)))
    return [(classes[int(index.item())], float(score.item())) for score, index in zip(scores, indices)]


def render_page_styles() -> None:
    st.markdown(
        """
        <style>
        :root { color-scheme: light; }
        .stApp { background: #f5f7f2; color: #18231e; }
        [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
        [data-testid="stHeader"], #MainMenu, footer { display: none !important; }
        .block-container { max-width: 835px; padding: 0 0 3rem; }
        .hero {
            position: relative; overflow: hidden; min-height: 266px; box-sizing: border-box;
            padding: 18px 38px 28px; margin: 0 0 22px; border-radius: 0 0 17px 17px;
            background: #14211c; color: #f6f8f4;
        }
        .hero-eyebrow { position: relative; z-index: 1; color: #c4ef4d; font: 500 10px/1.5 ui-monospace, SFMono-Regular, Consolas, monospace; letter-spacing: .08em; text-transform: uppercase; }
        .hero-title { position: relative; z-index: 1; max-width: 440px; margin: 26px 0 8px; color: #f6f8f4; font-size: clamp(42px, 6vw, 62px); font-weight: 760; letter-spacing: -.065em; line-height: .99; }
        .hero-copy { position: relative; z-index: 1; max-width: 400px; margin: 0; color: #c1cac3; font-size: 13px; line-height: 1.6; }
        .hero-number { position: absolute; right: 22px; bottom: -47px; color: #24362a; font-size: clamp(130px, 21vw, 190px); font-weight: 850; letter-spacing: -.09em; line-height: 1; user-select: none; }
        .status-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 0 0 26px; }
        .status-chip { display: inline-flex; align-items: center; gap: 5px; min-height: 29px; padding: 0 11px; border: 1px solid #e0e7de; border-radius: 999px; background: #fbfcf9; color: #718079; font-size: 10px; }
        .status-chip strong { color: #27352d; font-weight: 600; }
        .status-dot { width: 6px; height: 6px; border-radius: 50%; background: #63a57c; }
        .section-label { margin: 0 0 10px; color: #66756c; font: 500 10px/1.4 ui-monospace, SFMono-Regular, Consolas, monospace; letter-spacing: .035em; }
        .section-title { margin: 0 0 10px; color: #19251e; font-size: 25px; font-weight: 580; letter-spacing: -.035em; line-height: 1.2; }
        .section-copy { margin: 0 0 16px; color: #718078; font-size: 12px; line-height: 1.5; }
        [data-testid="stSelectbox"] { margin-bottom: 7px; }
        [data-testid="stSelectbox"] > label { display: none; }
        [data-testid="stSelectbox"] [data-baseweb="select"] > div { min-height: 36px; border-color: #dfe6dd; border-radius: 10px; background: #fbfcf9; color: #24342b; font-size: 12px; }
        [data-testid="stFileUploader"] { margin: 2px 0 12px; }
        [data-testid="stFileUploader"] > label { display: none; }
        [data-testid="stFileUploader"] section {
            min-height: 72px; padding: 11px 16px; border: 1px dashed #c3d0c0;
            border-radius: 13px; background: #fbfcf9; color: #38483e;
        }
        [data-testid="stFileUploader"] section > div { align-items: center; }
        [data-testid="stFileUploader"] section button {
            min-height: 32px; border: 0; border-radius: 9px; background: #c7ef59;
            color: #20301f; font-size: 11px; font-weight: 600;
        }
        [data-testid="stFileUploader"] section small { color: #748078; }
        [data-testid="stImage"] img { max-height: 205px; border-radius: 12px; object-fit: contain; }
        [data-testid="stVerticalBlockBorderWrapper"] {
            min-height: 208px; border-color: #e0e7de !important; border-radius: 14px !important;
            background: #fafbf8;
        }
        .empty-card {
            display: flex; min-height: 174px; box-sizing: border-box; flex-direction: column;
            align-items: center; justify-content: center; text-align: center;
            color: #718078; font-size: 12px; line-height: 1.6;
        }
        .empty-sparkle { margin-bottom: 12px; color: #748679; font-size: 24px; }
        .empty-title { color: #43534a; font-size: 12px; font-weight: 600; }
        .empty-copy { color: #859189; }
        .prediction-card {
            padding: 18px 18px 13px; border: 1px solid #e0e7de;
            border-radius: 14px; background: #fafbf8;
        }
        .prediction-heading { margin-bottom: 14px; color: #24342b; font-size: 13px; font-weight: 650; }
        .prediction-row { margin: 12px 0; }
        .prediction-meta { display: flex; justify-content: space-between; gap: 10px; margin-bottom: 6px; color: #3c4c42; font-size: 11px; }
        .prediction-score { color: #66766c; font-variant-numeric: tabular-nums; }
        .prediction-track { height: 5px; overflow: hidden; border-radius: 99px; background: #e7ece4; }
        .prediction-fill { height: 100%; border-radius: inherit; background: #80aa57; }
        @media (max-width: 680px) {
            .block-container { padding: 0 16px 2rem; }
            .hero { min-height: 250px; margin: 0 -16px 20px; padding: 17px 22px 25px; }
            .hero-title { max-width: 330px; font-size: 48px; }
            .hero-number { right: 8px; font-size: 130px; }
            .section-title { font-size: 22px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


render_page_styles()
try:
    models = load_catalog()
except (OSError, json.JSONDecodeError, ValueError) as error:
    st.error(f"Could not read the model configuration: {error}")
    st.stop()

available_models = [
    entry
    for entry in models
    if (ROOT_DIR / entry.get("checkpoint", "")).is_file()
    and entry.get("architecture") in ARCHITECTURES
]
if not available_models:
    st.error("No model checkpoints are configured or available. Add a model to models/model_catalog.json.")
    st.stop()

model_names = {entry["name"]: entry for entry in available_models}
st.markdown(
    """
    <section class="hero">
        <div class="hero-eyebrow">Computer vision&nbsp; / &nbsp;Food recognition</div>
        <h1 class="hero-title">What’s on the<br>plate?</h1>
        <p class="hero-copy">Drop in a food photo and let FoodVision identify it across 101 familiar dishes.</p>
        <div class="hero-number" aria-hidden="true">101</div>
    </section>
    """,
    unsafe_allow_html=True,
)

model_choice = st.selectbox("Choose a model", list(model_names))
selected_model = model_names[model_choice]
st.caption(selected_model.get("description", ""))
try:
    selected_classes = resolve_classes(selected_model)
except (OSError, ValueError, KeyError) as error:
    st.error(f"Could not read class labels for this model: {error}")
    st.stop()

st.markdown(
    f"""
    <div class="status-row">
        <span class="status-chip"><span class="status-dot"></span>Model <strong>ready</strong></span>
        <span class="status-chip">Architecture <strong>{html.escape(selected_model["architecture"])}</strong></span>
        <span class="status-chip">Runtime <strong>{DEVICE.type.upper()}</strong></span>
        <span class="status-chip">Classes <strong>{len(selected_classes)}</strong></span>
    </div>
    """,
    unsafe_allow_html=True,
)

left_column, right_column = st.columns([1, 1.18], gap="large")
image: Image.Image | None = None
with left_column:
    st.markdown('<div class="section-label">01&nbsp; / &nbsp;ADD AN IMAGE</div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Show us your meal.</h2>', unsafe_allow_html=True)
    st.markdown('<p class="section-copy">Use a clear JPG, JPEG, or PNG for the best read.</p>', unsafe_allow_html=True)
    image_upload = st.file_uploader(
        "Drop a food photo here",
        type=["jpg", "jpeg", "png"],
        help="Upload one food image at a time (up to 200 MB).",
    )
    st.markdown('<div class="section-label">IMAGE PREVIEW</div>', unsafe_allow_html=True)
    if image_upload is None:
        st.markdown(
            """
            <div class="empty-card" style="min-height:206px;border:1px solid #e0e7de;border-radius:14px;background:#fafbf8;">
                <div class="empty-sparkle">⌁</div>
                <div class="empty-title">Your preview will appear here</div>
                <div class="empty-copy">One image at a time · up to 200 MB</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        try:
            image = Image.open(image_upload).convert("RGB")
            st.image(image, use_container_width=True)
        except (OSError, ValueError) as error:
            st.error(f"Could not open the uploaded image: {error}")
            image_upload = None

with right_column:
    st.markdown('<div class="section-label">02&nbsp; / &nbsp;RESULTS</div>', unsafe_allow_html=True)
    if image is None:
        st.markdown(
            """
            <div class="empty-card" style="min-height:208px;border:1px solid #e0e7de;border-radius:14px;background:#fafbf8;">
                <div class="empty-sparkle">✦</div>
                <div class="empty-title">Waiting for a plate</div>
                <div class="empty-copy">Your top five predictions will land here.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        try:
            checkpoint_path = ROOT_DIR / selected_model["checkpoint"]
            model, transform = load_model(
                selected_model["architecture"],
                str(checkpoint_path),
                len(selected_classes),
                checkpoint_path.stat().st_mtime_ns,
            )
            predictions = classify_image(image, model, transform, selected_classes, 5)
        except (OSError, RuntimeError, ValueError, KeyError, pickle.UnpicklingError) as error:
            st.error(f"Could not run this model: {error}")
        else:
            rows = []
            for rank, (label, score) in enumerate(predictions, start=1):
                safe_label = html.escape(label.replace("_", " ").title())
                rows.append(
                    f"""
                    <div class="prediction-row">
                        <div class="prediction-meta">
                            <span>{rank:02d}&nbsp;&nbsp; {safe_label}</span>
                            <span class="prediction-score">{score:.1%}</span>
                        </div>
                        <div class="prediction-track"><div class="prediction-fill" style="width:{score:.1%}"></div></div>
                    </div>
                    """
                )
            st.html(
                '<div class="prediction-card"><div class="prediction-heading">Top five predictions</div>'
                + "".join(rows)
                + "</div>",
            )
