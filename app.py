import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from timeit import default_timer as timer

st.set_page_config(
    page_title="FoodVision",
    page_icon="🍕",
    layout="centered"
)

device = "cuda" if torch.cuda.is_available() else "cpu"

class_names = ["pizza", "steak", "sushi"]

model = model.to(device)
checkpoint = torch.load(
    "models/food101_model.pth",
    map_location=device
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# -----------------------------
# Image Transform
# -----------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Use the SAME normalization used during training
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Prediction Function
# -----------------------------

def predict(img):
    """Predict the food class and return probabilities and prediction time."""

    start_time = timer()

    # Transform image
    img = transform(img).unsqueeze(0).to(device)

    # Prediction
    with torch.inference_mode():
        pred_probs = torch.softmax(model(img), dim=1)

    # Create probability dictionary
    pred_labels_and_probs = {
        class_names[i]: float(pred_probs[0][i])
        for i in range(len(class_names))
    }

    pred_time = round(timer() - start_time, 5)

    return pred_labels_and_probs, pred_time


# -----------------------------
# Streamlit Interface
# -----------------------------

st.title("🍕 FoodVision")

st.write(
    "An EfficientNetB4 computer vision model "
    "that classifies food images as pizza, steak, or sushi."
)

st.divider()

uploaded_file = st.file_uploader(
    "Upload a food image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Predict", type="primary"):

        predictions, pred_time = predict(image)

        # Find highest probability class
        predicted_class = max(
            predictions,
            key=lambda class_name: predictions[class_name]
        )

        confidence = predictions[predicted_class]

        st.subheader("Prediction")

        st.success(
            f"{predicted_class.upper()} "
            f"({confidence * 100:.2f}% confidence)"
        )

        st.subheader("Class Probabilities")

        for class_name, probability in predictions.items():

            st.write(
                f"**{class_name.capitalize()}**: "
                f"{probability * 100:.2f}%"
            )

            st.progress(probability)

        st.caption(
            f"Prediction time: {pred_time:.5f} seconds"
        )
