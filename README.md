# FoodVision 101    Model Url:[🔗](https://foodvision101byrai.streamlit.app/)

FoodVision 101 is a Streamlit computer-vision app that identifies food from an uploaded image. It uses a fine-tuned EfficientNet-B4 model trained for the 101 classes in the Food-101 dataset and returns the five most likely dishes with confidence scores.

## Features

- Upload JPG, JPEG, or PNG food images
- Predict across all 101 Food-101 categories
- Display the top five predictions and confidence scores
- Run inference on CUDA when a compatible GPU is available, otherwise use CPU
- Use ImageNet-compatible preprocessing from the EfficientNet-B4 weights
- Show inference time for each prediction

## Demo flow

1. Start the Streamlit app.
2. Upload a clear food image.
3. Select **Identify this dish**.
4. Review the best match and the next four predictions.

## Tech stack

- Python
- Streamlit
- PyTorch
- Torchvision
- Pillow
- EfficientNet-B4
- Food-101 dataset

## Project structure

```text
Foodvision/
├── app.py                    # Streamlit user interface and inference flow
├── effnet_model.py           # Model factory, transforms, and Food-101 classes
├── predict.py                # Top-five prediction helper
├── food101.ipynb             # Notebook used for model experimentation/training
├── models/
│   ├── app.py                # Additional model-related app file
│   └── food101_model.pth     # Fine-tuned model checkpoint
├── data/
│   └── food-101/             # Food-101 images and metadata
└── Sample_Images/            # Sample images for testing
```

## Requirements

- Python 3.10 or newer is recommended
- A CPU can run the app, but inference will be slower
- A CUDA-enabled PyTorch installation is recommended for GPU inference
- Internet access may be required on the first run to download missing Torchvision assets or dataset files

## Installation

Create and activate a virtual environment from the repository root:

### Windows PowerShell

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install streamlit torch torchvision pillow
```

For GPU support, install the PyTorch and Torchvision builds that match your CUDA version using the official PyTorch installation selector before launching the app.

## Run the app

Run this command from the repository root so the relative model and dataset paths resolve correctly:

```bash
streamlit run app.py
```

Streamlit will print a local URL, usually:

```text
http://localhost:8501
```

Open that URL in a browser, upload a food image, and run the prediction.

## Model details

The application creates an EfficientNet-B4 backbone with ImageNet weights, replaces its classifier with a 101-class output layer, and loads the fine-tuned weights from:

```text
models/food101_model.pth
```

The model is switched to evaluation mode before inference. Input images are converted to RGB and processed with the transforms associated with the EfficientNet-B4 pretrained weights.

The returned result is a dictionary containing the five highest-probability Food-101 classes, ordered from most likely to least likely.

## Supported classes

The classifier recognizes the 101 Food-101 categories, including examples such as:

- Apple pie
- Bibimbap
- Caesar salad
- Cheesecake
- French fries
- Hamburger
- Pizza
- Sushi
- Tacos
- Waffles

The complete class list is read from the Food-101 dataset through Torchvision.

## Troubleshooting

### `ModuleNotFoundError`

Make sure the virtual environment is active and install the dependencies again:

```bash
python -m pip install streamlit torch torchvision pillow
```

### Model checkpoint not found

Start Streamlit from the repository root and confirm that this file exists:

```text
models/food101_model.pth
```

### Food-101 files are missing

The project expects the dataset under:

```text
data/food-101/
```

Torchvision may attempt to download missing Food-101 files when `effnet_model.py` is imported. Ensure the machine has internet access for the first setup.

### Predictions are slow

CPU inference is supported but slower. Install a CUDA-compatible PyTorch build and use a supported NVIDIA GPU to enable the CUDA execution path.

### PowerShell does not allow activation

Run PowerShell with an appropriate execution policy for your user account, then activate the environment again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

## Limitations

- Predictions are limited to the 101 Food-101 classes.
- Results depend on image quality, framing, lighting, and how closely the image matches the training data.
- The model should be treated as an image-classification demo, not as a nutritional, medical, or dietary advisory tool.

## License and dataset attribution

The Food-101 dataset has its own license and usage terms. Review `data/food-101/license_agreement.txt` and `data/food-101/README.txt` before redistributing the dataset or publishing derived assets.
