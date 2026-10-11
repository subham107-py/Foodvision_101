# FoodVision

FoodVision is a Streamlit interface for running and comparing Food-101 image classifiers.

## Run locally

Install the dependencies and start the app from the project directory:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The app includes the bundled ConvNeXt Tiny and Base checkpoints. To register another model, add its checkpoint to the repository and add an entry to [`models/model_catalog.json`](./models/model_catalog.json). This is a manual, Git-based workflow; there is no model administration or upload interface in the app.

Each model entry has an `id`, display `name`, `description`, `architecture`, checkpoint path relative to the project root, and `classes`. Supported architecture names are `ConvNeXt Tiny` and `ConvNeXt Base`. Set `classes` to `null` to use the Food-101 labels, or specify an array of class names in the same order as the model's output scores. Checkpoints need a compatible ConvNeXt classifier head with at least as many outputs as labels and are loaded as weights-only files. For example:

```json
{
  "id": "my-food-model",
  "name": "My Food Model",
  "description": "A brief description shown when selecting this model.",
  "architecture": "ConvNeXt Tiny",
  "checkpoint": "models/my_food_model.pth",
  "classes": null
}
```

Append the entry to the JSON array and commit both the configuration and checkpoint. GitHub has file-size limits for large checkpoints; use Git LFS or a suitable model-storage service when needed. Streamlit's light theme and app colors are configured in `.streamlit/config.toml`.
