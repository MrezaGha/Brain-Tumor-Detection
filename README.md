# Brain Tumor Detection — Short README
This project benchmarks two CNN architectures — ResNet-50 and EfficientNet-B0 — for classifying brain tumor MRI scans into four categories: glioma, meningioma, pituitary tumor, and no tumor. Both models were fine-tuned using transfer learning on roughly 13,000 MRI images from Kaggle and trained on Google Colab with an NVIDIA Tesla T4 GPU.

EfficientNet-B0 came out on top, converging faster and generalizing better across all four classes, while ResNet-50 showed more signs of overfitting despite its depth. On top of the classification pipeline, we integrated an agentic AI component that takes the model's prediction and generates a natural language explanation — making the output actually useful in a clinical context, not just a label.

Dataset name:
- Brain Cancer - MRI dataset  (6057 Samples)
- Brain tumor MRI Dataset ( 7200 Samples)

Source of Datasets
- Brain Cancer - MRI dataset: https://www.kaggle.com/datasets/orvile/brain-cancer-mri-dataset
- Brain tumor MRI Dataset: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection

Modality:
- Image


Quick start
- Install dependencies:
  pip install -r requirements.txt
- Run API (local):
  uvicorn agentic_agent.main:app --reload --port 8000
- Open http://localhost:8000/ for the simple UI.

Notebooks & training
- Training & preprocessing notebooks: src/EfficientNetB0_base_model.ipynb, src/ResNet50_base_model.ipynb
- Notebooks expect data/checkpoints under Google Drive paths when run in Colab (see notebook top cells).

Model & checkpoints
- Default inference checkpoint: agentic_agent/app/models/efficientnet_best.pt
- Change the path in app.tools.mri_classifier or via env vars.

Tests
- Run tests:
  pytest -q

Notes
- Provide OPENAI_KEY in environment for audio/LLM features.