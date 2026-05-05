# For testing mri_classifier only
from app.tools.mri_classifier import predict_mri

result = predict_mri(
    image_path="app/scans/patient_001.jpg",
    checkpoint_path="app/models/efficientnet_best.pt",
)
print(result["class"], result["confidence"])