import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter


DATASET_ROOT = "./Brain_Cancer"
OUTPUT_DIR   = "./preprocessed"
IMG_SIZE     = (224, 224)
TEST_SIZE    = 0.2
RANDOM_SEED  = 42

CLASS_MAP = {
    "brain_glioma": "glioma",
    "brain_menin":  "meningioma",
    "brain_tumor":  "pituitary",
}


def crop_brain_region(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    pad = 5
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(img.shape[1] - x, w + 2 * pad)
    h = min(img.shape[0] - y, h + 2 * pad)
    return img[y:y+h, x:x+w]


def preprocess_image(img_path: str) -> np.ndarray:

    img = cv2.imread(img_path)
    if img is None:
        return None
    img = crop_brain_region(img)
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img


def load_all_images():

    images, labels = [], []
    root = Path(DATASET_ROOT)

    for folder, label in CLASS_MAP.items():
        cls_path = root / folder
        if not cls_path.exists():
            print(f"  error: 0 pics {cls_path}")
            continue

        img_files = [f for f in cls_path.glob("*")
                     if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}]
        print(f"  {folder:20s} -> '{label}': {len(img_files)} images")

        for img_path in tqdm(img_files, desc=f"  {label}", leave=False):
            img = preprocess_image(str(img_path))
            if img is not None:
                images.append(img)
                labels.append(label)

    X = np.array(images, dtype=np.float32)
    le = LabelEncoder()
    y = le.fit_transform(labels)

    print(f"\n  generall: {X.shape[0]} pics | Shape: {X.shape}")
    print(f"  Label-Mapping: { {cls: int(i) for i, cls in enumerate(le.classes_)} }")
    return X, y, labels


def split_and_save(X, y, labels):

    print(f"\n-- Train/Test-Split (80/20) ----------------------------")

    X_train, X_test, y_train, y_test, l_train, l_test = train_test_split(
        X, y, labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y
    )

    print(f"  Training : {X_train.shape[0]} images")
    print(f"  Testing  : {X_test.shape[0]} images")

    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    for name, arr in [("train_X", X_train), ("train_y", y_train),
                      ("test_X",  X_test),  ("test_y",  y_test)]:
        path = out / f"{name}.npy"
        np.save(path, arr)
        mb = arr.nbytes / 1e6
        size_str = f"({mb:.1f} MB)" if mb > 0.1 else ""
        print(f"  saved: {path}  {size_str}")

    return X_train, X_test, y_train, y_test, l_train, l_test


def plot_class_distribution(l_train, l_test):

    classes = sorted(set(l_train))
    train_counts = [Counter(l_train)[c] for c in classes]
    test_counts  = [Counter(l_test)[c]  for c in classes]

    x = np.arange(len(classes))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width/2, train_counts, width, label="Training (80%)", color="#185FA5")
    ax.bar(x + width/2, test_counts,  width, label="Testing (20%)",  color="#1D9E75")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_ylabel("Number of Images")
    ax.set_title("Class Distribution After Split")
    ax.legend()
    plt.tight_layout()
    plt.savefig(Path(OUTPUT_DIR) / "classdistribution.png", dpi=120)
    plt.show()
    print("  saved: preprocessed/classdistribution.png")


def plot_samples(X, labels, n=6, title="examples"):
   
    idx = np.random.choice(len(X), min(n, len(X)), replace=False)
    fig, axes = plt.subplots(1, n, figsize=(14, 3))
    for i, ax in enumerate(axes):
        ax.imshow(X[idx[i]])
        ax.set_title(labels[idx[i]], fontsize=10)
        ax.axis("off")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(Path(OUTPUT_DIR) / "examples.png", dpi=120)
    plt.show()
    print("  saved: preprocessed/examples.png")


def print_stats(X, name):
    print(f"\n  {name} Statistiken:")
    print(f"    Shape  : {X.shape}")
    print(f"    Min    : {X.min():.4f}  |  Max: {X.max():.4f}")
    print(f"    Mean   : {X.mean():.4f}  |  Std: {X.std():.4f}")


# ─────────────────────────────────────────────
# Main ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Brain Tumor MRI - preprocessing & normalization")
    print("=" * 55)
    print(f"  Dataset  : {DATASET_ROOT}")
    print(f"  output  : {OUTPUT_DIR}")
    print(f"  Imagesize: {IMG_SIZE[0]}x{IMG_SIZE[1]} px")
    print(f"  Class  : {list(CLASS_MAP.values())}")

    X, y, labels = load_all_images()
    print_stats(X, "generally")

    X_train, X_test, y_train, y_test, l_train, l_test = split_and_save(X, y, labels)

    print("\n-- Visualisation------------------------------------")
    plot_class_distribution(l_train, l_test)
    plot_samples(X_train, l_train, n=6, title="training examples (normalised)")

    print_stats(X_train, "Training")
    print_stats(X_test,  "Testing")

    print("\n" + "=" * 55)
    print("  Finish! File in:", OUTPUT_DIR)
    print("=" * 55)