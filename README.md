# 🛒 Product Price Prediction Pipeline

A multimodal machine learning pipeline that predicts product prices using **text embeddings**, **image embeddings**, and **numerical feature extraction**, trained with LightGBM and 5-fold cross-validation.

---

## 📋 Overview

This pipeline combines three types of features extracted from product catalog data:

| Feature Type | Source | Model/Method |
|---|---|---|
| Text Embeddings | `catalog_content` | `all-MiniLM-L6-v2` (SentenceTransformer) |
| Image Embeddings | `image_link` (local files) | `EfficientNet-B0` (timm) |
| Numerical Features | `catalog_content` (regex) | Weight, Count, Volume extraction |

These are concatenated and fed into a **LightGBM** regressor trained with **5-fold cross-validation** on log-transformed prices.

---

## 📁 Project Structure

```
project/
│
├── dataset/
│   ├── train.csv          # Training data (75,000 rows)
│   └── test.csv           # Test data (75,000 rows)
│
├── images/                # Pre-downloaded product images (named by filename from URL)
│
├── ML_Model.py               # Main training pipeline script
├── test_out.csv           # Generated submission file
└── README.md
```

---

## 📦 Requirements

Install all dependencies via pip:

```bash
pip install pandas numpy lightgbm scikit-learn torch torchvision \
            sentence-transformers timm Pillow tqdm
```

> **GPU support**: If a CUDA-compatible GPU is available, the pipeline will automatically use it for embedding generation.

---

## 🗂️ Dataset Format

### `train.csv`

| Column | Description |
|---|---|
| `sample_id` | Unique identifier |
| `catalog_content` | Product description text |
| `image_link` | URL to the product image |
| `price` | Target variable (product price) |

### `test.csv`

Same as `train.csv` but without the `price` column.

---

## 🖼️ Image Preparation

Images must be **pre-downloaded** into the `images/` folder before running the pipeline. Each image should be named after the **last segment of its URL**.

Example:
```
https://example.com/products/img_001.jpg  →  images/img_001.jpg
```

If an image is missing or corrupted, a **zero vector** is used as a fallback.

---

## 🚀 Running the Pipeline

```bash
python train.py
```

The script will:
1. Load `train.csv` and `test.csv`
2. Extract numerical features (weight, count, volume) via regex
3. Generate text embeddings using `all-MiniLM-L6-v2`
4. Generate image embeddings using `EfficientNet-B0`
5. Concatenate all features
6. Train a LightGBM model with 5-fold CV on log-transformed prices
7. Output predictions to `test_out.csv`

---

## ⚙️ Model Configuration

### LightGBM Parameters

| Parameter | Value |
|---|---|
| Objective | `regression_l1` (MAE) |
| Metric | `mae` |
| Estimators | 2000 (with early stopping at 100 rounds) |
| Learning Rate | 0.01 |
| Feature Fraction | 0.8 |
| Bagging Fraction | 0.8 |
| Num Leaves | 31 |
| Cross-Validation | 5-fold KFold |

### Feature Dimensions

| Feature | Dimensions |
|---|---|
| Text Embeddings | 384 |
| Image Embeddings | 1280 |
| Numerical Features | 3 |
| **Total** | **1667** |

---

## 📤 Output

The final submission file `test_out.csv` contains:

| Column | Description |
|---|---|
| `sample_id` | Test sample identifier |
| `price` | Predicted product price |

Predictions are clipped to a minimum of `0` (no negative prices).

---

## 🔍 Feature Engineering Details

### Numerical Feature Extraction (Regex)

The pipeline scans `catalog_content` for common product attributes:

- **Weight**: `oz`, `ounce`, `lb`, `pound`, `g`, `gram`, `kg`, `kilogram`
- **Count**: `count`, `pack`, `ct`
- **Volume**: `ml`, `milliliter`, `l`, `liter`, `fl oz`, `fluid ounce`

### Price Transformation

Prices are log-transformed (`log1p`) before training and back-transformed (`expm1`) at inference to handle skewed price distributions.

---

## 📝 Notes

- The pipeline uses **CPU by default** and switches to **GPU automatically** if CUDA is available.
- Missing images return zero vectors — consider imputing with mean embeddings for better performance.
- Increasing `n_estimators` or tuning `num_leaves` may further improve results.
