# Heart Disease Prediction System

**AIC354 — Machine Learning Fundamentals | Assignment 2 — Task 2**

| | |
|---|---|
| **Student Name** | Muhammad Asad Awan |
| **Student ID** | FA24-BSE-073 |
| **Section** | A |

An end-to-end ML system that predicts heart-disease presence from **exactly four inputs** using a Support Vector Classifier (SVC) in a scikit-learn pipeline, served by a Streamlit app.

🚀 **Live App:** https://heart-desease-predict-5recjjkc3vyyndhhbjfdd3.streamlit.app/  
📁 **GitHub:** https://github.com/masadawan99/heart-desease-predict

> **Educational use only.** Not a medical device, not clinically validated, never for diagnosis or treatment decisions.

## Selected predictors

| Column (Kaggle) | Description | Type | Preprocessing |
|---|---|---|---|
| `Age` | age in years | numeric | StandardScaler |
| `Sex` | `M` / `F` | categorical | OneHotEncoder |
| `ChestPainType` | `TA` typical angina, `ATA` atypical angina, `NAP` non-anginal, `ASY` asymptomatic | categorical | OneHotEncoder (`handle_unknown="ignore"`) |
| `MaxHR` | maximum heart rate achieved | numeric | StandardScaler |

**Target:** `HeartDisease` — 0 = no heart disease, 1 = heart disease present. (Not the same as the Cleveland `num` column, 0–4, which the baseline binarises as `num > 0`.)

## Datasets and attribution

- **Final model — Kaggle:** fedesoriano (2021). *Heart Failure Prediction Dataset.* https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction — 918 rows, 12 columns, licensed ODbL 1.0 (per Kaggle). It combines five heart-disease datasets (including Cleveland).
- **Baseline — UCI Cleveland:** https://archive.ics.uci.edu/dataset/45/heart+disease (`processed.cleveland.data`, kept unchanged; 303 rows, 297 without missing values).

### Download the Kaggle data
The file `data/kaggle_heart_failure/heart.csv` is included. To re-download:
```powershell
pip install kaggle        # needs a Kaggle API token in ~/.kaggle/kaggle.json (never commit it)
kaggle datasets download -d fedesoriano/heart-failure-prediction -p data/kaggle_heart_failure
```
Then unzip `heart.csv` into `data/kaggle_heart_failure/`. Or download manually from the dataset page ("Download") and place `heart.csv` there.

## Method

- 80/20 split, stratified on the target, `random_state=0`.
- `ColumnTransformer` (scaler + one-hot encoder) + `SVC(gamma="auto", random_state=0)`; preprocessing is fitted on the training split only.
- Metrics on the held-out test set: accuracy, precision, recall, F1, confusion matrix, ROC-AUC (from SVC decision scores, which are not calibrated probabilities).
- Shared code lives in `heart_pipeline.py`, used by both `train_model.py` and the notebook.
- The Cleveland baseline (original workflow) is reproduced in the notebook for comparison. The two datasets differ in size, encoding, target definition and population, so the comparison is not a controlled test of dataset size, and they must not be concatenated.

Results (held-out test set, 184 patients; see the notebook for the full output): accuracy 0.788, precision 0.794, recall 0.833, F1 0.813, ROC-AUC 0.859. Cleveland baseline (60 test patients): accuracy 0.600.

## Run

```powershell
pip install -r requirements.txt
python train_model.py              # retrain and write heart_disease_model.pkl
streamlit run app.py               # run the app locally
```
Notebook (extra packages): `pip install matplotlib seaborn ipykernel nbconvert`, then run `Heart_Disease_Prediction.ipynb` from this folder.

## Project structure

```
Heart_Disease_Prediction.ipynb   notebook (both experiments)
app.py                           Streamlit app (4 inputs)
train_model.py                   Kaggle training script
heart_pipeline.py                shared preprocessing / model / evaluation code
heart_disease_model.pkl          fitted pipeline + metadata (Kaggle model)
baseline_cleveland_model.pkl     previous Cleveland SVC, kept for reference
processed.cleveland.data         original UCI Cleveland data (baseline)
data/kaggle_heart_failure/heart.csv
heart.csv                        copy of the Kaggle CSV at root for submission
requirements.txt
```

## Deployment (Streamlit Community Cloud)

1. Push this folder to GitHub (no credentials are in the repo; `.gitignore` excludes `kaggle.json`).
2. At https://share.streamlit.io, create a new app pointing to `app.py` on the `main` branch.
3. Under *Advanced settings* choose Python 3.12 or newer, to match the pinned package versions in `requirements.txt`. The model file was saved with scikit-learn 1.9.1; if you change versions, rerun `python train_model.py`.

## Disclaimer

Educational project built for a university assignment (AIC354, FA24-BSE-073). Predictions are based on four attributes from a small public dataset and must not be used for clinical decisions.
