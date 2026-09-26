# Parkinson's Disease Prediction — Learning Project

> **Educational use only.** This repository demonstrates a small machine-learning
> workflow using voice measurements. It is **not** a medical device, a diagnostic
> test, or a substitute for a qualified clinician's assessment.

## What you will learn

This project is a beginner-friendly, end-to-end binary-classification example.
By working through the notebook, you will practice how to:

1. load tabular data with pandas;
2. inspect its shape, types, summary statistics, missing values, and class balance;
3. separate predictors (features) from the outcome (target);
4. create reproducible train/test splits;
5. standardize numeric data without leaking information from the test set;
6. train a linear support-vector-machine (SVM) classifier;
7. evaluate predictions with accuracy; and
8. make a prediction for one new, correctly ordered set of measurements.

The complete walkthrough is in
[`parkison_disease_detection.ipynb`](parkison_disease_detection.ipynb). The
filename intentionally uses the repository's existing spelling, `parkison`.

## Project layout

```text
.
├── parkison_disease_detection.ipynb  # Exploratory analysis, training, and prediction
├── sample_data/
│   └── parkinsons.csv                # Voice-measurement dataset used by the notebook
└── README.md                         # This guide
```

## Dataset at a glance

The included CSV contains **195 rows** and **24 columns**. Each row contains an
identifier, 22 numeric voice measurements, and a binary `status` label:

| Label | Meaning used in this project | Rows |
| --- | --- | ---: |
| `0` | Parkinson's negative | 48 |
| `1` | Parkinson's positive | 147 |

The class distribution is uneven (about 75% positive), which is an important
lesson: accuracy by itself can be misleading on imbalanced data. Later in this
guide, you will see ways to extend the evaluation.

### Columns

- `name` is a recording identifier. The notebook excludes it from training
  because it is an identifier, not a clinical measurement.
- `status` is the target to predict.
- The remaining 22 fields are the numeric features. The names are retained from
  the data file so that their ordering is unambiguous:

| Feature group | Columns |
| --- | --- |
| Fundamental-frequency measures | `MDVP:Fo(Hz)`, `MDVP:Fhi(Hz)`, `MDVP:Flo(Hz)` |
| Jitter / frequency variation | `MDVP:Jitter(%)`, `MDVP:Jitter(Abs)`, `MDVP:RAP`, `MDVP:PPQ`, `Jitter:DDP` |
| Shimmer / amplitude variation | `MDVP:Shimmer`, `MDVP:Shimmer(dB)`, `Shimmer:APQ3`, `Shimmer:APQ5`, `MDVP:APQ`, `Shimmer:DDA` |
| Noise and harmonic measures | `NHR`, `HNR` |
| Nonlinear dynamical measures | `RPDE`, `DFA`, `spread1`, `spread2`, `D2`, `PPE` |

Before relying on any field clinically, consult the dataset documentation and
domain experts. This repository uses the fields only to teach a modeling
workflow.

## Quick start

### 1. Prerequisites

Install Python 3.9 or newer and `pip`. Create and activate a virtual
environment (recommended):

```bash
python -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

### 2. Install the notebook dependencies

```bash
python -m pip install --upgrade pip
python -m pip install numpy pandas scikit-learn jupyter
```

### 3. Start Jupyter from the repository root

```bash
jupyter notebook
```

Open `parkison_disease_detection.ipynb`, then choose **Kernel → Restart Kernel
and Run All Cells**. Starting Jupyter at the repository root matters because
the notebook loads the data from `./sample_data/parkinsons.csv`.

## Notebook walkthrough

### 1. Import the tools

The first code cell imports NumPy and pandas for data handling, plus selected
scikit-learn tools:

- `train_test_split` creates holdout data for evaluation;
- `StandardScaler` puts each feature on a comparable scale; and
- `svm.SVC` creates the SVM classifier.

### 2. Explore and validate the data

The next cells use `head()`, `tail()`, `shape`, `info()`, `isnull().sum()`, and
`describe()` to understand the table. Reproduce this compact check in a new
cell:

```python
import pandas as pd

data = pd.read_csv("sample_data/parkinsons.csv")
print(data.shape)                 # expected: (195, 24)
print(data.isnull().sum().sum())  # expected: 0
print(data["status"].value_counts())
```

This inspection is not busywork. It catches common errors early: a wrong file
path, non-numeric values in numeric columns, unexpected missing values, or a
target with an unexpected class balance.

### 3. Define features and target

```python
X = data.drop(columns=["name", "status"])
y = data["status"]
```

`X` has 22 columns of measurements and `y` holds the known label. Do not put
`status` in `X`: doing so would reveal the answer to the model (target leakage).
Likewise, including `name` can make a model memorize recording-specific
patterns instead of learning generalizable relationships.

### 4. Split before scaling

The notebook reserves 20% of the records for testing:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=20
)
```

With this data, that produces 156 training rows and 39 test rows. The fixed
`random_state` makes the split repeatable. A useful improvement for this
imbalanced target is `stratify=y`, which aims to preserve the class proportion
in both subsets.

### 5. Scale correctly

Voice features have very different numeric ranges. A linear SVM is sensitive to
those ranges, so the notebook fits `StandardScaler` **only** on training data,
then applies the learned mean and standard deviation to both datasets:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Fitting the scaler on all rows before splitting is data leakage: the test set
would influence the transformation used to train the model. The notebook avoids
that mistake.

### 6. Train and evaluate the SVM

```python
from sklearn import svm
from sklearn.metrics import accuracy_score

model = svm.SVC(kernel="linear")
model.fit(X_train_scaled, y_train)

print("Training accuracy:", accuracy_score(y_train, model.predict(X_train_scaled)))
print("Test accuracy:", accuracy_score(y_test, model.predict(X_test_scaled)))
```

The notebook's saved run reports approximately **0.929** training accuracy and
**0.872** test accuracy. These values are specific to the included data, the
split seed, and library behavior; rerun the notebook instead of treating them
as a clinical-performance claim. The gap between training and test performance
is also a prompt to investigate generalization rather than to declare success.

### 7. Predict one example

The final cells turn a 22-value tuple into a two-dimensional array, scale it
with the *same fitted scaler*, and call `model.predict`. The order must exactly
match `X.columns`; otherwise the model receives the wrong measurement in each
position. A safer version preserves feature names:

```python
import pandas as pd

new_measurement = pd.DataFrame([input_data], columns=X.columns)
prediction = model.predict(scaler.transform(new_measurement))
print("Predicted status:", int(prediction[0]))
```

Interpret a result only as the model's class label (`0` or `1`) for this
learning dataset—not as a diagnosis.

## Suggested learner exercises

Try these one at a time and record what changes:

1. **Use a stratified split.** Add `stratify=y` to `train_test_split`, compare
   the two class distributions, and rerun the score.
2. **Go beyond accuracy.** Print a confusion matrix and classification report:

   ```python
   from sklearn.metrics import ConfusionMatrixDisplay, classification_report

   print(classification_report(y_test, model.predict(X_test_scaled)))
   ConfusionMatrixDisplay.from_predictions(y_test, model.predict(X_test_scaled))
   ```

3. **Use a pipeline.** Combine scaling and classification to make leakage
   harder to introduce:

   ```python
   from sklearn.pipeline import make_pipeline

   pipeline = make_pipeline(StandardScaler(), svm.SVC(kernel="linear"))
   pipeline.fit(X_train, y_train)
   print(pipeline.score(X_test, y_test))
   ```

4. **Try validation strategies.** Use `StratifiedKFold` and cross-validation
   rather than drawing conclusions from a single 39-row test split.
5. **Compare models thoughtfully.** Test a logistic-regression baseline or an
   SVM with a different kernel, tuning settings only within a cross-validation
   workflow.
6. **Investigate the unit of observation.** Determine whether multiple rows
   belong to the same person/recording group before choosing a split strategy;
   random row-wise splits can overstate performance when groups overlap.

## Important limitations and responsible use

- The dataset is small, so score estimates have substantial uncertainty.
- The labels are imbalanced; a high accuracy can still hide poor results for one
  class. Inspect precision, recall, and the confusion matrix.
- The notebook evaluates one fixed holdout split, not external validation on a
  new population.
- A voice-based classifier can be affected by recording conditions, population
  differences, and data-collection choices not represented here.
- This project has no clinical validation, threshold calibration, fairness
  analysis, deployment safeguards, or regulatory review.

Use it to learn data science concepts, not to make health decisions about any
person.

## Troubleshooting

| Problem | Likely cause and fix |
| --- | --- |
| `FileNotFoundError` for the CSV | Launch Jupyter from the repository root, or update the path to the CSV. |
| `ModuleNotFoundError` | Activate the virtual environment and rerun the `pip install` command above. |
| A warning about feature names during prediction | Create the new sample as a DataFrame with `columns=X.columns`, as shown above. |
| Different scores from the saved output | Confirm the data file, `random_state=20`, split settings, dependency versions, and that cells were run in order. |

## Next steps

After completing the notebook, consider turning the experiment into a more
reproducible project: pin dependency versions, add automated tests for data
schema assumptions, save a pipeline with model metadata, and document a
validated evaluation protocol. Those practices matter as much as the model
itself.
