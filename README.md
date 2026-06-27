# Heart Disease Prediction ❤️

I built a machine learning project that predicts whether a patient is at risk of
heart disease, and wrapped it in an interactive web app using Streamlit. This
README walks through everything I did — from exploring the raw data all the way
to deploying a working prediction tool.

## The Dataset

I worked with a dataset of **918 patient records** and **11 medical features**
such as age, sex, chest pain type, resting blood pressure, cholesterol, maximum
heart rate, and ST slope. The target column, `HeartDisease`, tells whether each
patient was diagnosed with heart disease (1) or not (0).

## Step 1 — Exploring the Data (EDA)

I started by getting to know the data before touching any model:

- I checked the **shape, data types, and summary statistics** to understand what
  I was working with.
- I confirmed there were **no missing values and no duplicate rows**.
- I plotted **histograms** for the numeric features (Age, RestingBP, Cholesterol,
  MaxHR, Oldpeak) and checked their **skewness** to see how they were distributed.
- I drew **count plots** for the categorical features to see how the categories
  were balanced.
- I compared features against the target using **count plots** (for example
  ChestPainType vs HeartDisease) to spot which features looked predictive.
- I built a **correlation heatmap** to see how the numeric features related to
  each other.
- I used **box plots** to look for outliers in the numeric columns.

## Step 2 — Cleaning the Data

While exploring, I noticed two problems and fixed them:

- **Cholesterol had zeros**, which is medically impossible. I replaced every `0`
  with the **mean of the non-zero cholesterol values** so the placeholder zeros
  didn't distort the model.
- **RestingBP had an impossible low value**, so I replaced values of 60 or below
  with the **mean of the realistic readings**.

## Step 3 — Preparing the Features

- I **one-hot encoded** the categorical columns using `pd.get_dummies` with
  `drop_first=True` to avoid redundant columns.
- I **converted the resulting boolean columns to integers** (0/1) so the models
  could use them.
- I split the data into **features (X) and target (y)**, then into **80% training
  and 20% testing** sets.
- I **scaled the features** with `StandardScaler`, since distance-based models
  like KNN and SVM are sensitive to feature scale.

## Step 4 — Training and Tuning Each Model

I trained five different classifiers and tuned each one individually instead of
just accepting the defaults:

- **Logistic Regression** — I tested the regularization strength `C` across
  `[0.01, 0.1, 1, 10]` and selected **C = 0.1**, which gave the best accuracy.
- **K-Nearest Neighbors** — I looped through neighbor counts
  `[3, 5, 7, 9, 11, 15, 21]` and chose **k = 9** as the best value.
- **Decision Tree** — I tested `max_depth` of `[3, 4, 5, 6, 8, 10, None]` with
  `min_samples_leaf = 10`, and picked **max_depth = 6**. This stopped the tree
  from overfitting, which had been dragging its accuracy down.
- **Support Vector Machine** — I tried combinations of `C` `[0.1, 1, 10]` and
  kernels `['rbf', 'linear']`, and settled on **C = 1 with the RBF kernel**.
- **Naive Bayes** — I used `GaussianNB`, which has no real hyperparameters to
  tune, as a strong baseline.

## Step 5 — Evaluating the Models

After tuning, I evaluated every model on the test set using **accuracy** and
**F1 score**:

| Model               | Accuracy | F1 Score |
|---------------------|----------|----------|
| Logistic Regression | 86.96%   | 0.8846   |
| KNN                 | 86.96%   | 0.8846   |
| SVM                 | 85.87%   | 0.8762   |
| Naive Bayes         | 84.78%   | 0.8614   |
| Decision Tree       | 84.24%   | 0.8612   |

To make sure these numbers were trustworthy and not just luck from one split, I
ran **5-fold cross-validation** on each model:

| Model               | Cross-Validated Accuracy |
|---------------------|--------------------------|
| Logistic Regression | 85.83% (± 0.019)         |
| KNN                 | 85.69% (± 0.020)         |
| Naive Bayes         | 85.15% (± 0.020)         |
| SVM                 | 85.15% (± 0.017)         |
| Decision Tree       | 82.43% (± 0.036)         |

## Step 6 — Confusion Matrix

Because this is a medical problem, I went beyond accuracy and plotted a
**confusion matrix for each model**. I paid special attention to **false
negatives** — patients who actually have heart disease but get predicted as
healthy — since that is the most dangerous mistake the model could make.

## Step 7 — Choosing the Final Model

I selected **Logistic Regression** as my final model. It had the highest
cross-validated accuracy, a low variance between folds (meaning it's stable), and
it's simple and interpretable — a good fit for this dataset.

## Step 8 — Saving and Deploying

- I **saved the trained model, the scaler, and the column order** with `joblib`,
  so the app could reuse the exact same preprocessing as training.
- I **built an interactive Streamlit app** (`app.py`) where anyone can enter a
  patient's details and instantly get a risk prediction with a probability score.

## Project Structure

```
├── Heart_Disease.ipynb   # Full workflow: EDA, cleaning, training & tuning
├── heart.csv             # Dataset
├── app.py                # Streamlit web app
├── heart_model.pkl       # Saved trained model
├── scalar.pkl            # Saved StandardScaler
├── columns.pkl           # Saved training column order
├── requirements.txt      # Python dependencies
└── README.md
```

## How to Run It Locally

1. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch the app:
   ```bash
   streamlit run app.py
   ```
3. Open the link shown in the terminal (usually http://localhost:8501).

## Tech Stack

Python · pandas · NumPy · Matplotlib · Seaborn · scikit-learn · Streamlit · joblib

## Disclaimer

This is an educational project I built to practice the full machine learning
workflow. It is **not** a medical device and should never be used for real
diagnosis or treatment decisions.
