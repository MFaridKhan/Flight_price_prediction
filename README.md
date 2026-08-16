# ✈️ Flight Price Prediction

An end-to-end machine learning project that predicts flight ticket prices based on airline, route, timing, and stopover details — from raw data through EDA, model comparison, hyperparameter tuning, experiment tracking, and a deployed interactive web app.

## 📌 Overview

Flight prices are notoriously volatile — they shift with airline, route, time of day, number of stops, and how far in advance you book. This project builds a regression model that estimates a flight's fare from its booking details, then serves that model through a live, styled Streamlit interface.

The project follows a complete ML lifecycle:

```
Raw Data → EDA & Feature Engineering → Model Comparison → Hyperparameter Tuning
        → Experiment Tracking (MLflow) → Model Export → Streamlit App
```

---

## 🖼️ Demo

<img width="596" height="413" alt="screenshot" src="https://github.com/user-attachments/assets/5b8898a6-c3d8-42fe-b836-51465bbfd5cd" />


---

## 🧠 Problem Statement

Given flight details — airline, source, destination, departure/arrival time, and number of stops — predict the **ticket price** (in ₹). This is framed as a **supervised regression problem**.

---

## 🔍 Exploratory Data Analysis & Feature Engineering

Performed in `1_EDA.ipynb` on the [Flight Price Prediction dataset](https://www.kaggle.com/datasets/nikhilmittal/flight-fare-prediction-mh) (~10,700 train records, 2,671 test records):

- Parsed `Date_of_Journey` into `journey_day` / `journey_month`
- Split `Dep_Time` / `Arrival_Time` into hour/minute components
- Parsed the free-text `Duration` column (e.g. `"2h 50m"`, `"19h"`, `"5m"`) into numeric `duration_hour` / `duration_mins`, handling irregular formats
- Converted `Total_Stops` (e.g. *"non-stop"*, *"1 stop"*) into a numeric feature
- One-hot encoded categorical columns: `Airline`, `Source`, `Destination`
- Dropped low-value / high-cardinality columns (`Route`, `Additional_Info`)
- Used `ExtraTreesRegressor` to rank feature importance and validate the engineered feature set

---

## 🤖 Model Training & Comparison

Ten regression algorithms were trained and evaluated on an 80/20 train-test split in `2_Model_Training.ipynb`: Linear Regression, Decision Tree, Random Forest, Gradient Boosting, AdaBoost, KNN, SVR, **XGBoost**, and **LightGBM**.

### Baseline results (default hyperparameters)

| Model | Train R² | Test R² | RMSE | MAE |
|---|---|---|---|---|
| **XGBoost** | 0.937 | **0.846** | 1822 | 1127 |
| LightGBM | 0.873 | 0.826 | 1938 | 1243 |
| Random Forest | 0.953 | 0.798 | 2089 | 1176 |
| Gradient Boosting | 0.783 | 0.786 | 2150 | 1528 |
| Decision Tree | 0.969 | 0.725 | 2437 | 1336 |
| Linear Regression | 0.624 | 0.620 | 2864 | 1973 |
| KNN | 0.736 | 0.576 | 3025 | 1874 |
| AdaBoost | 0.524 | 0.520 | 3217 | 2460 |
| SVR | 0.003 | -0.000 | 4644 | 3566 |

**XGBoost** produced the strongest out-of-the-box test R² and was selected as the final model.

### Hyperparameter tuning

The top 6 candidates were tuned with `RandomizedSearchCV` (5-fold cross-validation, 20 iterations) across n_estimators, max_depth, learning_rate, subsample, and other model-specific parameters:

| Model | CV R² | Test R² | RMSE | MAE |
|---|---|---|---|---|
| Gradient Boosting | 0.821 | 0.838 | 1870 | 1220 |
| **XGBoost** | 0.831 | 0.832 | 1903 | 1149 |
| LightGBM | 0.816 | 0.826 | 1935 | 1177 |
| Random Forest | 0.794 | 0.817 | 1985 | 1304 |
| Decision Tree | 0.784 | 0.779 | 2181 | 1234 |
| AdaBoost | 0.544 | 0.523 | 3206 | 2250 |

**XGBoost** was kept as the production model — it combined the best cross-validated stability (0.831 CV R²) with the lowest MAE (₹1,149) among top performers, and required no further preprocessing changes to serve.

### Model interpretability

`shap.TreeExplainer` was used to explain predictions and confirm the model relies on sensible signals — `Duration`, `Total_Stops`, and `Airline` rank among the top drivers of predicted price.

---

## 🧪 Experiment Tracking

All training runs — parameters, metrics, and model artifacts — are logged with **MLflow**, run locally:

```bash
mlflow ui --port 5000
```

Then open **http://127.0.0.1:5000** to compare runs across all 9 models by R², RMSE, and MAE, and inspect each run's best hyperparameters.

---

## 🚀 Running the App

### 1. Clone the repository

```bash
git clone https://github.com/MFaridKhan/flight-price-prediction.git
cd flight-price-prediction
```

### 2. Set up the environment

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Launch the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 🖥️ App Features

- Clean, responsive two-column form — departure details on the left, arrival details on the right
- Live prediction powered by the trained XGBoost model (`model.ubj`)
- Custom-styled UI with a gradient result card showing the estimated fare
- Input fields: Airline, Source, Destination, Departure date/time, Arrival date/time, Total Stops

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Data & EDA | `pandas`, `numpy`, `seaborn`, `matplotlib` |
| Modeling | `scikit-learn`, `xgboost`, `lightgbm` |
| Experiment Tracking | `mlflow` |
| Model Explainability | `shap` |
| App / Serving | `streamlit` |

---

## 📈 Final Model Performance

| Metric | Score |
|---|---|
| Test R² | 0.83 – 0.85 |
| RMSE | ~₹1,820 – 1,900 |
| MAE | ~₹1,130 – 1,150 |

---

## 🔮 Future Improvements

- [ ] Replace manual one-hot encoding in `app.py` with a saved `OneHotEncoder`/`feature_columns.pkl` pipeline for more robust handling of unseen categories
- [ ] Fix overnight-flight duration calculation to use full datetime differences instead of raw hour subtraction
- [ ] Register the production model in the MLflow Model Registry with staged promotion (Staging → Production)
- [ ] Version datasets with DVC via DagsHub for full reproducibility
- [ ] Containerize with Docker and deploy to Streamlit Community Cloud / Render
- [ ] Add automated tests for the preprocessing pipeline

---


---

## 🙋 Author

Built as part of an end-to-end ML deployment portfolio project — covering data cleaning, model selection, experiment tracking, and deployment.
