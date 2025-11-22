# Urban Traffic, Weather, and Road Safety

> **Tagline:** Data-driven insights on how weather and calendar patterns impact road accident risk and severity.

This repository operationalizes an end-to-end data science pipeline to quantify and predict how **weather conditions** and **calendar effects** (weekends, holidays, seasons) influence **traffic accidents** across the United States.

The project is designed as a full-scope DSA210 deliverable: from raw data ingestion and cleaning, through exploratory data analysis (EDA) and hypothesis testing, to machine learning models for **risk classification** and **accident count regression**.

---

## 1. Business Question

> **How do weather and calendar patterns impact the likelihood and severity of road accidents, and can we predict high-risk conditions in advance?**

From this core question, we drive two main analytics workstreams:

1. **Analytics & Hypothesis Testing**  
   - Do adverse weather conditions (rain, snow, fog) significantly increase daily accident counts?  
   - Are accidents on holidays/weekends more severe than on regular weekdays?

2. **Predictive Modeling**  
   - Can we classify **high-risk days** (days with at least one severe accident) based on weather + calendar features?  
   - Can we forecast **daily accident counts** using only exogenous signals (weather, calendar, demographics)?

---

## 2. Project Scope & Objectives

**Strategic Objectives:**

- Quantify the impact of **weather** and **calendar events** on accident risk.
- Build **interpretable ML models** that highlight key drivers (e.g., precipitation, visibility, holidays).
- Deliver reproducible **notebooks, plots, and a written report** that comply with DSA210 expectations.

**Key Deliverables:**

- Cleaned and integrated dataset combining **accidents + weather + holiday + (optional) demographics**.
- EDA notebook with descriptive statistics, visualizations, and correlation analysis.
- Hypothesis testing notebook with properly stated H₀/H₁, test selection, and interpretation.
- ML modeling notebook(s) for:
  - **Classification:** High-risk day prediction.
  - **Regression:** Daily accident count prediction.
- Final report (PDF / notebook section) summarizing methodology, results, limitations, and future work.

---

## 3. Data Assets

> All raw data is obtained from public/open sources (e.g., Kaggle, official open data portals) and then stored locally under `data/`.

### 3.1. Dataset 1 – US Traffic Accidents

- **Name:** US Accidents (2016–2023) – Countrywide Traffic Accident Dataset  
- **Source:** Kaggle (`US Accidents (2016 - 2023)`)  
- **Content:**  
  - Per-incident records including:
    - `Start_Time`, `End_Time`, `City`, `State`, `Severity`
    - Location (lat/lon), road attributes (e.g., `Traffic_Signal`, `Junction`)
    - Additional contextual descriptors  
- **Usage:** Core source for **accident count** and **severity** metrics.

### 3.2. Dataset 2 – Weather Data (US)

- **Name:** Daily Weather Dataset (US)  
- **Source:** Kaggle / public weather datasets  
- **Content:**
  - Daily metrics: `Temperature`, `Precipitation`, `Snow`, `Visibility`, `WindSpeed`, etc.  
- **Usage:** Enables mapping weather conditions to accident dates and locations.

### 3.3. Dataset 3 – US Holidays (Calendar Enrichment)

- **Name:** US Holiday Dates  
- **Source:** Kaggle or official holiday calendars  
- **Content:**  
  - `Date`, `Holiday_Name`, `Is_Holiday`  
- **Usage:** Encodes **holiday effects** and supports `is_holiday`, `pre_holiday`, `post_holiday` flags.

### 3.4. Dataset 4 – Demographics / Urban Context (Optional)

- **Name:** US City / Metro Demographics  
- **Source:** U.S. Census / Kaggle  
- **Content:**  
  - City-level `Population`, `Density`, `MedianIncome`, etc.  
- **Usage:** Enables normalized metrics (`AccidentsPer100k`) and richer feature sets.

---

## 4. Data Pipeline

The end-to-end data preparation flow is as follows:

1. **Raw Ingestion**
   - Load `us_accidents.csv`, `weather_us_daily.csv`, `us_holidays.csv`, and optional `demographics.csv` into `pandas`.

2. **Data Cleaning**
   - Convert timestamps (`Start_Time`, `End_Time`) to `datetime`.
   - Derive:
     - `Year`, `Month`, `Day`, `Hour`, `DayOfWeek`
   - Drop records with:
     - Missing `Severity` or critical location fields.
     - Clearly corrupted coordinates or unusable timestamps.
   - Filter to a manageable scope (e.g., top N cities by accident count).

3. **Aggregation**
   - Aggregate accidents to **city-day** level:
     - `AccidentCount` = number of accidents per city per day.
     - `AverageSeverity` = mean severity per city per day.
   - Aggregate weather to **city-day** via nearest station or mapping.

4. **Integration**
   - Merge:
     - `Accidents` ⨝ `Weather` on (`City`, `Date`)
     - ⨝ `Holidays` on `Date`
     - ⨝ `Demographics` on (`City`, optional)

5. **Feature Engineering**
   - Time features: `is_weekend`, `is_holiday`, `Season`.
   - Weather flags: `is_rainy`, `is_snowy`, `is_foggy`, `low_visibility`.
   - Normalization: `AccidentsPer100k = AccidentCount / Population * 100000`.
   - Lagged variables:
     - `AccidentCount_lag1`
     - `AccidentCount_7d_ma`
   - Classification target example:
     - `HighSeverityFlag = 1` if any severity ≥ 3 accident occurs that day in that city; else 0.

---

## 5. Exploratory Data Analysis (EDA)

All EDA is captured in **`notebooks/01_eda.ipynb`** and includes:

- Summary statistics via `.describe()` and `.skew()` for:
  - `AccidentCount`, `AccidentsPer100k`, `AverageSeverity`
  - Weather metrics (precipitation, snow, visibility, etc.)
- Time-series plots:
  - Accident counts over months and years.
  - Seasonal patterns by `Month`, `Season`.
- Segmentation analysis:
  - Clear vs rainy/snowy days.
  - Weekday vs weekend vs holiday.
- Correlation matrices & heatmaps:
  - Accident metrics vs weather, calendar, and demographic features.

**Expected patterns:** higher accident counts around **rush hours**, increased risk in **adverse weather**, and specific behaviors on **holidays/weekends**.

---

## 6. Hypothesis Testing

Formal statistical tests are implemented in **`notebooks/02_hypothesis_testing.ipynb`**.

### Example Hypotheses

1. **Weather & Accident Count**
   - **H₀:** Mean daily accident count is the same on clear and rainy days.  
   - **H₁:** Mean daily accident count differs between clear and rainy days.  
   - **Method:** Two-sample t-test or Mann-Whitney U (depending on distribution).

2. **Holidays & Severity**
   - **H₀:** Average accident severity on holidays equals that of non-holidays.  
   - **H₁:** Average accident severity on holidays differs from non-holidays.  
   - **Method:** t-test or ANOVA across (weekday, weekend, holiday) groups.

3. **Precipitation & Accident Count Correlation**
   - **H₀:** No linear relationship between daily precipitation and accident count.  
   - **H₁:** Significant correlation exists.  
   - **Method:** Pearson and/or Spearman correlation.

Each hypothesis section includes:

- Clear statement of H₀ / H₁  
- Rationale for test selection  
- Test statistic, p-value, and conclusion at α = 0.05

---

## 7. Machine Learning Models

ML experiments are implemented in **`notebooks/03_ml_models.ipynb`**.

### 7.1. Classification – High-Risk Day Prediction

**Goal:** Flag **city-days** where at least one high-severity accident (severity ≥ 3) is expected.

- **Target:** `HighSeverityFlag` (0/1)
- **Features:**
  - Calendar: `Month`, `DayOfWeek`, `is_weekend`, `is_holiday`, `Season`
  - Weather: `Temp`, `Precip`, `Snow`, `WindSpeed`, `Visibility`, weather flags
  - Lagged: `AccidentCount_lag1`, `AccidentCount_7d_ma`
  - Demographics (optional): `Population`, `Density`
- **Preprocessing:**
  - Handle missing values.
  - One-hot encode categorical features.
  - Address class imbalance using **SMOTE** or similar oversampling.
- **Model:**
  - **RandomForestClassifier** (baseline, focus on interpretability and robustness)
- **Evaluation:**
  - Train/test split with stratification.
  - Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC.
  - Confusion matrix and feature importance plot.

### 7.2. Regression – Daily Accident Count Forecasting

**Goal:** Predict **daily accident counts** (or `AccidentsPer100k`) for each city using exogenous drivers.

- **Target:** `AccidentCount` (or normalized variant).
- **Features:** Same as above, minus any future accident information (no leakage).
- **Models:**
  - **LinearRegression** baseline.
  - Optional: `RandomForestRegressor` or Ridge/Lasso for comparison.
- **Metrics:**
  - **R²**, **MAE**, **MSE** / **RMSE**
- **Visuals:**
  - Actual vs predicted scatter plot.
  - Residual plots for pattern checks.

---

## 8. Limitations & Future Work

**Current constraints:**

- Aggregation to city-day level may mask local (intersection-level) heterogeneity.
- Accident reporting may be incomplete for minor incidents.
- No direct traffic volume metrics (vehicle-kilometers traveled) included at this stage.
- Causal interpretation is out of scope; we focus on **associations**, not strict causality.

**Future enhancements:**

- Integrate traffic volume / mobility data to normalize risk further.
- Move to more granular spatial resolution (road segments, corridors).
- Explore basic causal inference frameworks (e.g., weather shocks as natural experiments).
- Prototype a simple risk-scoring API / dashboard fed by weather forecasts and holiday calendars.

---

## 9. Tech Stack

- **Language:** Python
- **Core Libraries:**
  - `pandas`, `numpy`
  - `matplotlib`, `seaborn`
  - `scikit-learn`
  - `imbalanced-learn`
  - `scipy`
- **Tooling:**
  - Jupyter Notebook
  - VSCode (optional)
  - Git / GitHub for version control

---

## 10. Repository Structure

Proposed structure:

```text
.
├── data/
│   ├── raw/
│   │   ├── us_accidents.csv
│   │   ├── weather_us_daily.csv
│   │   ├── us_holidays.csv
│   │   └── demographics.csv        # optional
│   └── processed/
│       └── city_day_merged.parquet # or .csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_hypothesis_testing.ipynb
│   └── 03_ml_models.ipynb
├── src/
│   ├── __init__.py
│   ├── data_prep.py
│   ├── features.py
│   └── models.py
├── reports/
│   └── final_report.pdf            
├── requirements.txt
└── README.md
