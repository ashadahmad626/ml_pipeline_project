# 💰 Income Prediction ML Pipeline

A **production-ready** Machine Learning pipeline that predicts whether an individual earns more or less than $50K per year, based on the Adult Census Income dataset.

---

## 🗂 Project Structure

```
ml_pipeline_project/
├── src/
│   ├── components/
│   │   ├── data_ingestion.py       # Downloads & splits data
│   │   ├── data_transformation.py  # Cleans, engineers features, encodes, scales
│   │   ├── model_trainer.py        # Trains 9 models + GridSearchCV
│   │   └── eda.py                  # Generates 10 EDA plots
│   ├── pipeline/
│   │   ├── training_pipeline.py    # Orchestrates full training
│   │   └── prediction_pipeline.py  # Serves predictions
│   ├── utils/
│   │   └── utils.py                # Save/load objects, evaluate_models
│   ├── logger.py                   # Centralized logging
│   └── exception.py                # Custom exception with traceback
├── artifacts/                      # Saved models, preprocessors, plots
├── logs/                           # Auto-generated log files
├── app_streamlit.py                # 🎨 Streamlit UI (main frontend)
├── requirements.txt
└── setup.py
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit app
```bash
streamlit run app_streamlit.py
```

### 3. (Optional) Train via CLI
```bash
python -m src.pipeline.training_pipeline
```

---

## 🤖 Models Trained
| Model | Tuned |
|---|---|
| Logistic Regression | ✅ |
| Decision Tree | ✅ |
| Random Forest | ✅ |
| Gradient Boosting | ✅ |
| AdaBoost | ✅ |
| Extra Trees | ✅ |
| XGBoost | ✅ |
| LightGBM | ✅ |
| KNN | ✅ |

Best model is selected automatically by **F1 Score** with **5-fold StratifiedKFold CV**.

---

## 🔬 Feature Engineering
- `net_capital` = capital_gain - capital_loss
- `work_intensity` = hours_per_week × age
- `high_education` = education_num ≥ 13
- `is_married` = Married-civ-spouse or Married-AF-spouse
- `has_capital` = any capital activity

---

## 📊 EDA Plots Generated
1. Target Distribution
2. Numerical Feature Distributions
3. Correlation Heatmap
4. Categorical Features vs Income
5. Age Analysis
6. Hours/Week Analysis
7. Capital Gain/Loss Analysis
8. Education Level Analysis
9. Missing Values
10. Occupation Income Rate

---

## 📋 Dataset
- **Source**: [UCI ML Repository — Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
- **Records**: ~48,842
- **Features**: 14 input features
- **Target**: Income >50K (binary)