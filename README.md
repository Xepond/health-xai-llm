# LLM-Assisted Explainable AI for Health Risk Prediction

## 📖 Overview

The goal of this project is to predict patient health risks using machine learning models and provide human-readable explanations for each prediction using a Large Language Model (LLM). This project bridges the gap between complex ML predictions and clinical interpretability, allowing healthcare professionals and patients to understand *why* a model made a specific prediction.

## ✨ Features

- **Exploratory Data Analysis (EDA)**: Comprehensive analysis of patient data to uncover insights, correlations, and data distributions before modeling.
- **ML Prediction**: Robust machine learning models trained on health data to accurately predict patient risks.
- **Explainable AI (XAI)**: Extraction of feature importance and decision paths from the trained black-box or ensemble models.
- **LLM-Assisted Explanations**: Integration with Large Language Models to translate technical XAI outputs into intuitive, human-readable explanations.
- **Interactive Notebooks**: Step-by-step Jupyter notebooks for exploratory data analysis, model training, and LLM integration.

## 🚀 Getting Started

### Prerequisites

Make sure you have Python 3.8+ installed. You will also need API access to your chosen LLM (e.g., OpenAI API key) if you are using a cloud-based model.

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/health-risk-ai.git
   cd health-risk-ai
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables (e.g., for LLM API keys):

   ```bash
   export LLM_API_KEY="your-api-key-here"
   ```

### Usage

You can explore the iterative development process in the `notebooks/` directory:

- `01_eda.ipynb`: Exploratory Data Analysis.

To run a sample prediction pipeline from the command line:

```bash
python src/predict.py --input data/raw/sample_patient.json
```

## 💻 Example Code Snippet

Here is an example of how to use the trained machine learning model alongside the LLM explainer to generate readable insights:

```python
from src.models import RiskPredictor
from src.explainers import LLMExplainer

# 1. Load patient data
patient_data = {
    "age": 65, 
    "cholesterol": 240, 
    "blood_pressure": "140/90", 
    "smoking_status": "Former"
}

# 2. Initialize the ML model and LLM Explainer
predictor = RiskPredictor(model_path="results/models/rf_model.pkl")
explainer = LLMExplainer(api_key="YOUR_API_KEY")

# 3. Get the ML prediction and top contributing features
risk_score, top_features = predictor.predict(patient_data)
print(f"Predicted Risk Score: {risk_score:.2f}")

# 4. Generate a human-readable explanation using the LLM
explanation = explainer.generate_explanation(
    patient_data=patient_data, 
    risk_score=risk_score, 
    feature_importance=top_features
)
print(f"Explanation:\n{explanation}")
```

## 📁 Project Structure

```text
health-risk-ai/
│
├── data/               # Datasets used for training and testing
│   ├── raw/            # Unprocessed data (e.g., heart.csv)
│   └── processed/      # Cleaned and transformed data ready for modeling
│
├── notebooks/          # Jupyter notebooks for interactive development
│   └── 01_eda.ipynb        # Exploratory Data Analysis
│
├── src/                # Source code for the project
│   ├── data_prep.py    # Data loading and preprocessing scripts
│   ├── models.py       # ML model definitions and training logic
│   └── explainers.py   # LLM integration and explainability tools
│
├── results/            # Saved models, figures, and predicted outputs
├── requirements.txt    # Python project dependencies
└── README.md           # Project documentation
```

## 🤝 Contributing

Contributions are welcome! If you'd like to improve the models, add new LLM integrations, or enhance the explanations, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/amazing-feature`).
3. Commit your changes (`git commit -m 'Add amazing feature'`).
4. Push to the branch (`git push origin feature/amazing-feature`).
5. Open a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
