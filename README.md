## Dataset Setup

Download the Kaggle dataset (IBM HR Attrition) and place the CSV in this folder.

Rename the file to:
- employee_attrition.csv

Note: The dataset is not uploaded to GitHub.

# Employee Attrition Prediction using Machine Learning (Python)

This project builds a machine learning pipeline to predict **employee attrition** using Python and scikit-learn.  
The goal is to help organizations identify employees who are likely to leave, enabling proactive HR decisions.

---

## 📂 Project Structure

```text
EmployeeattritionMLpython/
│
├── data/
│   ├── raw.csv              # Input dataset (not pushed to GitHub if sensitive)
│   └── README.md            # Data description
│
├── src/
│   ├── __init__.py
│   ├── config.py            # Paths & configuration
│   ├── preprocess.py       # Data loading & preprocessing
│   └── train.py             # Model training pipeline
│
├── models/
│   └── README.md            # Model artifacts (model.pkl ignored by git)
│
├── outputs/
│   ├── metrics.txt          # Model evaluation results
│   └── README.md
│
├── requirements.txt         # Python dependencies
├── .gitignore
└── README.md
## 🧠 Machine Learning Workflow

Load employee attrition dataset (raw.csv)

Preprocess data (handle target variable & features)

Split data into train/test sets

Train a Logistic Regression model

Evaluate model using accuracy

Save:

Trained model (model.pkl)

Feature metadata

Evaluation metrics (metrics.txt)

⚙️ Technologies Used

Python 3 (Anaconda)

Pandas

NumPy

Scikit-learn

Joblib

VS Code

Git & GitHub

## 📈 Model Performance

Algorithm: Logistic Regression

Accuracy: ~0.87

🚀 How to Run the Project
1️⃣ Clone the repository
git clone https://github.com/Venkatatech-git/employee-attrition-ml-python.git
cd employee-attrition-ml-python

2️⃣ Create and activate environment (optional but recommended)
conda create -n attrition python=3.10
conda activate attrition

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run training pipeline
python -m src.train

📁 Outputs

After successful execution:

Model saved to:
models/model.pkl

Metrics saved to:
outputs/metrics.txt


👤 Author

Venkata Sai Teja
Beginner Machine Learning & Python Developer

⭐ If you find this project useful, feel free to star the repository!