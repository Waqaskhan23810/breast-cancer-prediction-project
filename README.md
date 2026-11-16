# 🩺 Breast Cancer Prediction

A beginner-friendly **machine learning** project that predicts whether a tumor is **malignant** or **benign** using the built-in Breast Cancer Wisconsin dataset from scikit-learn.

## 📌 Project Overview
- **Problem:** Binary classification (malignant vs benign)
- **Dataset:** 569 patients, 30 numeric features (no download needed)
- **Models:** Logistic Regression and Random Forest
- **Result:** Logistic Regression reached about **98% accuracy** on the test set

## 🛠️ Tools Used
Python • Pandas • Matplotlib • Scikit-learn

## 📂 Project Structure
```
breast-cancer-prediction/
├── main.py            # complete project code
├── requirements.txt   # libraries
├── outputs/           # charts and results (created after running)
└── README.md
```

## 🚀 How to Run
```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/breast-cancer-prediction.git
cd breast-cancer-prediction

# 2. Install libraries
pip install -r requirements.txt

# 3. Run
python main.py
```
Charts and results are saved in the `outputs/` folder.

## 🔍 Steps Followed
1. Loaded the data and checked shape and missing values
2. Explored class distribution and the most important features
3. Split data into 80% train / 20% test
4. Trained two models and compared accuracy
5. Evaluated the best model with a classification report and confusion matrix

## 📊 Results
| Model | Accuracy |
|-------|----------|
| Logistic Regression | 0.982 |
| Random Forest | 0.956 |

Charts: `outputs/class_distribution.png`, `outputs/top_features.png`, `outputs/confusion_matrix.png`

## 💡 Key Takeaways
- Data is clean (no missing values), so preprocessing was limited to scaling.
- A simple model like Logistic Regression can beat a more complex one.
- This is an educational project and **not** for real medical diagnosis.

## 👤 Author
**Your Name** — [LinkedIn]www.linkedin.com/in/waqas73 | [GitHub]https://github.com/Waqaskhan23810