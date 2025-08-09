# 🚢 Titanic Survival Predictor

## 📌 Overview
This project uses the famous Titanic dataset to build a **machine learning classification model** that predicts whether a passenger survived or not based on key features such as **Age, Gender, Ticket Class, and Fare**. The dataset is cleaned, preprocessed, and analyzed to extract meaningful insights before training the model.

---

## 🗂 Dataset
The dataset used is the **Titanic dataset** from [Kaggle](https://www.kaggle.com/c/titanic/data), containing passenger details and survival status.

**Key Features:**
- `Pclass` – Passenger ticket class (1, 2, or 3)
- `Sex` – Gender of the passenger
- `Age` – Passenger's age
- `Fare` – Ticket price
- `SibSp` – Number of siblings/spouses aboard
- `Parch` – Number of parents/children aboard
- `Embarked` – Port of embarkation

**Target Variable:**
- `Survived` – Survival status (0 = Did not survive, 1 = Survived)

---

## ⚙️ Steps Performed
1. **Data Loading & Inspection**
   - Load CSV dataset
   - Check for missing values and data types

2. **Data Cleaning**
   - Handle missing values
   - Encode categorical features (`Sex`, `Embarked`)
   - Feature selection

3. **Exploratory Data Analysis (EDA)**
   - Survival rate by gender, class, and age group
   - Correlation heatmaps

4. **Data Preprocessing**
   - Scaling numerical features
   - Splitting into training and testing sets

5. **Model Training**
   - Logistic Regression
   - Decision Tree / Random Forest
   - Model hyperparameter tuning (if applicable)

6. **Model Evaluation**
   - Accuracy, Precision, Recall, F1-score
   - Confusion Matrix
   - Classification Report
   - Cross-validation scores

---

## 🛠️ Tools & Libraries
- Python 3.x
- Pandas
- NumPy
- Matplotlib & Seaborn
- Scikit-learn

---

## 📊 Results
**Overall Performance:**
- **Accuracy:** 0.8101
- **Precision:** 0.7778
- **Recall:** 0.7568
- **F1-Score:** 0.7671

---

## 📌 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Titanic_Survival_Predictor.git

2. Navigate into the project folder:
pip install numpy pandas matplotlib seaborn scikit-learn

3. Install the required Python libraries:
pip install numpy pandas matplotlib seaborn scikit-learn

4. Open and run the Jupyter Notebook:
jupyter notebook titanic_survival_classification.ipynb

