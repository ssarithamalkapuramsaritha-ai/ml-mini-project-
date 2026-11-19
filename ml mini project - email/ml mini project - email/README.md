# Email Spam Detection using Machine Learning

A machine learning project that classifies text messages as **Spam** or **Ham (Not Spam)** using TF-IDF text features and compares three classification algorithms: Naive Bayes, Logistic Regression and Random Forest.

## Team

| Roll No | Name |
|---|---|
| 25RBAID0001 | M Saritha |
| 25RBAID0016 | P Anjali Anudipthi |
| 25RBAIM0068 | M Bhupal |
| 25RBAIM0086 | V Vinay |

---

## Dataset

**SMS Spam Collection Dataset** (`spam.csv`)

| Property | Value |
|---|---|
| Total messages | 5,572 |
| Ham (not spam) | 4,825 |
| Spam | 747 |
| Raw columns | 5 (`v1`, `v2`, `Unnamed: 2`, `Unnamed: 3`, `Unnamed: 4`) |
| Columns used | `v1` (label) and `v2` (message text) |
| Encoding | `latin-1` |

The three `Unnamed` columns are almost entirely empty (50, 12 and 6 non-null values) and are dropped during cleaning. The dataset is imbalanced (about 87% ham, 13% spam), so a stratified split is used.

**Where to get it:** download `spam.csv` from the "SMS Spam Collection Dataset" on Kaggle, or the original from the UCI Machine Learning Repository ("SMS Spam Collection"). Place it in the same folder as the notebook.

---

## Project Structure

```
email-spam-detection/
├── email_spam_detection.ipynb   # Main notebook (all code)
├── spam.csv                     # Dataset (download separately)
├── algorithm_comparison.csv     # Generated: accuracy of each algorithm
├── spam_model.pkl               # Generated: Naive Bayes model
├── tfidf_vectorizer.pkl         # Generated: TF-IDF vectorizer
├── final_spam_model.pkl         # Generated: final Random Forest model
├── final_tfidf.pkl              # Generated: TF-IDF vectorizer for final model
└── model/
    ├── spam_model.pkl           # Generated: Random Forest model
    └── tfidf.pkl                # Generated: TF-IDF vectorizer
```

---

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- jupyter (to run the notebook)

Install everything with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

---

## How to Run

1. Download `spam.csv` and put it next to the notebook.
2. Start Jupyter:
   ```bash
   jupyter notebook
   ```
3. Open `email_spam_detection.ipynb` and choose **Kernel → Restart & Run All**.
4. At the end, the notebook opens an interactive prompt. Type a message to check it, or type `exit` to stop.

---

## Workflow

1. **Import libraries**
2. **Load dataset** with `latin-1` encoding
3. **Dataset information**: shape, dtypes, missing values
4. **Data cleaning**: keep only `v1` and `v2`, rename to `Category` and `Message`
5. **Class distribution**: count of ham vs spam
6. **Label encoding**: ham = 0, spam = 1
7. **Visualization**: bar chart of spam vs ham
8. **Train/test split**: 80/20, `random_state=42`, stratified (4,457 train / 1,115 test)
9. **TF-IDF feature extraction**: English stop words removed, lowercase
10. **Naive Bayes training** and prediction
11. **Accuracy** and **confusion matrix**
12. **Classification report**
13. **`check_spam()` function** for single messages
14. **Save** Naive Bayes model and vectorizer with `pickle`
15. **Algorithm comparison**: Naive Bayes vs Logistic Regression vs Random Forest
16. **Random Forest evaluation**: confusion matrix and report
17. **Save final model** and vectorizer
18. **Interactive prediction** loop
19. **Final performance summary** and conclusion

---

## Results

### Algorithm comparison

| Algorithm | Accuracy |
|---|---|
| Naive Bayes | 96.86% |
| Logistic Regression | 96.77% |
| **Random Forest** | **97.49%** |

### Final model: Random Forest

| Metric | Score |
|---|---|
| Accuracy | 97.49% |
| Precision (spam) | 1.00 |
| Recall (spam) | 0.81 |
| F1 score (spam) | 0.90 |

### Interpretation

- **Precision of 1.00** means no legitimate message was wrongly flagged as spam on the test set.
- **Recall of 0.81** means about 19% of spam messages were missed. For example, the interactive test "Congratulations! You won a lottery!" was classified as *NOT SPAM*.
- Accuracy alone is misleading here because of the class imbalance, so precision, recall and F1 for the spam class are the more useful measures.

---

## Using the Saved Model

```python
import pickle

with open("model/spam_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("model/tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

def predict_spam(message):
    vec = tfidf.transform([message])
    return "SPAM MESSAGE" if model.predict(vec)[0] == 1 else "NOT SPAM"

print(predict_spam("Claim your free cash prize now"))
```

Only load `.pkl` files that you created yourself or trust, since pickle can run arbitrary code when loading.

---

## Possible Improvements

- Handle class imbalance (`class_weight="balanced"`, SMOTE or threshold tuning) to improve spam recall.
- Add `ngram_range=(1, 2)` to TF-IDF so phrases such as "you won" are captured.
- Try cross-validation and hyperparameter tuning (`GridSearchCV`).
- Test on real email data, since this dataset consists of SMS messages.
- Wrap the model in a small web app (Flask or Streamlit).

---

## Notes

- The dataset contains SMS messages, while the project title refers to email. The same approach applies to both, but results on real emails may differ.
- The notebook saves the model several times under different names (steps 14, 19 and 21). The files in `model/` are the ones to use for deployment.

## Credits

Dataset: SMS Spam Collection (UCI Machine Learning Repository / Kaggle).
