from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 4.1 Prepare Modeling Data
X = participation_clean.drop(columns=['arts_engaged'])
y = (participation_clean['arts_engaged'] == 2.0).astype(int) 
# Class 1: Under-engaged, Class 0: Engaged

# Identifing variables
numerical_cols = ['AGEBAND', 'CHILDHH']
categorical_cols = ['SEX', 'QWORK', 'EDUCAT3', 'FINHARD', 'CINTOFT', 'gor', 'rur11cat', 'COHAB']

# Preprocessing for Logistic Regression
lr_preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ])

# Preprocessing for XGBoost
xgb_preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Stratified Splitting (0.7 / 0.15 / 0.15)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=random_state)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=(0.15 / 0.85), stratify=y_train_val, random_state=random_state)

print(f"Train size: {len(X_train)}  |  Validation size: {len(X_val)}  |  Test size: {len(X_test)}")
print(f"Target Imbalance (Under-engaged %): {y.mean()*100:.1f}%")

# 4.2 Create evaluation harness
def evaluate_model(model, X_eval, y_eval, model_name="Model"):
    y_pred = model.predict(X_eval)
    y_prob = model.predict_proba(X_eval)[:, 1]
    
    print(f"==== Evaluation Results: {model_name} ====")
    print(classification_report(y_eval, y_pred))
    
    roc_auc = roc_auc_score(y_eval, y_prob)
    pr_auc = average_precision_score(y_eval, y_prob)
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    
    cm = confusion_matrix(y_eval, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Engaged (0)', 'Under-engaged (1)'])
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f"Confusion Matrix: {model_name}")
    plt.show()

# 4.3 Baseline Model: Logistic Regression
lr_pipeline = Pipeline([
    ('preprocessor', lr_preprocessor),
    ('classifier', LogisticRegression(random_state=random_state, max_iter=1000, class_weight='balanced'))
])

lr_pipeline.fit(X_train, y_train)
evaluate_model(lr_pipeline, X_val, y_val, model_name="Baseline Logistic Regression (Validation Set)")
