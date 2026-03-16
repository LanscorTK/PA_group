from sklearn.model_selection import ParameterGrid
import time
from xgboost import XGBClassifier
import warnings

# Suppress sklearn convergence warnings for clean output
warnings.filterwarnings('ignore')

# Helper function for manual validation set tuning
def tune_model_on_validation(pipeline, param_grid, X_train, y_train, X_val, y_val, model_name):
    grid = ParameterGrid(param_grid)
    best_score = -1.0
    best_params = None
    best_model = None
    
    print(f"\n--- Tuning {model_name} ---")
    start_time = time.time()
    total_iters = len(grid)
    
    for i, params in enumerate(grid):
        # Apply parameters to the pipeline
        pipeline.set_params(**params)
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Evaluate PR-AUC on validation set
        y_val_prob = pipeline.predict_proba(X_val)[:, 1]
        score = average_precision_score(y_val, y_val_prob)
        
        if score > best_score:
            best_score = score
            best_params = params
            best_model = pipeline
            
    end_time = time.time()
    
    # 5.4 Print structured tuning summary
    print("Structured Tuning Summary:")
    print(f"- Tuning method used: Manual Grid Search on Validation Set")
    print(f"- Hyperparameters searched: {list(param_grid.keys())}")
    print(f"- Candidate values: {param_grid}")
    print(f"- Total configurations evaluated: {total_iters}")
    print(f"- Iterations completed: {total_iters}")
    print(f"- Best hyperparameter setting: {best_params}")
    print(f"- Best Validation set PR-AUC: {best_score:.4f}")
    
    # Refit the best model on the combined Train+Val set for optimal final testing
    # Or keep it trained on Train set just to strictly follow "tune on validation only"
    # To be extremely safe and strict to the protocol, we will re-train the optimal model on Train set
    # actually standard practice is training on X_train + X_val before X_test, but we will just set the params.
    pipeline.set_params(**best_params)
    pipeline.fit(X_train_val, y_train_val)
    return pipeline

# 5.1 Tune Logistic Regression
lr_param_grid = {
    'classifier__C': [0.01, 0.1, 1.0, 10.0],
    'classifier__class_weight': ['balanced', None],
    'classifier__solver': ['lbfgs', 'liblinear']
}
tuned_lr_pipeline = tune_model_on_validation(
    Pipeline([('preprocessor', lr_preprocessor), ('classifier', LogisticRegression(max_iter=1000, random_state=random_state))]),
    lr_param_grid, X_train, y_train, X_val, y_val, "Logistic Regression"
)

# 5.2 Train and tune XGBoost
xgb_param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [3, 5, 7],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__scale_pos_weight': [1.0, (len(y_train)-sum(y_train))/sum(y_train)]
}
xgb_pipeline = Pipeline([
    ('preprocessor', xgb_preprocessor),
    ('classifier', XGBClassifier(random_state=random_state, eval_metric='logloss', use_label_encoder=False))
])
tuned_xgb_pipeline = tune_model_on_validation(
    xgb_pipeline, xgb_param_grid, X_train, y_train, X_val, y_val, "XGBoost"
)

# 5.3 Model Comparison on TEST SET
print("\n#####################################################")
print("############# TEST SET MODEL COMPARISON #############")
print("#####################################################\n")

print("Evaluating Baseline Logistic Regression (Test Set)")
# lr_pipeline was fit on X_train. Refit on X_train_val for fairness, or keep as is.
lr_pipeline.fit(X_train_val, y_train_val)
evaluate_model(lr_pipeline, X_test, y_test, "Baseline LR (Test Set)")

print("\nEvaluating Tuned Logistic Regression (Test Set)")
evaluate_model(tuned_lr_pipeline, X_test, y_test, "Tuned LR (Test Set)")

print("\nEvaluating Tuned XGBoost (Test Set)")
evaluate_model(tuned_xgb_pipeline, X_test, y_test, "Tuned XGBoost (Test Set)")
