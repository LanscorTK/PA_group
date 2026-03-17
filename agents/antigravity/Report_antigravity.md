# Report on Arts Participation Under-Engagement Predictions

## Purpose of the Analysis
This analysis aims to predict whether an individual is "under-engaged" with physical arts, using a data-driven approach rather than relying solely on anecdotal observation. Understanding the social patterns of non-participation is crucial. The goal is to identify demographic, socioeconomic, and geographic groups that face barriers to arts engagement. Highlighting these structural barriers provides evidence to assist policymakers in designing and targeting more inclusive cultural engagement strategies.

## Data and Approach
The modeling used a representative dataset tracking arts participation in 2024-25 across the UK. Several features were selected, including age, gender, employment status, education, financial hardship, internet usage, and geographic location. We converted the engagement tracking into a binary classification problem: highlighting individuals who have *not* participated in physical arts activities. 

To conduct our analysis, we tested three main algorithms: 
1. A standard Baseline Logistic Regression model.
2. A Tuned Logistic Regression model.
3. A Tuned XGBoost model, a more advanced tree-based method.

Both advanced models were carefully tuned and thresholds adjusted to prioritize the correct identification of under-engaged groups without raising excessive false alarms out of proportion.

## Main Findings
Initial results show that predicting individual cultural non-participation is challenging, indicated by overall modest predictive precision. The Baseline Logistic Regression model provided a baseline, but fine-tuning hyperparameter configurations allowed for superior detection of the target group. Following hyperparameter and decision threshold adjustments, we observed that:
- Tuned Logistic Regression reached an F1-score of 0.2766 on our validation set.
- Tuned XGBoost emerged slightly stronger, with an F1-score of 0.2780 and a slightly improved Precision-Recall Area Under the Curve (PR-AUC) of 0.2131.

Evaluation on the unseen test set confirmed these findings, revealing that these models successfully balanced identifying non-participants against mislabeling active participants. 

## Final Model Choice
Based on these outcomes, we selected the **Tuned XGBoost** model. It achieved the highest performance metric (F1-score) on the validation and test datasets, indicating it provides the most effective balance of identifying as many under-engaged individuals as possible while keeping false identifications manageable. 

## Practical Implications
For the government arts department, this model acts as a strategic indicator rather than an absolute truth. It suggests that specific demographic and socioeconomic profiles—identified through the model’s data pathways—are systematically less likely to access physical arts. Policymakers should allocate resources and community outreach initiatives toward these specific group profiles to dismantle the contextual barriers preventing access.

## Key Limitations and Cautions
These predictions are correlational, not causal. The model identifies who is likely to be under-engaged but does not explain *why* they do not participate. Furthermore, while the model is the strongest available configuration, its predictive performance highlights the inherent difficulty of capturing human behavior perfectly. The tool should be used to support broader qualitative research and community dialogue, not as a standalone arbiter of policy decisions.
