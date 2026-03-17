# Arts Engagement Risk Screening Report

## Purpose of the analysis
This analysis was designed to help the department identify groups that may be less likely to engage with arts activities in person. The goal is practical: support better targeting of outreach and inclusion efforts, especially where barriers may be linked to social and economic conditions rather than simple personal preference.

## Data and approach in brief
We used 34,378 survey records, then removed 40 records where the outcome field was marked as missing. The final modelling sample was 34,338 respondents. The model used ten background factors available to policy teams: age band, gender, employment status, education, financial situation, internet-use frequency, region, rural/urban area, children in household, and living-as-a-couple status.

Three models were tested with the same train/validation/test process:
- Baseline Logistic Regression
- Tuned Logistic Regression
- Tuned XGBoost

Because non-participation is a smaller group, we evaluated not only overall discrimination but also how well each model identifies likely under-engagement cases.

## Main findings
On the held-out test set, all models showed moderate predictive ability. The tuned XGBoost model gave the strongest overall balance:
- ROC-AUC: **0.726**
- PR-AUC: **0.225**
- Recall for under-engagement: **0.621**
- Precision for under-engagement: **0.178**
- F2 score (recall-weighted): **0.415**

The tuned Logistic Regression reached higher recall (**0.746**) but produced more false alarms (precision **0.141**) and weaker ranking quality (PR-AUC **0.206**). The baseline model was weaker than both tuned models.

## Final model choice and rationale
Using a pre-defined weighted framework (emphasising recall, then ranking quality, balanced accuracy, and operational precision), the selected model was **Tuned XGBoost**. Its selection score was **0.383**, compared with **0.364** for tuned Logistic Regression.

In policy terms, this means XGBoost provided the most useful trade-off between finding at-risk groups and keeping over-targeting at a manageable level.

## Practical implications for public arts engagement
This model can support a screening-style workflow: identify areas or respondent profiles with elevated risk of non-participation, then prioritise targeted engagement actions (for example, tailored community programming, geographic outreach, or affordability-focused messaging). It should be used as a decision-support signal, not an automatic decision maker.

## Limitations and cautions
The model is predictive, not causal. It cannot tell us why people are under-engaged, only which patterns are associated with it in this dataset. Precision remains modest, so many flagged individuals may still engage. Results may shift over time if population behaviour changes, so periodic refresh and monitoring are needed before operational deployment.
