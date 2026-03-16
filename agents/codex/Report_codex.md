# Arts Engagement Risk Modelling Report (Codex)

## Purpose
This analysis was designed to help a government arts department identify groups that may be at higher risk of **not** engaging with arts activities in person. The goal is not to explain individual behaviour causally, but to provide practical evidence that can support more targeted and inclusive outreach planning.

## Data and Approach
We used 34,378 survey records from the UK Participation Survey subset, with one target variable and ten demographic, socioeconomic, digital, and geographic predictors. Following protocol, records with unusable target values were removed, leaving 34,338 records for modelling.

The target is imbalanced: most respondents were engaged, while a smaller group were not engaged. Because of this, we used an evaluation approach that focused on identifying under-engagement reliably, rather than relying only on overall accuracy.

Three models were compared:
- Baseline Logistic Regression
- Tuned Logistic Regression
- Tuned XGBoost

All models used the same train/validation/test split and the same evaluation framework.

## Main Findings
On the test set:
- **Baseline Logistic Regression**: Recall for under-engaged respondents = **0.678**, PR-AUC = **0.287**, balanced accuracy = **0.713**, accuracy = **0.741**.
- **Tuned Logistic Regression**: performance was effectively the same as baseline under this dataset and tuning range.
- **Tuned XGBoost**: higher overall accuracy (**0.913**) but much lower recall for under-engagement (**0.077**), meaning many under-engaged respondents would be missed.

This pattern matters for policy use: a model that misses most under-engaged people is less useful for targeted intervention, even if overall accuracy appears high.

## Final Model Choice
Using a weighted selection framework that prioritised under-engagement recall and PR-AUC, and also considered interpretability and operational practicality, **Baseline Logistic Regression** was selected as the final model.

The chosen model provides substantially stronger identification of under-engaged respondents than XGBoost in this setting, and its transparency supports communication with policy stakeholders.

## Practical Implications
The model can be used as a decision-support tool to help prioritise outreach cohorts for arts participation programmes. It is most suitable for screening and targeting, not for replacing local judgement or community consultation.

## Limitations and Cautions
- This is a predictive model, not a causal analysis.
- Non-response and “unknown” categories are present in several variables and may reflect survey process effects.
- Performance can vary over time and should be monitored if reused on future survey waves.
- Any deployment should include fairness and impact checks before operational use.
