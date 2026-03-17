# Arts Participation Risk Screening: Summary for Policy Teams

## Purpose
This analysis was designed to help identify people who may be at higher risk of not engaging physically with arts activity in the last 12 months. The goal is practical: support better targeting of outreach and inclusion work, not to make causal claims about why participation does or does not happen.

## Data and approach in plain terms
We used survey data with 34,378 responses and focused on whether someone reported physical arts engagement. For prediction, we used age band, gender, employment status, qualification level, financial situation, internet use frequency, region, urban/rural location, children in household, and cohabitation status.

Some survey responses used special numeric codes that mean “not answered,” “not applicable,” or similar. Instead of deleting large parts of the dataset, we converted those cases to an explicit “Unknown” category. This kept broad population coverage, although it means some variables carry limited detail in many records.

We compared three models:
- baseline logistic regression,
- tuned logistic regression,
- tuned XGBoost.

The comparison used a shared scorecard, with strongest weight on recall (finding under-engaged people), then F1, PR-AUC, balanced accuracy, and precision.

## Main findings
All three models performed similarly overall. On the final test set:
- Baseline logistic regression recall: **0.678**
- Tuned logistic regression recall: **0.676**
- Tuned XGBoost recall: **0.689**

Tuned XGBoost also had the strongest PR-AUC (**0.293**) and ROC-AUC (**0.790**) among tested options, although differences were moderate. Precision remained low across models (around **0.21**), which means many flagged cases will not actually be under-engaged.

## Final model choice
Using the agreed weighted framework (recall-first), **tuned XGBoost** was selected as the final model. The gain over tuned logistic regression is real but not large, so this should be viewed as an incremental improvement rather than a major step-change.

## Practical implications for the department
The model can be used as an early-warning screening tool to prioritize supportive engagement efforts and local follow-up. It is better suited to broad targeting than to strict eligibility decisions. Because precision is low, actions should be proportionate and non-stigmatizing.

## Limitations and cautions
This is a predictive model, not evidence of cause and effect. Results are sensitive to survey coding and missing-information patterns (for example, cohabitation had a very large “Unknown” share). Performance should be rechecked over time and across regions before operational rollout.
