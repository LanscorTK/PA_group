# Understanding Patterns of Under-Engagement in Physical Arts Participation

## Purpose

This analysis was conducted to help identify groups of adults in England who may face barriers to engaging with the arts physically. Rather than treating non-participation as a simple personal choice, the analysis investigates whether under-engagement follows systematic patterns across demographic, socioeconomic, and geographic factors. The aim is to support evidence-based cultural policy by highlighting where targeted outreach may be most needed.

## Data and Approach

The analysis used the UK Participation Survey 2024-25, covering over 34,000 adult respondents. The dataset records whether each respondent attended or participated in the arts physically in the last 12 months, alongside ten background characteristics including age, gender, employment status, education level, financial situation, internet usage, region, urban/rural setting, household composition, and cohabitation status.

After cleaning the data to address survey non-response patterns, approximately 29,800 records were used for modelling. The data was split into training, validation, and test sets, and three classification models were compared: a baseline logistic regression, a tuned logistic regression, and a tuned XGBoost model.

## Main Findings

The analysis confirmed substantial class imbalance: roughly 91% of respondents had engaged with the arts, while only 9% had not. This imbalance is important context for interpreting results.

On the held-out test set, the baseline logistic regression achieved the strongest overall performance, correctly identifying 67% of under-engaged individuals (recall) with a balanced accuracy of 69% and an ROC-AUC of 0.75. The tuned logistic regression performed comparably, while XGBoost showed slightly lower recall at 66%.

## Model Choice

The baseline logistic regression was selected as the final model. It achieved the highest recall and balanced accuracy among all three models, and its transparent, interpretable structure makes it well suited for policy communication. Logistic regression allows clear identification of which factors are most associated with under-engagement, supporting actionable recommendations.

## Practical Implications

The model can help identify demographic and socioeconomic profiles associated with lower arts engagement. This could inform targeted outreach strategies, helping arts organisations and public bodies focus resources on communities that are less likely to participate. For example, differences across age bands, employment status, and financial hardship may point to structural barriers that policy interventions could address.

## Limitations

This analysis identifies statistical associations, not causal relationships. The model cannot determine why certain groups are less engaged, only that patterns exist. The dataset's class imbalance means that precision remains relatively low: many individuals flagged as potentially under-engaged may in fact be engaged. Additionally, the high rate of missing data for cohabitation status (71%) and education level (24%) required recoding to an "Unknown" category, which may obscure relevant patterns in these variables. Any policy decisions informed by these results should be supplemented with qualitative research and local knowledge.
