# Predicting Arts Engagement in England: Findings and Implications

## Purpose

This analysis was conducted to better understand who participates in arts activities in England and, more importantly, who does not. Using data from the national Participation Survey (2024–25), we developed a predictive model to identify population groups that may be under-engaging with the arts. The aim is to support evidence-based cultural policy by highlighting where barriers to participation may exist.

## Data and Approach

The analysis drew on survey responses from over 29,000 adults aged 16 and above in England. The central question was whether a respondent had physically attended or participated in an arts activity in the past 12 months. Ten demographic and socioeconomic characteristics were used as predictors, including age, gender, employment status, education level, financial situation, internet usage, geographic region, urban/rural setting, household composition, and cohabitation status.

Approximately 93% of respondents reported arts engagement, while 7% did not — a significant imbalance that shaped our modelling approach. We tested two types of model: Logistic Regression (a transparent, interpretable method) and XGBoost (a more complex machine learning technique). Both were tuned and compared using metrics that account for the difficulty of identifying the smaller non-engaged group.

## Key Findings

The best-performing model was a Logistic Regression with balanced class weighting. On held-out test data, this model achieved a ROC-AUC of 0.76 and correctly identified 67% of non-engaged individuals, while maintaining 72% overall accuracy. The model showed that age, education level, financial hardship, and internet usage were among the most informative predictors of engagement patterns.

XGBoost achieved marginally higher overall accuracy (76%) but was less effective at identifying the non-engaged minority. Given the policy goal of finding under-served groups, the Logistic Regression model was selected for its stronger recall of non-participants and its greater transparency.

## Practical Implications

The results suggest that arts non-participation is not random but is associated with identifiable social and economic factors. Older adults, those without degree-level qualifications, individuals experiencing financial difficulty, and less frequent internet users appear more likely to be non-engaged. These patterns may reflect structural barriers — such as cost, access, or awareness — rather than simple lack of interest.

## Limitations

This analysis identifies statistical associations, not causal relationships. The survey's self-reported nature may introduce response bias. The high engagement rate (93%) makes it inherently difficult to predict the minority class with high precision. Additionally, our definition of "engagement" covers a broad range of activities and may not capture more informal or community-based participation.

These findings should be treated as indicative rather than definitive, and used alongside qualitative research and local knowledge when informing policy decisions.
