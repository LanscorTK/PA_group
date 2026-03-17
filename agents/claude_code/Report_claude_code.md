# Report: Identifying Under-Engagement in Physical Arts Participation

## Purpose

This analysis supports evidence-based cultural policy by identifying population groups that are less likely to engage in physical arts activities. Using data from the UK Participation Survey 2024–25, we developed predictive models to detect patterns of under-engagement across demographic, socioeconomic, digital, and geographic factors. The goal is to help arts organisations and policymakers direct outreach and resources towards groups facing potential barriers to participation.

## Data and Approach

The dataset comprises 31,351 respondents from the annual UK Participation Survey, each described by ten variables covering age, gender, employment status, education, financial circumstances, internet usage, region, urban/rural setting, household composition, and cohabitation status. The target variable indicates whether a respondent physically engaged with the arts in the previous twelve months.

Approximately 8% of respondents were identified as under-engaged. After cleaning coded missing values using a tiered strategy (dropping rows for low-missingness variables, recoding to an "Unknown" category for high-missingness ones), we trained and compared three models: a baseline logistic regression, a tuned logistic regression, and a tuned XGBoost gradient-boosted tree classifier. Data was split into training (70%), validation (15%), and test (15%) sets, with the test set reserved exclusively for final evaluation.

## Findings

All three models achieved similar performance on the held-out test set. The baseline logistic regression correctly identified 68% of under-engaged respondents (recall), with a balanced accuracy of 70.0% and an ROC-AUC of 0.77. The tuned logistic regression achieved 67% recall and 70.2% balanced accuracy. The tuned XGBoost model reached 67% recall and 69.4% balanced accuracy.

## Model Choice

The **Baseline LR** was selected as the final model using a weighted scoring framework that prioritises recall (identifying under-engaged groups), balanced accuracy, and interpretability. Logistic regression offers transparent coefficients that are straightforward to communicate to non-technical stakeholders, which is essential for policy applications.

## Practical Implications

The model can help identify demographic and socioeconomic profiles associated with lower arts participation. This information may guide targeted outreach campaigns, programme design, and resource allocation to improve inclusivity in cultural engagement.

## Limitations

The model identifies statistical associations, not causal relationships — non-participation may reflect preferences rather than barriers. Precision for the under-engaged class remains modest, meaning some engaged individuals are incorrectly flagged. High missingness in cohabitation status and education data may limit the model's sensitivity to these factors. Results should be used alongside qualitative research and domain expertise rather than as a standalone decision tool.
