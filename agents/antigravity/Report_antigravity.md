# Predicting Physical Arts Engagement: Analysis and Modelling Report

## Purpose of the Analysis
This analysis aims to identify populations that may face structural or contextual barriers to physical arts engagement. Rather than viewing arts participation as merely a matter of individual preference, this project frames non-participation as an indicator of potential under-engagement. By predicting whether individuals have engaged with the arts physically in the past 12 months, we seek to provide actionable evidence that can inform more inclusive cultural policies and targeted public engagement strategies.

## Overview of the Data and Approach
We used survey data from the 2024-25 participation dataset, which includes demographic, socioeconomic, and geographic information such as age, gender, employment status, education, financial hardship, and internet usage. After cleaning the data to remove non-informative responses, we prepared the features to train predictive algorithms. We tested and compared two distinct approaches: Logistic Regression (a standard additive model) and XGBoost (an advanced tree-based model that can capture complex, non-linear patterns). Our evaluation focused highly on the models' ability to distinguish between engaged and non-engaged individuals, measured through a metric known as ROC-AUC, which is robust against the dataset's heavy tilt toward active arts participants.

## Main Findings
Our exploratory analysis and modelling revealed that arts engagement is indeed patterned across different demographic features. In terms of predictive performance on unseen data:
- A basic, untuned model achieved a baseline discrimination score (ROC-AUC) of approximately 0.635.
- Through careful tuning, the Logistic Regression model improved this score to 0.660.
- The tuned XGBoost model achieved the best overall performance, reaching a score of 0.688.

However, the dataset presents a stark imbalance—the vast majority of respondents reported engaging with the arts, leaving very few non-engaged records (around 5% of the test pool). Consequently, while the models are modestly effective at ranking individuals by their likelihood to participate, they currently predict "engaged" for almost all users by default in order to maximize raw accuracy.

## Final Model Choice
We selected the **Tuned XGBoost** model as our final choice. It demonstrated the strongest ability to mathematically separate those who engaged from those who did not, outperforming the Logistic Regression models on our held-out test data. Its capacity to learn complex relationships between demographic factors made it the most capable tool for this specific dataset.

## Practical Implications
For policymakers, these predictive insights can guide resource allocation. Although the model cannot perfectly pinpoint every non-participant without threshold adjustments, it securely highlights the demographic and socioeconomic profiles most closely associated with non-engagement. Cultural programs and funding can be proactively directed toward these identified at-risk profiles—such as specific age brackets or communities facing financial hardship—to dismantle barriers and foster broader inclusivity.

## Key Limitations and Cautions
It is critical to note that predictive modeling does not establish cause and effect. A variable strongly associated with non-participation does not necessarily cause it. Furthermore, the extreme imbalance in our data means the model currently prioritizes the majority class and struggles to explicitly label the minority correctly using standard thresholds. Any policy interventions developed from these findings should use the model's probability rankings as a supportive guide rather than an absolute directive, and should be supplemented with qualitative research into the lived experiences of non-participants.
