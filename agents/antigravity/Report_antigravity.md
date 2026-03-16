# Identifying Barriers to Arts Engagement: Predictive Analysis Report

**Purpose of the Analysis**
The primary goal of this analysis is to identify structural and socioeconomic patterns associated with under-engagement in the arts. Rather than treating arts participation purely as an issue of individual taste, this project treats "under-engagement" as a reflection of potential societal barriers. By predicting which populations are least likely to engage physically with the arts, we can purposefully direct cultural policy and tailor public outreach strategies to foster inclusivity.

**Data and Approach**
This analysis utilized the UK Participation Survey (2024–25), examining responses from roughly 34,000 adults. We tracked physical arts engagement over the last 12 months as our target outcome. We evaluated standard demographic and socioeconomic factors—such as age, rural/urban geography, employment status, education, and financial hardship—to see if they could predict a lack of engagement. 

We built and compared two types of predictive models: Logistic Regression (a traditional statistical approach) and XGBoost (a more complex, non-linear machine learning algorithm). In establishing a fair test, a portion of the data was strictly reserved to evaluate how successfully the models could flag "under-engaged" citizens on strictly unseen data.

**Main Findings**
Our exploratory analysis confirmed that arts engagement is deeply socially patterned. Notably, individuals experiencing severe financial hardship and specific age demographics showed notably lower engagement rates.
During modeling, simply predicting that everyone "engages" achieves high baseline accuracy because a vast majority of the population does participate. However, our models specifically prioritized identifying the minority who *do not* engage. The models successfully flagged around 70-71% of the truly under-engaged citizens (Recall). 

**Final Model Choice**
We selected the **Tuned XGBoost** model as our final choice. While the baseline Logistic Regression provided similar baseline indicators, XGBoost proved slightly more adept at capturing complex, overlapping factors (like the intersection of rural living contexts and financial difficulty) and maintained an excellent identification rate (71% recall) without entirely sacrificing precision. 

**Practical Implications**
The findings clearly emphasize that a "one-size-fits-all" policy will fail to encompass the entire population. Policymakers should target interventions directly toward the communities our model flags as highly likely to be under-engaged—specifically focusing on removing financial entry barriers and increasing targeted community outreach in specific geographic constraints. By understanding exactly *who* is slipping through the cracks, departments can confidently allocate funding where it is needed most.

**Limitations and Cautions**
While powerful, this predictive model serves strictly as a decision-support tool, not a measure of causation. The algorithm identifies *correlations* between financial hardship and lack of engagement, but it cannot definitively prove that one directly causes the other. Furthermore, the model has a noticeable false-positive rate, meaning it will sometimes flag individuals as under-engaged when they are actually participating. Policies should, therefore, remain broad enough to benefit entire communities rather than rigidly restricting resources to individual model predictions.
