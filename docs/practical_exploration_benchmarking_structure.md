# Practical Exploration and Benchmarking — Suggested Structure

## 1. Benchmark overview *(~180–220 words)*
- 3 agents compared  
- same dataset / target / features  
- same prompt pipeline or workflow constraints  
- goal: benchmark **end-to-end data science workflow capability**  
- focus not only on final model performance, but also on execution quality and methodological discipline  
- brief note on why this setup is fair and comparable  

## 2. Task coverage and task specification *(~220–260 words)*
Organise tasks into **4 task blocks**, not many small subsections:

- **Data preparation**  
  - ingestion  
  - schema checks  
  - missingness handling  

- **Analytical exploration**  
  - EDA  
  - plots  
  - insight generation  

- **Modelling and improvement**  
  - baseline model  
  - evaluation harness  
  - tuning / feature engineering / model comparison  

- **Reproducibility and communication**  
  - packaging  
  - requirements / seeds / run scripts  
  - README / model card  

For each block, mention briefly:
- task spec  
- success criteria  
- evidence collected  
- failure modes monitored  
- reproducibility check / limitation  

## 3. Evaluation approach *(~180–220 words)*
- **objective criteria**
  - whether the task ran
  - whether outputs met the spec

- **quality criteria**
  - completeness of schema checks
  - appropriateness of EDA outputs
  - justification of missingness handling
  - suitability of evaluation metrics
  - alignment between documentation and actual results

- emphasis on:
  - methodological correctness
  - class imbalance awareness
  - leakage / evaluation discipline
  - usefulness of outputs, not just runnable code

## 4. Main findings *(~250–300 words)*
- which agent was most stable overall  
- which agent was strongest in:
  - cleaning
  - EDA
  - modelling
  - documentation  
- which task types exposed differences most clearly  
- common failure modes:
  - incomplete checks
  - weak imbalance handling
  - evaluation mistakes
  - brittle code
  - shallow documentation  
- brief note on how failures were detected / corrected  

## 5. Short takeaway *(~80–120 words)*
- practical value of the benchmark  
- what the experiments showed overall  
- main limitation(s)  
- transition to later comparative analysis section  

---

## Writing rules
- write as **benchmark design + findings summary**
- do **not** write as Step 1 / Step 2 / Step 3...
- do **not** describe every prompt
- do **not** explain every file or every chart
- move logs / screenshots / detailed scoring / prompt records to appendix

---

## Ultra-short version for teammates
- **P1:** benchmark setup and fairness  
- **P2:** 4 task blocks + task spec / evidence / reproducibility  
- **P3:** evaluation logic: objective + quality criteria  
- **P4–P5:** main findings, strengths, weaknesses, failure modes  
- **P6:** short takeaway / limitation  
