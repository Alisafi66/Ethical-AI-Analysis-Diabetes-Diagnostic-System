# Ethical AI Analysis: Diabetes Diagnostic System

## Project Overview
This repository contains the technical implementation, ethical evaluation, and bias-mitigation analysis for a diabetes diagnostic prediction system. The project assesses model fairness across protected demographic characteristics (**Gender** and **Age Group**) to ensure equitable clinical diagnostic support and alignment with Trustworthy AI principles.

---

## Key Findings
* **Model Fairness:** The **Random Forest** classifier demonstrated the highest individual and group fairness, consistently maintaining the lowest disparate impact across all tested demographics compared to Logistic Regression and Decision Trees.
* **Age Sensitivity (Systemic Bias):** Analysis revealed that all baseline models exhibited age-based bias, with the "Senior" demographic experiencing disproportionately higher selection and error rates. This indicates that model performance is heavily influenced by age-related feature distributions.
* **Mitigation Effectiveness:** Implementing **sample-weighting** successfully reduced disparate mistreatment in the tree-based models, effectively narrowing the performance gap between demographic segments without significantly sacrificing overall diagnostic accuracy.
* **Individual Fairness:** The Logistic Regression model achieved perfect individual fairness (0.00% disparate treatment), ensuring that patients with identical clinical profiles received identical predictions regardless of their protected attributes.

---

## Methodology & Ethical Framework
The pipeline rigorously evaluates the "black box" nature of machine learning algorithms through the following ethical lenses:
1. **Baseline Evaluation:** Establishing predictive performance using standard metrics and confusion matrices.
2. **Disparate Impact:** Measuring selection rate bias across demographic subgroups.
3. **Disparate Mistreatment:** Analyzing the variance in error rates to ensure no group bears a disproportionate burden of misdiagnosis.
4. **Disparate Treatment (Individual Fairness):** Conducting pair-matching analysis to isolate the effect of sensitive attributes on predictions.
5. **Discrimination Discovery:** Utilizing the **eLift** metric to ensure specific intersectional rules (e.g., "Female & Senior") are non-discriminatory.
6. **Bias Mitigation:** Applying class/sample weighting to penalize errors in underrepresented or historically disadvantaged groups.

---

## Repository Contents

| File / Folder | Description |
| :--- | :--- |
| 📓 `notebooks/M515A_Ethical_issues_for_AI.ipynb` | The core Python notebook containing all model training, fairness calculations, and visualizations. |
| 📄 `docs/M515A_Project_Report.pdf` | The formal academic report detailing the methodology, clinical context, and critical ethical discussion. |
| 🌐 `docs/M515A_Notebook_Preview.html` | A rendered, human-readable HTML version of the technical workflow for easy browser viewing. |

---

## Technical Stack
* **Language:** Python
* **Machine Learning:** `scikit-learn` (Logistic Regression, Random Forest, Decision Tree)
* **Data Manipulation:** `pandas`, `numpy`
* **Data Visualization:** `matplotlib`, `seaborn`

## How to Navigate
To review the raw code and mathematical proofs for the fairness metrics, please refer to the Jupyter Notebook in the `/notebooks` directory. For a synthesized analysis regarding the trade-off between statistical bias and clinical biological reality, please consult the formal report in the `/docs` folder.
