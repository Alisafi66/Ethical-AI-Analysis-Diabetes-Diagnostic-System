# Fair Diabetes Diagnostic System

## Detecting and Mitigating Demographic Bias in AI-Driven Clinical Predictions

### About This Project

A clinical diagnostic support system uses machine learning to predict whether a patient has diabetes based on key health indicators (such as HbA1c and blood glucose levels). This project investigates whether the AI inadvertently discriminates against patients based on their **Gender** or **Age Group**. 

We systematically detect statistical bias, prove its existence across three baseline classifiers (Logistic Regression, Random Forest, Decision Tree), and implement a mitigation strategy to ensure equitable clinical outcomes without sacrificing medical accuracy.

### Dataset

* **Source:** Clinical Diabetes Prediction Dataset
* **Target:** Does the patient have diabetes? (1 = Yes / 0 = No)
* **Sensitive Attributes:**
  * **Gender** (Male / Female)
  * **Age Group** (Young, Middle, Senior)

---

## Three-Component Fairness Analysis

### 1. Group Fairness (Disparate Impact & Mistreatment)
To detect systemic discrimination, we generated per-group confusion matrices and evaluated the classifiers using base error-rate metrics:
* **FPR** (False Positive Rate) = FP / (FP + TN)
* **FNR** (False Negative Rate) = FN / (FN + TP)
* **FDR** (False Discovery Rate) = FP / (FP + TP)
* **FOR** (False Omission Rate) = FN / (FN + TN)

**Disparate Impact (Selection Rate Bias):**
Measures the difference in the rate at which different demographic groups receive a positive prediction $P(\hat{y}=1 | \text{group})$.
* **Logistic Regression:** 0.1556 Disparity (Senior: 16.35% vs. Young: 0.79%)
* **Random Forest:** 0.1416 Disparity (Senior: 15.29% vs. Young: 1.13%) — *Best Baseline*
* **Decision Tree:** 0.1954 Disparity (Senior: 21.52% vs. Young: 1.98%)

**Disparate Mistreatment (Error Rate Bias):**
Measures the difference in overall misclassification rates across groups. The fairness threshold is a disparity of < 0.05.
* **Logistic Regression:** 0.0771 Disparity (DETECTED for Age)
* **Random Forest:** 0.0655 Disparity (DETECTED for Age) — *Best Baseline*
* **Decision Tree:** 0.0970 Disparity (DETECTED for Age)

### 2. Individual Fairness & Discrimination Discovery
Proves whether patients with identical clinical profiles receive different predictions based solely on protected attributes.

**Disparate Treatment (Pair-Matching):**
Compares identical patient profiles that differ *only* by the sensitive attribute.
* **Logistic Regression:** 0.00% difference (Perfect Individual Fairness)
* **Random Forest:** 1.35% difference based on Gender
* **Decision Tree:** 2.69% difference based on Gender

**Discrimination Discovery (eLift):**
Evaluates specific intersectional demographic rules using the Extended Lift (eLift) equation:
$$eLift = \frac{\text{Confidence}(A \cap B \to C)}{\text{Confidence}(B \to C)}$$
Where a score $\ge 1.5$ is discriminatory.
* **Rule 1 (Female, Senior $\to$ Diabetic):** Max eLift = 0.90
* **Rule 2 (Male, Senior $\to$ Diabetic):** Max eLift = 1.18
* **Rule 3 (Female, Young $\to$ Diabetic):** Max eLift = 0.97
* **Verdict:** All models remained below the 1.5 threshold, making the system statistically **$\alpha$-PROTECTIVE** against targeted intersectional discrimination.

### 3. Bias Mitigation
**Fix applied:** Algorithmic Sample Weighting.
We implemented a weighted training methodology using `sample_weight` to penalize the algorithms for misclassifying historically disadvantaged demographic segments, mathematically adjusting the model's decision boundary for equity.

**Mitigation Results Summary:**
The implementation of sample weights successfully served as a proactive bias-mitigation strategy. The **Weighted Random Forest** experienced a negligible reduction in overall F1-score but achieved a more equitable distribution of error rates (mistreatment) across age groups. This tradeoff is clinically justified to ensure fair diagnostic support for Senior patients.

---

### Project Structure

```text
├── docs/
│   ├── M515A_Project_Report.pdf        # Full written academic report
│   └── M515A_Notebook_Preview.html     # Rendered HTML version of the code
├── notebooks/
│   └── M515A_Ethical_issues_for_AI.ipynb # Main Python code and analysis
└── README.md                           # This file
