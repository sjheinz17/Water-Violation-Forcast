# Water Quality Violation Forecasting

This repository contains the code and analysis for a machine learning project that proactively predicts drinking water quality violations in California. The goal of this project is to help regulators and water system operators identify high-risk systems *before violations are detected*, enabling earlier intervention and more efficient allocation of public health resources.

## Project Overview

We develop a **time-stacked Random Forest classification model** that uses historical contaminant measurements, prior violation history, and system-level characteristics to forecast the likelihood of future Maximum Contaminant Level (MCL) violations. The model is trained on multiple year-to-year transitions and evaluated on a fully held-out future year, demonstrating strong generalization and real-world applicability.

## Data Sources

All data used in this project comes from publicly available datasets maintained by the **California State Water Resources Control Board – Drinking Water Program**.

Specifically, we use datasets made available through the **Drinking Water Open Environmental Data Project (OEDP)**:

> https://data.cnra.ca.gov/dataset/drinking-water-open-environmental-data-project-march-23-2023-workshop

These datasets were developed for the March 23, 2023 OEDP workshop, whose purpose was to engage community members, researchers, and policymakers in understanding and improving access to drinking water data in California. The workshop aimed to support transparency, data-driven policy advocacy, and community-based decision-making.

### Years Used
- **Training data:** 2020–2024  
- **Held-out test data:** 2025  

Only data from **2020–2024** is used for model training and hyperparameter tuning. The 2025 data is strictly reserved for final evaluation.

### Dataset Contents
The datasets include:
- Drinking water test results for regulated contaminants (e.g., arsenic, nitrates)
- Maximum Contaminant Level (MCL) exceedances
- Water system metadata (population served, service connections, system classification)
- Facility and source water characteristics
- Geographic information (county-level identifiers)

All datasets are publicly available and contain no personally identifiable information.

## Methodology Summary

- **Feature Engineering:** Lagged contaminant statistics (mean, max, standard deviation, prior failures) computed from the previous year
- **Model:** Random Forest classifier
- **Training Strategy:** Time-stacked year-to-year transitions (2020→2021 through 2023→2024)
- **Evaluation:** Fully held-out 2024→2025 prediction
- **Optimization:** Cross-validated randomized hyperparameter search using ROC-AUC

## Intended Use

This repository is intended for:
- Academic research and coursework
- Policy analysis and regulatory decision support
- Community and advocacy organizations interested in proactive water safety analytics

The model is designed as a **decision-support tool**, not a replacement for regulatory testing or enforcement.

## License & Attribution

Data is provided by the California State Water Resources Control Board via the CNRA Open Data Portal. Users of this repository should cite the original data source appropriately when reusing or extending this work.

---

