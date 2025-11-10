# Predicting 30-Day Hospital Readmissions with Responsible AI

Hey PLP community! 👋 Our team just wrapped the “Understanding the AI Development Workflow” assignment for the AI for Software Engineering course. We built a full pipeline for predicting which patients are likely to be readmitted within 30 days of discharge—and we focused on doing it responsibly.

## Why It Matters
Hospitals face financial penalties and, more importantly, patient safety risks when readmissions spike. Our goal was to give clinicians an explainable risk score so they can intervene early and support vulnerable patients.

## What’s Inside the Project
- **Problem Framing:** Objectives and KPIs that align with hospital stakeholders.
- **Data Strategy:** EHRs, insurance claims, and social determinants data, all managed with HIPAA-grade governance.
- **Preprocessing:** Imputation, scaling, categorical encoding, TF-IDF embeddings, and SMOTE for class imbalance.
- **Modeling:** LightGBM with hyperparameter tuning, SHAP explanations, and robust evaluation (AUROC + PR-AUC).
- **Deployment Plan:** Containerized inference service, FHIR integration, automated monitoring for concept drift.
- **Ethics Focus:** Bias audits, fairness-aware reweighting, and clinician feedback loops.

## Key Takeaways
1. **Workflow Discipline:** Following CRISP-DM keeps the project aligned from problem definition through monitoring.
2. **Explainability Matters:** Clinicians trust models when they can see the “why” behind risk scores.
3. **Monitoring is Essential:** Concept drift in healthcare is real; we use PSI and periodic retraining triggers.

## Explore the Repository
All code, docs, and synthetic data live on GitHub: _add repo link after publishing_.

## Next Steps
We plan to extend the pipeline with a feature store, real-world usability testing, and expanded fairness analysis.

Curious to learn more or collaborate? Drop a comment or reach out! 🚀

