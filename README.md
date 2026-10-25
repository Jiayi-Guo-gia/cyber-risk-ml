# Entity-Specific Cyber Risk Assessment Using Insurtech-Empowered Risk Factors

Code for **"Entity-Specific Cyber Risk Assessment Using Insurtech-Empowered Risk Factors"**
Jiayi Guo, Linfeng Zhang, Zhiyu Quan. *Variance*, Vol. 19, 2026. <https://doi.org/10.66573/001c.155715>

Cyber incident records are scarce and organizations are reluctant to disclose incidents. This study enriches
publicly collected cyber incident data with proprietary InsurTech entity-level features and builds machine learning
models for organization-level cyber risk assessment. For each firm-year, we model

- **incident occurrence** as a multi-label classification problem, and
- **incident frequency** as a multi-output regression problem,

over five incident types (`Data Breach`, `Privacy Violation`, `Extortion/Fraud`, `IT Error`, `Other`), and examine
which entity-specific features drive the predictions and whether incident types are dependent at the entity level.

> **The data are not publicly available.** The entity-level features come from a proprietary InsurTech platform
> and cannot be redistributed, and the merged firm-year dataset is derived from them. This repository therefore
> contains **code only**, together with tuned hyper-parameters and aggregate results (feature-importance tables and
> figures). No raw or processed data, and no row-level model outputs, are included.

## Data (not included)

| Source | Content |
|---|---|
| Cyber incident data (public domain) | 66,215 incident records (2000–2018 retained), assembled from public incident databases and NLP-extracted news reports; incident types follow the Kesan & Zhang (2021) taxonomy; at most one incident of the same type per organization per day |
| InsurTech data (proprietary) | 527 entity-level features for 31,782 companies: business identity (geography, enterprise structure, industry classification, licensing, disclosed risk characteristics) and operational signals (operating status, open hours, shareholding, customer reviews, firmographics) |
| Merged dataset | 39,636 firm-year observations, joined on `COMPANY_ID`; `d = 528` features (527 InsurTech features + `ACCIDENT_YEAR`), `q = 5` incident types |

## Modeling framework

Base learners are Random Forest and LightGBM (scikit-learn / LightGBM implementations). Label-specific decision
thresholds (classification), the label order (chain models) and the hyper-parameters are searched jointly with Optuna
(TPE sampler, 100 trials, 5-fold cross-validation, seed 42).

| Task | Method | Code prefix | Implementation |
|---|---|---|---|
| Multi-label classification (occurrence) | Binary Relevance (BR) | `MOC` | `MultiOutputClassifier` |
| | Classifier Chain (CC) | `CC` | `ClassifierChain` |
| | Multi-label Classification Trees (MCTs) | `MLRF` | single Random Forest over all labels |
| Multi-output regression (frequency) | Multi-output Regressor (MOR) | `MOR` | `MultiOutputRegressor` |
| | Regressor Chain (RC) | `RC` | `RegressorChain` |
| | Multi-output Regression Trees (MRTs) | `MORF` | multi-output `RandomForestRegressor` |

**Evaluation metrics.** Classification: Weighted-F1, Macro-F1, Micro-F1, Sample-F1, Jaccard index, Hamming loss.
Regression: average MSE (aMSE), average RMSE (aRMSE), average correlation coefficient (aCC), average relative RMSE
(aRRMSE), and global Euclidean distance (EU_DIST).

**Feature importance.** Impurity-based (gain and split for LightGBM; Gini and entropy for classification forests;
MSE-based impurity for regression forests), permutation importance (ROC-AUC for classification, MSE for regression),
and TreeSHAP: 20 model–method combinations for classification and 15 for regression.

**Dependence structure.** The residual-analysis notebooks compute partial correlations between the residuals of
the different incident types to assess conditional dependence at the entity level.

Script names follow `<METHOD>_<LEARNER>_<IMPORTANCE>.py`, e.g. `CC_RF_shap.py` = Classifier Chain + Random Forest +
TreeSHAP; models are named `LGBM_MOC`, `RF_CC`, `MLRF`, `LGBM_MOR`, `RF_RC`, `MORF`, etc.

## Repository structure

```
.
├── classification/  # multi-label classification (incident occurrence)
│   ├── baselines/  # untuned reference models
│   │   ├── CC_lgbm_original.py
│   │   ├── CC_rf_original.py
│   │   └── MOC_original.py
│   ├── data/  # place data here (contents git-ignored)
│   │   └── README.md
│   ├── feature_importance/  # one script per (method, learner, importance technique)
│   │   ├── CC_LGBM_feature_gain.py
│   │   ├── CC_LGBM_feature_split.py
│   │   ├── CC_LGBM_permutation.py
│   │   ├── CC_LGBM_shap.py
│   │   ├── CC_RF_feature_entropy.py
│   │   ├── CC_RF_feature_gini.py
│   │   ├── CC_RF_permutation.py
│   │   ├── CC_RF_shap.py
│   │   ├── MLRF.py
│   │   ├── MLRF_feature_entropy.py
│   │   ├── MLRF_feature_gini.py
│   │   ├── MLRF_permutation.py
│   │   ├── MLRF_shap.py
│   │   ├── MOC_LGBM_feature_gain.py
│   │   ├── MOC_LGBM_feature_split.py
│   │   ├── MOC_LGBM_permutation.py
│   │   ├── MOC_LGBM_shap.py
│   │   ├── MOC_RF_feature_entropy.py
│   │   ├── MOC_RF_feature_gini.py
│   │   ├── MOC_RF_permutation.py
│   │   └── MOC_RF_shap.py
│   ├── notebooks/  # model performance, importance heatmaps, residual analysis
│   │   ├── feature_importance_heatmaps.ipynb
│   │   ├── model_performance.ipynb
│   │   └── residual_analysis.ipynb
│   ├── results/  # tuned hyper-parameters and aggregate results
│   │   ├── CC/  # Classifier Chain: hyper-parameters, label order, thresholds (JSON)
│   │   │   ├── LGBM_CC_best_chain_order.json
│   │   │   ├── LGBM_CC_best_model_params.json
│   │   │   ├── LGBM_CC_best_thresholds.json
│   │   │   ├── RF_CC_best_chain_order.json
│   │   │   ├── RF_CC_best_model_params.json
│   │   │   └── RF_CC_best_thresholds.json
│   │   ├── feature/  # top-20 feature-importance tables (CSV) and plots (PNG)
│   │   │   ├── LGBM_CC_permutation_importance.csv
│   │   │   ├── LGBM_CC_shap_importance.csv
│   │   │   ├── LGBM_CC_top20_features_gain.csv
│   │   │   ├── LGBM_CC_top20_features_split.csv
│   │   │   ├── LGBM_MOC_permutation_importance.csv
│   │   │   ├── LGBM_MOC_shap_importance.csv
│   │   │   ├── LGBM_MOC_top20_features_gain.csv
│   │   │   ├── LGBM_MOC_top20_features_split.csv
│   │   │   ├── MLRF_top20_features_entropy.csv
│   │   │   ├── MLRF_top20_features_gini.csv
│   │   │   ├── MLRF_top20_permutation_importance.csv
│   │   │   ├── MLRF_top20_shap_importance.csv
│   │   │   ├── RF_CC_permutation_importance.csv
│   │   │   ├── RF_CC_shap_importance.csv
│   │   │   ├── RF_CC_top20_features_entropy.csv
│   │   │   ├── RF_CC_top20_features_gini.csv
│   │   │   ├── RF_MOC_permutation_importance.csv
│   │   │   ├── RF_MOC_shap_importance.csv
│   │   │   ├── RF_MOC_top20_features_entropy.csv
│   │   │   ├── RF_MOC_top20_features_gini.csv
│   │   │   └── *.png  # 29 figures
│   │   ├── figures/  # performance heatmaps
│   │   │   └── *.png  # 2 figures
│   │   └── MOC/  # Binary Relevance: hyper-parameters, thresholds (JSON)
│   │       ├── LGBM_MOC_best_model_params.json
│   │       ├── LGBM_MOC_best_thresholds.json
│   │       ├── RF_best_model_params.json
│   │       ├── RF_best_thresholds.json
│   │       ├── RF_MOC_best_model_params.json
│   │       └── RF_MOC_best_thresholds.json
│   └── tuning/  # Optuna search (hyper-parameters, thresholds, label order) + SLURM scripts
│       ├── CC_LGBM_ORDER.py
│       ├── CC_RF_ORDER.py
│       ├── MLRF_TUNE.py
│       ├── MOC_LGBM_TUNE.py
│       ├── MOC_RF_TUNE.py
│       ├── run_CC_LGBM_ORDER.sh
│       ├── run_CC_RF_ORDER.sh
│       ├── run_MLRF_TUNE.sh
│       ├── run_MOC_LGBM_TUNE.sh
│       └── run_MOC_RF_TUNE.sh
├── preprocessing/  # data preparation
│   ├── raw_cleaning/  # raw JSON -> CSV, cleaning and flattening of the entity-level records
│   │   ├── data_clean_cyber.py
│   │   ├── data_flatten_cyber.py
│   │   └── data_loader.py
│   ├── data_processing.py
│   ├── split_classification.py
│   └── split_regression.py
├── regression/  # multi-output regression (incident frequency)
│   ├── baselines/  # untuned reference models (RF, kernel ridge, SVR, multi-output SVR)
│   │   ├── MOR_KRR_original.py
│   │   ├── MOR_SVR_original.py
│   │   ├── MORF_original.py
│   │   └── MSVR_original.py
│   ├── data/  # place data here (contents git-ignored)
│   │   └── README.md
│   ├── feature_importance/  # one script per (method, learner, importance technique)
│   │   ├── MOR_LGBM_feature_gain.py
│   │   ├── MOR_LGBM_feature_split.py
│   │   ├── MOR_LGBM_permutation.py
│   │   ├── MOR_LGBM_shap.py
│   │   ├── MOR_RF_feature_impurity.py
│   │   ├── MOR_RF_permutation.py
│   │   ├── MOR_RF_shap.py
│   │   ├── MORF_feature_impurity.py
│   │   ├── MORF_permutation.py
│   │   ├── MORF_shap.py
│   │   ├── RC_LGBM_feature_gain.py
│   │   ├── RC_LGBM_feature_split.py
│   │   ├── RC_LGBM_permutation.py
│   │   ├── RC_LGBM_shap.py
│   │   ├── RC_RF_feature_impurity.py
│   │   ├── RC_RF_permutation.py
│   │   └── RC_RF_shap.py
│   ├── notebooks/  # model performance, importance heatmaps, residual analysis, RF tuning
│   │   ├── feature_importance_heatmaps.ipynb
│   │   ├── model_performance.ipynb
│   │   ├── residual_analysis.ipynb
│   │   └── rf_hyperparameter_tuning.ipynb
│   ├── results/  # tuned hyper-parameters and aggregate results
│   │   ├── feature/  # top-20 feature-importance tables (CSV) and plots (PNG)
│   │   │   ├── gain/  # LightGBM gain importance
│   │   │   │   ├── LGBM_MOR_top20_features_gain.csv
│   │   │   │   ├── LGBM_RC_top20_features_gain.csv
│   │   │   │   └── *.png  # 2 figures
│   │   │   ├── LGBM_MOR_top20_features_gain.csv
│   │   │   ├── LGBM_MOR_top20_features_permutation.csv
│   │   │   ├── LGBM_MOR_top20_features_shap.csv
│   │   │   ├── LGBM_RC_top20_features_gain.csv
│   │   │   ├── LGBM_RC_top20_features_permutation.csv
│   │   │   ├── LGBM_RC_top20_features_shap.csv
│   │   │   ├── MORF_top20_features_impurity.csv
│   │   │   ├── MORF_top20_features_permutation.csv
│   │   │   ├── MORF_top20_features_shap.csv
│   │   │   ├── RF_MOR_top20_features_impurity.csv
│   │   │   ├── RF_MOR_top20_features_permutation.csv
│   │   │   ├── RF_MOR_top20_features_shap.csv
│   │   │   ├── RF_RC_top20_features_impurity.csv
│   │   │   ├── RF_RC_top20_features_permutation.csv
│   │   │   ├── RF_RC_top20_features_shap.csv
│   │   │   └── *.png  # 24 figures
│   │   ├── MOR/  # Multi-output Regressor: hyper-parameters (JSON)
│   │   │   ├── LGBM_MOR_best_params.json
│   │   │   ├── RF_best_params.json
│   │   │   └── RF_MOR_best_params.json
│   │   └── RC/  # Regressor Chain: hyper-parameters, chain order (JSON)
│   │       ├── LGBM_RC_best_chain_order.json
│   │       ├── LGBM_RC_best_params.json
│   │       ├── RF_RC_best_chain_order.json
│   │       └── RF_RC_best_params.json
│   └── tuning/  # Optuna search (hyper-parameters, chain order) + SLURM scripts
│       ├── MOR_LGBM_optimized.py
│       ├── MOR_RF_optimized.py
│       ├── MORF_optimized.py
│       ├── RC_LGBM_ORDER.py
│       ├── RC_RF_ORDER.py
│       ├── run_MOR_LGBM_TUNE.sh
│       ├── run_MOR_RF_TUNE.sh
│       ├── run_RC_LGBM_ORDER.sh
│       ├── run_RC_RF_ORDER.sh
│       └── run_RF_TUNE.sh
├── CITATION.cff
├── LICENSE
├── README.md
└── requirements.txt
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the results

Reproduction requires access to the data described above. Without it, the code can be read and the aggregate
results in `*/results/` inspected, but the pipeline cannot be run.

**1. Preprocessing** (run from the repository root)

`data_processing.py` expects two input files in `classification/data/`, `label_w_year.csv` (incident labels) and
`cyber_cleaned.csv` (entity-level features), derived from the raw data with `preprocessing/raw_cleaning/` plus
manual cleaning steps.

```bash
python preprocessing/data_processing.py      # -> classification/data/label_cyber_year.csv
python preprocessing/split_classification.py # -> classification/data/{X,Y}_{train,test}.csv
cp classification/data/label_cyber_year.csv regression/data/
python preprocessing/split_regression.py     # -> regression/data/{X,Y}_{train,test}.csv
```

**2. Hyper-parameter tuning** (run from the *task* directory)

```bash
cd classification
python tuning/MOC_LGBM_TUNE.py     # also: MOC_RF_TUNE, CC_LGBM_ORDER, CC_RF_ORDER (label order is searched too), MLRF_TUNE
sbatch tuning/run_MOC_LGBM_TUNE.sh # SLURM wrapper (edit the partition / module / conda env for your cluster)

cd ../regression
python tuning/MOR_LGBM_optimized.py  # also: MOR_RF_optimized, MORF_optimized, RC_LGBM_ORDER, RC_RF_ORDER
```

Each script writes best parameters, metrics, predictions and the Optuna study database under `results/`.
The tuned parameters used in the paper are provided in `results/{CC,MOC}` (classification) and `results/{MOR,RC}`
(regression); the regression tuning scripts write to `results/{LGBM,RF,ORDER}`, so copy their JSON files into
`results/{MOR,RC}` (names as in the provided files) before running the feature-importance scripts on a re-tuned model.

**3. Feature importance** (run from the *task* directory; paths in the scripts are relative to it)

```bash
cd classification
python feature_importance/CC_RF_shap.py            # writes results/feature/RF_CC_shap_importance.csv + plot
python feature_importance/MOC_LGBM_permutation.py
```

Available techniques per learner — LightGBM: `feature_gain`, `feature_split`, `shap`, `permutation`;
Random Forest: `feature_gini`/`feature_entropy` (classification) or `feature_impurity` (regression), `shap`, `permutation`.
Hyper-parameters are read from `results/<METHOD>/*.json`.

**4. Figures and tables** — open the notebooks in `*/notebooks/` with Jupyter started from the task directory
(`cd classification && jupyter lab`), since they read `results/` and `data/` via relative paths.

## Citation

If you use this code, please cite:

```bibtex
@article{guo2026entity,
  title   = {Entity-Specific Cyber Risk Assessment Using Insurtech-Empowered Risk Factors},
  author  = {Guo, Jiayi and Zhang, Linfeng and Quan, Zhiyu},
  journal = {Variance},
  volume  = {19},
  year    = {2026},
  doi     = {10.66573/001c.155715}
}
```

See also [`CITATION.cff`](CITATION.cff).

## License

MIT — see [`LICENSE`](LICENSE). The license covers the code only, not the data.
