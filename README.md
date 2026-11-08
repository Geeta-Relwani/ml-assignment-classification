# Predicting Purchase Intent from Online Shopping Sessions

Machine Learning — Assignment 2 · Five classification models with a Streamlit front end.

---

## a. Problem statement

An e-commerce site records what every visitor does during a session: how many
pages of each type they open, how long they spend on them, the bounce and exit
rates of those pages, the month, the traffic source, and whether they are a
returning visitor. Most of those sessions end with nothing.

The task is to predict, from that session behaviour alone, whether the session
ends in a purchase — the `Revenue` column. This is a **binary classification**
problem, and an imbalanced one: only about 15% of sessions convert.

Getting this right has a direct use. A model that can flag a live session as
likely-to-convert lets the site intervene while the visitor is still on the page
— a discount prompt, a free-shipping banner, a support chat offer. Because a
missed buyer is a lost sale while a false alarm costs only a wasted impression,
**recall matters more than precision here**, and that asymmetry drives the model
comparison below.

## b. Dataset description

**UCI Online Shoppers Purchasing Intention Dataset** ([ID 468](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)),
downloaded by `src/download_data.py`.

| | |
|---|---|
| Instances | 12,330 sessions (12,205 after dropping 125 exact duplicates) |
| Features | 17 |
| Target | `Revenue` — True if the session ended in a transaction |
| Class balance | 1,908 purchases vs 10,422 non-purchases — **15.5% positive** |
| Missing values | None |

Each row is one session, and the data was collected so that each belongs to a
different user over a one-year period.

**Numeric features (10)** — scaled with `StandardScaler`:

| Feature | Meaning |
|---|---|
| `Administrative`, `Informational`, `ProductRelated` | Number of pages of each type visited |
| `Administrative_Duration`, `Informational_Duration`, `ProductRelated_Duration` | Seconds spent on each page type |
| `BounceRates` | Average bounce rate of the pages visited |
| `ExitRates` | Average exit rate of the pages visited |
| `PageValues` | Average value of the pages visited before completing a transaction |
| `SpecialDay` | Closeness of the session to a special day (e.g. Mother's Day) |

**Categorical features (7)** — one-hot encoded:

| Feature | Levels |
|---|---|
| `Month` | 10 |
| `OperatingSystems` | 8 |
| `Browser` | 13 |
| `Region` | 9 |
| `TrafficType` | 20 |
| `VisitorType` | 3 (New, Returning, Other) |
| `Weekend` | 2 |

`OperatingSystems`, `Browser`, `Region` and `TrafficType` arrive as integers but
are **nominal ID codes, not quantities** — browser 7 is not "more" than browser 3
— so they are one-hot encoded rather than scaled. Treating them as numeric would
invent an ordering that does not exist.

### Preparation

- 125 exact duplicate rows dropped **before** splitting, so identical rows cannot
  land on both sides and leak training data into the evaluation.
- Stratified 80/20 split (`random_state=42`): 9,764 training rows, 2,441 held out.
- The held-out 20% is written verbatim to `test_data.csv` — this is the file to
  upload in the Streamlit app, and no model has ever seen it.

## c. GitHub repository link

<https://github.com/Geeta-Relwani/ml-assignment-classification>

**Live Streamlit app:** _<!-- paste the Streamlit Community Cloud URL here after deploying -->_

### Repository contents

```
ml-assignment-classification/
├── app.py                              Streamlit application
├── requirements.txt                    Dependencies (sklearn/numpy/scipy pinned)
├── README.md                           This file
├── test_data.csv                       Held-out 20% (2,441 rows) for the app
├── .streamlit/config.toml              App theme
├── data/
│   └── online_shoppers_intention.csv   Raw dataset from UCI
├── src/
│   ├── download_data.py                Fetch the dataset
│   ├── preprocessing.py                Feature schema + shared transformer
│   ├── evaluate.py                     The six metrics
│   ├── train_models.py                 Train, tune and save the five models
│   └── export_test_data.py             Write test_data.csv
├── notebooks/
│   └── model_development.ipynb         EDA and the full training walkthrough
└── model/
    ├── logistic_regression.joblib      Saved pipelines, one per model
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    └── metrics.json                    Metrics recorded at training time
```

Each `.joblib` holds a **complete `Pipeline`** — preprocessor plus fitted
estimator — not a bare model. That is what lets the app score a raw uploaded CSV:
the saved object already knows how to scale and encode its input, so the app
cannot accidentally apply a different transform from the one the model was
fitted on.

### Reproducing the results

```bash
pip install -r requirements.txt
python src/download_data.py
python src/train_models.py
python src/export_test_data.py
streamlit run app.py
```

## d. Models used

All five models are trained on the same dataset and the same 80/20 split.

Hyperparameters are chosen by **5-fold stratified cross-validation on the
training split only** — the test set is touched once, at the end, and never
influences model selection. The search optimises **F1 rather than accuracy**,
because at a 15% positive rate a model that predicts "no purchase" everywhere
already scores 85% accuracy while being useless.

`class_weight="balanced"` is set wherever the estimator supports it. Gaussian
Naive Bayes and KNN have no equivalent parameter, which turns out to matter.

| Model | Settings chosen by cross-validation |
|---|---|
| Logistic Regression | `C=0.01`, `class_weight="balanced"`, `max_iter=1000` |
| Decision Tree | `max_depth=4`, `min_samples_leaf=10`, `class_weight="balanced"` |
| KNN | `n_neighbors=5`, `weights="uniform"` |
| Naive Bayes | `GaussianNB(var_smoothing=0.1)` |
| Support Vector Machine | `SVC(kernel="rbf", C=1.0, class_weight="balanced")` |

### Comparison table

Measured on the 2,441 held-out sessions in `test_data.csv`.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8587 | 0.9099 | 0.5325 | 0.7932 | 0.6372 | 0.5706 |
| Decision Tree | 0.8480 | **0.9242** | 0.5085 | **0.8586** | 0.6388 | **0.5807** |
| KNN | **0.8734** | 0.7958 | **0.6622** | 0.3901 | 0.4909 | 0.4435 |
| Naive Bayes | 0.8161 | 0.8418 | 0.4419 | 0.6675 | 0.5318 | 0.4371 |
| Support Vector Machine | 0.8689 | 0.9095 | 0.5589 | 0.7696 | **0.6476** | 0.5805 |

Best value in each column is in bold.

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | A strong, well-behaved baseline. AUC of 0.910 means a linear boundary ranks sessions nearly as well as anything else here — most of the signal is close to linearly separable once `PageValues` is in the model. CV chose heavy regularisation (`C=0.01`), unsurprising given the wide one-hot matrix. |
| **Decision Tree** | Best AUC (0.924) and by far the best recall (0.859) — it catches roughly six of every seven real purchases. CV chose a shallow tree (`max_depth=4`), and the shallowness is the point: it splits on the few genuinely strong features and stops before memorising noise. The cost is precision: just over half its positive calls are correct. |
| **KNN** | Highest raw accuracy (0.873) and the worst AUC (0.796) — the most instructive result in the table. It buys that accuracy by being conservative: recall of 0.390 means it misses nearly two thirds of all purchases. In a ~75-dimensional one-hot space neighbourhoods are sparse, and a 15% minority class is simply outvoted by its neighbours. It also has no `class_weight` to correct this. |
| **Naive Bayes** | Weakest overall (MCC 0.437). Its feature-independence assumption is badly violated: `ProductRelated` and `ProductRelated_Duration` are almost the same variable, as are `BounceRates` and `ExitRates`. It also needed heavy variance smoothing (`var_smoothing=0.1`) to work at all — one-hot columns for rare browser and traffic codes have near-zero within-class variance, and at the library default of 1e-9 the Gaussian likelihood overflows and the model predicts "purchase" almost everywhere, scoring 28% accuracy. |
| **Support Vector Machine** | Best F1 (0.648) and the best-balanced model of the five. The RBF kernel gains a little over logistic regression on precision at comparable recall, suggesting a mildly non-linear boundary. It is also the slowest to train, and the only model whose AUC comes from `decision_function` rather than predicted probabilities. |
| **Overall, winner for this dataset?** | **Decision Tree**, narrowly — best AUC, best recall, and an MCC (0.5807) level with the SVM's (0.5805). Recall is what matters commercially here, and the tree is also the only model a marketing team could read directly. **The SVM is the better pick if precision matters more**; on MCC the two are statistically indistinguishable, and a different random seed could reorder them. |

### The accuracy trap

KNN has the highest accuracy in the table and is the worst model in it. On a
15%-positive dataset, accuracy and the other five metrics disagree — and the
other five are right. This is exactly why the assignment asks for AUC and MCC:
MCC accounts for all four cells of the confusion matrix and drops KNN to 0.44
where accuracy flattered it at 0.87.

## Streamlit application

The app loads the saved pipelines and scores whatever test CSV it is given. It
**never retrains** — every number it shows is a genuine held-out result.

| Required feature | Where it is |
|---|---|
| Dataset upload option (CSV) | Sidebar — "Upload my own CSV", or score the bundled `test_data.csv` |
| Model selection dropdown | Sidebar — "Inspect a single model" |
| Display of evaluation metrics | **Single model** tab (all six) and **All five compared** tab (full table) |
| Confusion matrix / classification report | **Single model** tab — both |

Two extras beyond the requirement: an **ROC curves** tab overlaying all five
models on the uploaded data, and a **Training reference** tab showing the metrics
and chosen hyperparameters recorded at training time, so the app's live numbers
can be checked against the run that produced the models.

An uploaded CSV must contain all 17 feature columns plus `Revenue`. The target is
accepted as `True`/`False`, `TRUE`/`FALSE` or `1`/`0`, and unseen categorical
codes are tolerated rather than rejected, so a file that has been round-tripped
through Excel still scores.
