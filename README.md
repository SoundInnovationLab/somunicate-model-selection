# Code Supplement for Submission of TISMIR Research Article

This repository provides the code supplement for the research article
"Expanding MIR into new audio domains: Predicting Perceived Meaning of
Functional Sounds using Unsupervised Feature Extraction and Ensemble
Learning" submitted to "TISMIR - Transactions of the International Society
for Music Information Retrieval".

This repository contains code for performing model selection for a
regression model that predicts perceived semantic expression from the
acoustic topic features of functional sounds. The goal is to identify the
best model configuration (hyperparameters). We explore two learning
paradigms (Deep Neural Networks and Random Forests), different output
configurations (Multioutput and Singleoutput) and the inclusion of
additional input features (meatdata). The process involves hyperparameter
grid search with k-fold cross-validation, evaluation of grid search
results, and retraining/evaluation of the best model.

Due to copyright restrictions, the original dataset used in the publication
cannot be made publicly available. However, a dummy dataset with the same
structure and number of samples as the original is provided in the `data/`
directory to allow users to test and reproduce the code.

# How to Use

## Preqrequisites

Make sure you have
[uv installed](https://docs.astral.sh/uv/getting-started/installation/).

### If using VSCode

Make sure you have the following packages installed.

[Flake8](https://marketplace.visualstudio.com/items?itemName=ms-python.flake8)

[Ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff)

[MyPy](https://marketplace.visualstudio.com/items?itemName=matangover.mypy)

## Running the project

### 1. Sync dependencies and environment

```bash
uv venv   # Creates a virtual environment
source .venv/bin/activate
uv sync   # Installs dependencies
```

### 2. Install autohooks and pre-commit

```bash
uv run pre-commit install --hook-type pre-push --hook-type post-checkout --hook-type pre-commit # Install the pre-commit hooks
uv run pre-commit autoupdate # Update the pre-commit hooks
uv run pre-commit run --all-files # Test if the pre-commit hooks work
```

### 3. Running the project

The model selection process consists of three steps:

1. Performing the extensive hyperparameter gridserach with k-fold
   cross-validation
2. Determining the hyperparameters for the given configuration (averaging
   the performance over the folds)
3. Retraining and evaluating a model with those optimal hyperparameters

For step 1 and step 3 the different learning paradigmns (Deep Neural
Network and Random Forest) have separate scripts. The evaluation works for
both learners.

In the publication all combinations of the following setups were tested:

- Learning Paradigmn: Deep Neural Network / Random Forest
- Output Configuration: Multioutput (all dimensions) / Multioutput (1 model
  per communication level) / Singleoutput (one model per dimension)
- Inclusion of Industry Metadata: Yes / No

For the different output configurations the process has to be repeated for
each model trained, so once for Multioutput (all dimensions), three times
for Multioutput (1 model per communication level) and 19 times for
Singleoutput (one model per dimension).

### 1. Performing the k-fold cross-validation with hyperparameter gridserach

Both scripts are structured the same and share the following arguments:

- --subset (str): The options are 'all', 'status', 'appeal', 'brand
  identity' and 'dimensions'
- --target_index (int): only needed when subset==dimensions. Then you have
  to provide the index of the target dimension (from 0 to 18)
- --include_industry: True or False
- --folds (int): either 3,4,5 or 6
- --log_folder (str): Path to the location where the training results
  should be stored
- --log_subfolder (str): subfolder within the log_folder. We recommend
  indicating the number of cross-validation folds

An example for training a multioutput DNN Regressor using the industry
metadata and doing a 5-fold validation

```bash
uv run python somunicate_model_selection/gridsearch_kfold_dnn.py --subset all --include_industry True --folds 5 --log_folder ./logs/gridsearch_dnn_with_industry --log_subfolder k_5
```

Inside the log_folder a folder called "all" will be created and inside this
the "k_4" will contain a "results.json" file when the gridsearch is done.

### 2. Finding the best hyperparameters

This script will access the "results.json" to determine the best
hyperparameters over all folds of the gridsearch. It needs the following
arguments:

- --log_dir (str): Path to the location from where the the result.json
  should be gathered. All subdirs of the given path are searched for a file
  called "results.json" so you could provide a high level path and all
  gridsearches inside would be evaluated at once.
- --learner (str): Options are "dnn" and "rf". depending on the learning
  paradigmn the hyperparameters are different.

```bash
uv run python somunicate_model_selection/evaluate_gridsearch_results.py --log_dir ./logs/gridsearch_dnn_with_industry/all/k_5 --learner dnn
```

Within the same folder the evaluation generates two files, one storing the
average gridserach results per hyperparameter combination (avg over folds)
called "fold_average_results.json" and one containing the best
hyperparameters called "best_model_hparams.json".

### 3. Retraining and evaluating a model with the best hyperparameters

This last script works similarly to the first one and accesses the best
hyperparams found in the previous script.

It needs the following arguments:

- --hparam_file (str): Path to where the best hyperparameters are stored
- --include_industry: True or False
- --folds (int): either 3,4,5 or 6. This is important for the split sizes
  of the data.
- --log_folder (str): Path to the location where the best model training
  results should be stored
- --n_experiments (int): The best model training can be repeated several
  times if needed.

After the training is done the model is also evaluated on a holdout test
set.

```bash
uv run python somunicate_model_selection/train_best_models_dnn.py --hparam_file ./logs/gridsearch_dnn_with_industry/all/k_5/best_model_hparams.json --include_industry True --n_folds 5 --log_folder ./logs/best_dnn_with_industry
```

Model predictions on the complete dataset ("predictions_0.json"), the test
results ("results.json") and a file storing the indices of the datapoints
for the test set ("test_indices.json") automatically created inside the log
folder.

## Recommendations

### Logging Directory Structure

Since testing all possible model configrations and the whole process is
very complex we recommend following a structure for the result logs like
this (same structure with a different base for the best model trainings):

```
logs/gridsearch/
    └── dnn/
        └── with_industry/
        └── without_industry/
    └── rf/
        └── with_industry/
        └── without_industry/
```

Within those directories a folder for each model (e.g. "all" or
"being_ready") will be created automatically.

### Dataset

Since the dataset is copyrighted and not publicly available, we provided a
dummy dataset that can be used to test the scripts. The dummy dataset is
located in the `´data/` directory. It contains the same structure and
number of samples as the original dataset, but the values are randomly
generated.

To use another dataset, you can replace the path in the following files:

- `somunicate_model_selection/gridsearch_kfold_dnn.py`
- `somunicate_model_selection/gridsearch_kfold_rf.py`
- `somunicate_model_selection/train_best_models_dnn.py`
- `somunicate_model_selection/train_best_models_rf.py`

## Building the project

### Building a wheel

```bash
uv build
```

### With Docker

```bash
docker build .
```

# License

This project is licensed under the MIT License - see the [LICENSE](LICENSE)
file for details.

<!-- ## 2. Install python version

```bash
uv python install 3.13
```

## 3. Create virtual env

```bash
uv venv --python 3.13.0
```

## 4. Pin the required python version to the project

```bash
uv python pin 3.13
```

## 5. Install development dependencies with uv

```bash
uv pip install -r requirements.txt
```

## 6. Activate autohooks

```bash
uv run autohooks activate --mode pythonpath
```

Make sure that your [pre.commit](.git/hooks/pre-commit) file starts with
the following line to ensure the correct python version

```
#!/usr/bin/env -S uv run python
``` -->
