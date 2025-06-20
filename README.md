# Code Supplement for Submission of TISMIR Research Article

This repository provides the code supplement for the research article
**"Predicting Perceived Meaning of Functional Sounds using Unsupervised
Feature Extraction and Ensemble Learning"** submitted by Annika Frommholz
to the
[special collection](https://account.transactions.ismir.net/index.php/up-j-tismir/libraryFiles/downloadPublic/4)
on "Digital Musicology" in
[TISMIR - Transactions of the International Society for Music Information Retrieval](https://transactions.ismir.net/).
The research was conducted in collaboration between the
[Audiocommunication Group](https://www.tu.berlin/en/ak) of Technische
Universität Berlin and the
[Sound Innovation Lab](https://www.soundinnovationlab.com/).

This repository contains code for performing model selection for a
regression model that predicts perceived semantic expression from the
acoustic topic features of functional sounds. The goal is to identify the
best model configuration (hyperparameters).

## Key Features

- **Learning Paradigms**: Deep Neural Networks (DNN) and Random Forests
  (RF)
- **Output Configurations**:
  - Multioutput (all dimensions)
  - Multioutput (1 model per communication level)
  - Singleoutput (1 model per dimension)
- **Additional Input Features**: Inclusion of metadata (e.g., industry
  information)

The process involves:

1. Hyperparameter grid search with k-fold cross-validation.
2. Evaluation of grid search results.
3. Retraining and evaluation of the best model.

> **Note**: Due to copyright restrictions, the original dataset cannot be
> made publicly available. A dummy dataset with the same structure and
> number of samples is provided in the `data/` directory for testing and
> reproducing the code.

______________________________________________________________________

## How to Use

### Prerequisites

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. If using VSCode, ensure the following extensions are installed:
   - [Flake8](https://marketplace.visualstudio.com/items?itemName=ms-python.flake8)
   - [Ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff)
   - [MyPy](https://marketplace.visualstudio.com/items?itemName=matangover.mypy)

### Setting Up the Environment

1. **Sync Dependencies and Environment**:

   ```bash
   uv venv   # Creates a virtual environment
   source .venv/bin/activate
   uv sync   # Installs dependencies
   ```

2. **Install Autohooks and Pre-commit**:

   ```bash
   uv run pre-commit install --hook-type pre-push --hook-type post-checkout --hook-type pre-commit
   uv run pre-commit autoupdate
   uv run pre-commit run --all-files
   ```

______________________________________________________________________

## Running the Project

### Overview of the Model Selection Process

The model selection process consists of three steps:

1. **Performing k-fold cross-validation with hyperparameter grid search**.
2. **Determining the best hyperparameters** by averaging performance over
   folds.
3. **Retraining and evaluating the model** with the optimal
   hyperparameters.

Each step is explained below.

______________________________________________________________________

### Step 1: Performing k-fold Cross-validation with Hyperparameter Grid Search

#### Arguments

- `--subset` (str): Options are `all`, `status`, `appeal`,
  `brand identity`, and `dimensions`.
- `--target_index` (int): Required only when `subset=dimensions`. Provide
  the index of the target dimension (0 to 18).
- `--include_industry` (bool): `True` or `False`.
- `--folds` (int): Number of folds (3, 4, 5, or 6).
- `--log_folder` (str): Path to store training results.
- `--log_subfolder` (str): Subfolder within `log_folder` (e.g., indicate
  the number of folds).

#### Example Command

Train a multioutput DNN Regressor using industry metadata with 5-fold
validation:

```bash
uv run python somunicate_model_selection/gridsearch_kfold_dnn.py --subset all --include_industry True --folds 5 --log_folder ./logs/gridsearch_dnn_with_industry --log_subfolder k_5
```

> **Output**: Inside the `log_folder`, a folder named `all` will be
> created. The subfolder `k_5` will contain a `results.json` file upon
> completion.

______________________________________________________________________

### Step 2: Finding the Best Hyperparameters

#### Arguments

- `--log_dir` (str): Path to the directory containing `results.json`. All
  subdirectories are searched for this file.
- `--learner` (str): Options are `dnn` or `rf` (depending on the learning
  paradigm).

#### Example Command

Evaluate grid search results for a DNN:

```bash
uv run python somunicate_model_selection/evaluate_gridsearch_results.py --log_dir ./logs/gridsearch_dnn_with_industry/all/k_5 --learner dnn
```

> **Output**: Two files are generated:
>
> - `fold_average_results.json`: Average grid search results per
>   hyperparameter combination.
> - `best_model_hparams.json`: Best hyperparameters.

______________________________________________________________________

### Step 3: Retraining and Evaluating the Best Model

#### Arguments

- `--hparam_file` (str): Path to the file containing the best
  hyperparameters.
- `--include_industry` (bool): `True` or `False`.
- `--folds` (int): Number of folds (3, 4, 5, or 6).
- `--log_folder` (str): Path to store training results.
- `--n_experiments` (int): Number of times to repeat the training.

#### Example Command

Retrain and evaluate the best DNN model:

```bash
uv run python somunicate_model_selection/train_best_models_dnn.py --hparam_file ./logs/gridsearch_dnn_with_industry/all/k_5/best_model_hparams.json --include_industry True --n_folds 5 --log_folder ./logs/best_dnn_with_industry
```

> **Output**: The following files are created in the log folder:
>
> - `predictions_0.json`: Model predictions on the complete dataset.
> - `results.json`: Test results.
> - `test_indices.json`: Indices of the test set datapoints.

______________________________________________________________________

## Recommendations

### Logging Directory Structure

To manage the complexity of testing all model configurations, we recommend
the following structure for result logs:

```
logs/gridsearch/
    └── dnn/
        └── with_industry/
        └── without_industry/
    └── rf/
        └── with_industry/
        └── without_industry/
```

Within these directories, a folder for each model (e.g., `all` or
`being_ready`) will be created automatically.

______________________________________________________________________

### Dataset

A dummy dataset is provided in the `data/` directory. It has the same
structure and number of samples as the original dataset but with randomly
generated values.

To use another dataset, update the dataset path in the following files:

- `somunicate_model_selection/gridsearch_kfold_dnn.py`
- `somunicate_model_selection/gridsearch_kfold_rf.py`
- `somunicate_model_selection/train_best_models_dnn.py`
- `somunicate_model_selection/train_best_models_rf.py`

______________________________________________________________________

## Building the Project

### Building a Wheel

```bash
uv build
```

### Using Docker

```bash
docker build .
```

______________________________________________________________________

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE)
file for details.
