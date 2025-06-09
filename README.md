# python-uv-template

Please rename the top part to match your repository. Also make sure that
[pyproject.toml](./pyproject.toml) contains the correct repository name.

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

Make sure you edit this once you have a specific project that you are using
(e.g. [FastAPI](https://fastapi.tiangolo.com/)).

```bash
uv run app/main.py
```

## Building the project

### Building a wheel

```bash
uv build
```

### With Docker

```bash
docker build .
```

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
