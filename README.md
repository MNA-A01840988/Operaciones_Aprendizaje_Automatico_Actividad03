# Employee Attrition Prediction — Elastic Net Logistic Regression

This project trains and evaluates a Logistic Regression model using Elastic Net regularization to predict employee attrition. The workflow supports single model executions with configurable parameters and multiple executions using predefined test configurations.

## Prerequisites

Before running the project, ensure the following tools are installed:

- **Git** — to clone the repository.
- **Python** — to create a virtual environment and run the project.
- **MSYS2** — to provide the required GNU Make environment on Windows.
- **MLflow** — installed through the project's `requirements.txt` file.

### 1. Install MSYS2 and GNU Make

Download and install MSYS2 from the official website:

https://www.msys2.org/

Ensure that `make` is available in your terminal. Verify the installation by executing:

```bash
make --version
```

The expected output should include:

```text
GNU Make 4.4.1
Built for x86_64-pc-cygwin
Copyright (C) 1988-2023 Free Software Foundation, Inc.
License GPLv3+: GNU GPL version 3 or later <https://gnu.org/licenses/gpl.html>
```

The version and build information may vary depending on your installation. If `make` is not recognized, verify your MSYS2 installation and the terminal's `PATH` configuration.

## Installation

### 2. Clone the repository

Download the repository to your local machine:

```bash
git clone https://github.com/MNA-A01840988/Operaciones_Aprendizaje_Automatico_Actividad03.git
cd Operaciones_Aprendizaje_Automatico_Actividad03
```

### 3. Create and activate a Python virtual environment

Create the virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows using Command Prompt:

```bat
.venv\Scripts\activate.bat
```

If you are using PowerShell instead, activate it with:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install project dependencies

With the virtual environment activated, install the required libraries:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Running the project

**Important:** Start the MLflow tracking server before executing any model training or test commands. Otherwise, the execution may fail if the project is configured to connect to the local MLflow server.

Open a terminal in the project directory, activate the virtual environment, and run:

```bash
mlflow ui
```

By default, the MLflow UI is available at:

http://127.0.0.1:5000

Keep this terminal running while executing the commands below. Open a separate terminal in the project directory for model execution, and activate the virtual environment there as well.

### Option 1: Single execution with default parameters

Run the model with the default configuration:

```bash
make train
```

Default parameters:

| Parameter | Default value | Description |
|---|---|---|
| `C` | `1.0` | Inverse of regularization strength. Smaller values indicate stronger regularization. |
| `L1_RATIO` | `0.5` | Elastic Net mixing parameter. A value of 0 corresponds to L2 regularization, while 1 corresponds to L1 regularization. |
| `CLASS_WEIGHT` | `none` | Does not apply class weighting by default. Use `balanced` to automatically adjust weights inversely proportional to class frequencies. |
| `RANDOM_STATE` | `17` | Controls random behavior where applicable to improve reproducibility. |

### Option 2: Single execution with custom parameters

Override the default parameters directly in the Make command:

```bash
make train C=2.0 L1_RATIO=0.2 CLASS_WEIGHT='balanced' RANDOM_STATE=123
```

You can modify one or more parameters depending on the experiment you want to perform.

For more information about the Logistic Regression parameters, refer to the official scikit-learn documentation:

https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

### Option 3: Multiple executions

Run the predefined model tests using the parameter configurations specified in `tests/test_data.py`:

```bash
make test_model
```

This option is intended for executing multiple model runs using the configurations defined in the test file. Review `tests/test_data.py` to inspect or modify the parameter combinations used by the project.

## MLflow Tracking

The project uses MLflow to track model experiments. Depending on the configured workflow, recorded information can include:

- Model parameters.
- Evaluation metrics.
- Model artifacts.
- Individual experiment runs.

After executing the training or test commands, open the MLflow UI:

http://127.0.0.1:5000

Select the relevant experiment and run to review the recorded results and compare executions.