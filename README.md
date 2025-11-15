# Iris NN Project

This workspace contains a corrected `train_iris.py` script that trains a small Keras model on the Iris dataset, performs a simple grid search, and saves the best model and logs.

## Quick steps to run (Windows PowerShell)

1. Open a PowerShell terminal in this workspace (or in VS Code terminal).

2. Create a venv and activate it:

```powershell
python -m venv .venv
# If activation is blocked, you can allow it for this session:
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

4. Run the training script:

```powershell
python train_iris.py
```

5. Files written by script:
- `iris_nn_project/models/` - saved models and scaler params
- `iris_nn_project/logs/` - training logs and grid search report
- `iris_nn_project/plots/` - (placeholder for saved plots)

## VS Code tips
- Install the `Python` extension (ms-python.python) and `Pylance` for better editor support.
- After creating the venv, open the Command Palette (Ctrl+Shift+P) and run `Python: Select Interpreter`, choose the `.venv` interpreter.
- Use the terminal in VS Code to run the activation commands above before running the script.

## Common problems & fixes
- "Activate.ps1 cannot be loaded" — run the `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` command first.
- `tensorflow` installation on Windows may require a 64-bit Python and appropriate pip version; if you hit issues, see TensorFlow install docs: https://www.tensorflow.org/install

If you want, I can:
- Run the script here (if you want me to execute in this environment) — tell me to proceed.
- Create a `streamlit` frontend scaffold.
## Quick Copy/Paste Commands (PowerShell)

Use these commands from the workspace root (`C:\workspace_iris`). Run the setup commands once, then run the training and frontend commands as needed.

Setup (one-time):
```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
.venv\Scripts\python.exe -m pip install -r .\requirements.txt
```

Train (when you want to retrain):
```powershell
.venv\Scripts\python.exe .\train_iris.py
```

Start Streamlit frontend (foreground):
```powershell
$env:STREAMLIT_CLIENT_EMAIL_REQUIRED='false'
.venv\Scripts\python.exe -m streamlit run .\streamlit_app.py
```

Start Streamlit frontend (background, terminal stays free):
```powershell
$env:STREAMLIT_CLIENT_EMAIL_REQUIRED='false'
Start-Process -FilePath .\.venv\Scripts\python.exe -ArgumentList '-m','streamlit','run','./streamlit_app.py'
```

Single-line (not recommended every time — avoids reinstalling deps repeatedly):
```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt; .venv\Scripts\python.exe train_iris.py; $env:STREAMLIT_CLIENT_EMAIL_REQUIRED='false'; .venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

