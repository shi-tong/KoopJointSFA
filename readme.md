# 🏭 Industrial Process Monitoring Framework (CSTR & TE)
This is for the paper we submitted "Joint Koopman-Guided Slow Feature Analysis with Dynamic Consistency for Nonlinear Industrial Process Monitoring".

This project provides a unified experimental framework for comparing multiple multivariate statistical process monitoring (MSPM) methods on two classic industrial process datasets: **CSTR** (Continuous Stirred Tank Reactor) and **TE** (Tennessee Eastman).

The framework integrates state‑of‑the‑art algorithms including **KPCA, SFA, KSFA, KoopJointSFA(proposed), GLPQR, JTSVA, and DiCCA**. It automatically computes **False Alarm Rate (FAR)** and **Fault Detection Rate (FDR)** for each fault scenario, making it easy to reproduce experiments and compare algorithm performance.

---

## 📁 File Descriptions

| File | Process | Description |
| :--- | :--- | :--- |
| `CSTR_main.py` | CSTR | Loads CSTR data, runs the selected monitoring method on all fault types, and outputs FAR/FDR statistics. |
| `TE_main.py` | TE | Loads TE data, runs the monitoring method on 18 selected fault cases, and outputs FAR/FDR. |

Both scripts follow a similar pipeline: **data loading → standardisation → method invocation → performance evaluation**, which facilitates extension to other processes or new algorithms.

---

## ✨ Key Features

- **Multiple methods integrated** – supports 7+ mainstream monitoring algorithms; switch by simply changing the `method` variable.
- **Automated data handling** – built‑in functions (`construct_cstr_data_path` / `construct_data_path`) automatically split training and test sets.
- **Standardised preprocessing** – uses `StandardScaler` fitted on training data and applied consistently to test data.
- **Automatic performance metrics** – calls `monitoring_metrics` to compute FAR/FDR and `printing_metrics` to display results in a table.
- **Result saving support** – creates a dedicated folder per method (reserved for future plots or statistics).
- **Reproducibility** – sets `np.random.seed(42)` for deterministic results.

---

## 🛠️ Dependencies

- Python 3.7+
- NumPy
- scikit‑learn
- tqdm
- Custom internal modules: `Process.cstr_load`, `Process.te_load`, `Process.Monitor`, `Process.tools`

Install the required packages via:

```bash
pip install numpy scikit-learn tqdm
```

> **Note**: The `Process` package must be in your Python path or located in the same directory as the main scripts.

---

## 🚀 Usage

### 1. Choose a monitoring method
In the `__main__` section, set the `method` variable to one of the following:

- `"KPCA"`
- `"SFA"`
- `"KSFA"`
- `"KoopJointSFA"` (proposed)
- `"GLPQR"`
- `"JTSVA"`
- `"DiCCA"`

### 2. Configure data paths
- **CSTR**: `train_path, test_path = construct_cstr_data_path()` builds the internal paths; modify the function if needed.
- **TE**: ensure `global_root_folder` points to the root directory of the TE dataset.

### 3. Run the experiment
Execute the script:

```bash
python CSTR_main.py
# or
python TE_main.py
```

The program iterates through all faults (all for CSTR; 18 selected for TE) and for each:

1. Loads the corresponding test data.
2. Preprocesses the data according to the method (e.g., selecting 33 variables, splitting quality variables, etc.).
3. Calls the appropriate `Monitor` function to train the model and perform online monitoring.
4. Computes FAR and FDR for that fault.
5. Finally prints the average FAR/FDR across all faults.

---

## 📈 Output

After execution, the console shows results similar to:

```
=================Method:KoopJointSFA=================
Average FAR: 0.0234
Average FDR: 0.8765
...... ...... ...... ...... ...... ......
...... ...... ...... ...... ...... ......
=================Method:KoopJointSFA=================
```

- **FAR** (False Alarm Rate) – fraction of false alarms under normal conditions (evaluates stability).
- **FDR** (Fault Detection Rate) – fraction of successfully detected faults (evaluates sensitivity).

---

## 🔧 Customisation & Extension

To add a new monitoring method:

1. Implement `YourMethodMonitor` in `Process/Monitor.py` with the same interface.
2. Add a new `elif` branch in `CSTR_main.py` / `TE_main.py` to call your method.
3. Set `method = "YourMethod"` to run it.


- **License**: MIT © 2026
