# Run Commands for the Hybrid GNN NOMA Project

Example command-line runs for testing and demonstration.

## How to use

1. Open the project folder in Visual Studio Code (**File → Open Folder**, choose the folder that contains `main.py`).
2. Open the terminal (**Terminal → New Terminal**). It opens in the project folder automatically.
3. Install the dependencies once:
   ```
   pip install -r requirements.txt
   ```
4. Copy and run **one** of the commands below.

> If `python` is not recognized (macOS/Linux), use `python3` instead. Outputs (figures, metrics, model) are saved in the folder given by `--output`.

## 1) Quick smoke test
```
python main.py --users 6 --epochs 2 --slots 3 --output test_run
```

## 2) Quick smoke test with plots displayed
```
python main.py --users 6 --epochs 2 --slots 3 --output test_run --show-plots
```

## 3) Small validation run
```
python main.py --users 8 --epochs 5 --slots 4 --output test_run_small
```

## 4) Mid-size experiment
```
python main.py --users 10 --epochs 10 --slots 5 --output results_mid
```

## 5) Default project run (reported results)
```
python main.py --users 10 --epochs 25 --slots 5 --output results
```

## 6) Run and save log to file
```
python main.py --users 6 --epochs 2 --slots 3 --output test_run > terminal_output_log.txt 2>&1
```

## 7) Run with explicit seed for reproducibility
```
python main.py --users 8 --epochs 3 --slots 4 --seed 42 --output reproducible_run
```

## 8) Full benchmark-style run
```
python main.py --users 12 --epochs 20 --slots 6 --output benchmark_run --show-plots
```

## Notes
- Use smaller values for fast tests and debugging.
- Use bigger values for stronger benchmarking and demonstration runs.
- Keep the same `--seed` if you want reproducible datasets and comparable graphs.
- Use `--show-plots` only on a desktop/local environment where a GUI can open figures.