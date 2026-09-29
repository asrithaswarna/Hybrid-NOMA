# Experimental Evaluation Protocol & Benchmark Guide

## 0. Additional ML Workflows

The default `--mode train --data-source synthetic` command preserves the original GNN/NOMA workflow. Optional validated CSV scenarios can be used for GNN training/evaluation, classical regression, and clustering:

```bash
python main.py --mode compare --data-source synthetic --users 6 --output results_compare
python main.py --mode compare --data-source csv --dataset data/example_users.csv --users 4 --output results_csv
python main.py --mode clustering --data-source synthetic --users 6 --output results_clusters
python main.py --mode inference --model-path results_compare/classical_models/random_forest.joblib --output results_compare
python main.py --mode retrain --data-source synthetic --users 6 --output results_retrain
```

The classical task is user-rate regression using scenario-level train/validation/test separation. Reported metrics are MAE, MSE, RMSE, and R2. Wireless metrics such as sum rate, Jain fairness, outage, BER, and SER remain separate domain metrics from the GNN/NOMA evaluator. A candidate retraining run does not silently overwrite an existing model.

## 1. Objective & Benchmark Principles
This protocol outlines the scientific procedure to reproduce and validate all experimental comparisons presented in the B.Tech project.

To maintain strict scientific validity:
1. **Identical Topologies**: All algorithms (OMA, Random NOMA, Near-Far NOMA, Greedy NOMA, Oracle, GNN) are evaluated on the exact same channel realizations for each trial.
2. **Zero Data Leakage**: The test set topologies use independent random seeds (`seed = 2000` to `2049`) not seen during GNN training or validation.
3. **Statistical Reporting**: All results are reported as **Mean $\pm$ Standard Deviation** over multiple independent random network instances.

---

## 2. Experimental Execution Commands

### Execution of Modular Project (Version A)
```bash
# Standard 10-user run with 25 epochs
python main.py --users 10 --epochs 25 --seed 42 --slots 5 --output results

# Fast verification run (e.g. quick test on 6 users)
python main.py --users 6 --epochs 10 --seed 42 --slots 3 --output results_quick
```

### Execution of Standalone Script (Version B)
```bash
# Standard run
python hybrid_noma_gnn_complete.py --users 10 --epochs 25 --seed 42 --slots 5 --output results

# Custom user count
python hybrid_noma_gnn_complete.py --users 8 --epochs 30 --seed 123 --slots 4 --output results
```

### Automated Unit Testing Suite
```bash
# Run all tests with verbose output
python -m pytest tests/ -v
```

---

## 3. Evaluation Metrics

1. **Total Network Sum Rate ($\text{Mbps}$)**:
   $$R_{\text{total}} = \sum_{s=1}^S R_s \quad [\text{Mbps}]$$
2. **Average User Rate ($\text{Mbps}$)**:
   $$\bar{R} = \frac{1}{N} \sum_{i=1}^N R_i \quad [\text{Mbps}]$$
3. **Minimum User Rate ($\text{Mbps}$)**:
   $$R_{\min} = \min_{i \in \{1, \dots, N\}} R_i$$
   Reflects quality-of-service for disadvantaged cell-edge users.
4. **Jain's Fairness Index**:
   $$J = \frac{\left(\sum_{i=1}^N R_i\right)^2}{N \sum_{i=1}^N R_i^2} \in \left[\frac{1}{N}, 1.0\right]$$
5. **Outage Probability**:
   $$P_{\text{outage}} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(R_i < R_{\text{threshold}})$$
6. **Slot Utilization Rate ($\%$)**:
   $$\eta_{\text{slot}} = \frac{S_{\text{active}}}{S} \times 100\%$$
7. **Inference Latency ($\text{ms}$)**:
   Runtime per network topology instance measured via high-resolution performance timers.
8. **Pairing Validity Rate ($\%$)**:
   Percentage of produced pairings satisfying all hard physical constraints (must equal 100.0%).

---

## 4. Expected Output Artifacts

Running the experiment produces the following structured outputs in `results/`:
- `results/figures/network_topology.png`: Spatial positions, Base Station, and pairing lines.
- `results/figures/training_loss_curves.png`: Training and validation loss convergence curves.
- `results/figures/sum_rate_benchmark.png`: Bar chart comparing Sum Rate and Jain's Fairness.
- `results/figures/scma_ber_curve.png`: Semilogy BER/SER curves for SCMA MPA detector across SNRs.
- `results/metrics/benchmark_results.csv`: Complete numerical metrics across all tested methods.
- `results/metrics/summary.json`: JSON metadata including hyperparameter records and performance aggregates.
- `results/best_gnn_model.pt`: Checkpoint of best GNN model weights.
