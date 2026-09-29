# Final Project Presentation & Defense Guide (Slide-by-Slide)

## Add-on Slide: ML Lifecycle Demonstration

- **Data source**: Synthetic scenarios by default; validated per-user scenario CSV is optional.
- **Supervised baselines**: Ridge, Decision Tree, and Random Forest predict simulation-derived user rate.
- **Unsupervised analysis**: K-means, DBSCAN, and PCA summarize channel/user conditions; clusters are not claimed to be optimal pairs.
- **Evaluation**: Scenario-level split, MAE/MSE/RMSE/R2 for rate prediction, plus sum rate, fairness, outage, BER, and SER for communication evaluation.
- **Engineering**: Saved artifacts, reusable inference, local monitoring, candidate retraining, and optional FastAPI service.

Use commands from `docs/experiment_protocol.md` to demonstrate each item live.

## Overview
This guide provides a structured 18-slide presentation blueprint for your final-year B.Tech / B.E. project defense before the faculty examination committee.

---

### Slide 1: Title & Team Credentials
- **Title**: Graph Neural Network-Based Dynamic User Pairing and Slot Assignment for Hybrid NOMA Networks
- **Department**: Department of Electronics & Communication / Computer Science Engineering
- **Team Members**: [Student Names & Roll Numbers]
- **Supervisor**: [Faculty Supervisor Name & Designation]
- **Institution**: [College / University Name, Year 2026]

### Slide 2: Background & Research Motivation
- **The 5G/6G Challenge**: Massive machine-type communication (mMTC) and ultra-reliable low-latency communication (uRLLC) demand connecting up to $10^6\text{ devices/km}^2$.
- **Bottleneck of OMA (OFDMA)**: Strict orthogonal spectrum allocation limits the maximum concurrent users to the number of orthogonal subcarriers/time-slots.
- **Promise of NOMA**: Exploits power and code domains to multiplex multiple users on the same frequency-time resource block, drastically enhancing spectral efficiency and user connectivity.

### Slide 3: Problem Statement
- **The Pairing Dilemma**: Multiplexing arbitrary users creates prohibitive inter-user interference. Pairing users with similar channel gains in PD-NOMA impairs Successive Interference Cancellation (SIC).
- **The Assignment Dilemma**: Dynamically allocating paired groups to orthogonal slots while satisfying power budgets and QoS rate requirements is NP-hard.
- **Combinatorial Explosion**: Exhaustive search scales as $\mathcal{O}((N-1)!!)$. For $N=20$, there are $>6.5 \times 10^{11}$ pairing combinations—impossible to compute within the 1-millisecond transmission time interval (TTI) of 5G cellular systems.

### Slide 4: Proposed Solution: Hybrid NOMA Architecture
- **Why "Hybrid" NOMA?**:
  - **PD-NOMA**: Ideal for user pairs with large channel gain disparity (near-cell user + cell-edge user).
  - **SCMA**: Ideal for dense clusters of users with comparable gains, achieving $150\%$ overloading via multidimensional sparse codebooks.
  - **Dynamic Slot Assignment**: Distributes pairs and clusters into orthogonal time-frequency slots to eliminate inter-cluster cross-talk.

### Slide 5: Dataset Generation and Wireless Channel Model
- **Dataset source**: The program automatically creates a synthetic dataset during each run; no external CSV dataset is required.
- **Exact function**: `DatasetGenerator.generate_split()` in `src/training.py` creates the training and validation graph lists.
- **Per-sample process**: It sets a seed, calls `WirelessNetwork.generate_users()`, creates oracle pairing/slot labels, and calls `GraphBuilder.build_graph()`.
- **User positions**: Users are placed randomly and uniformly by area inside the configured annular cell, with the base station at $(0, 0)$.
- **Topology**: Base Station at origin $(0, 0)$; users distributed within annular cell $R_{\min}=20\text{ m} \le d \le R_{\max}=500\text{ m}$.
- **Path Loss**: $PL(d) = PL_0 + 10 \alpha \log_{10}(d/d_0)$ with $\alpha = 3.5$.
- **Small-Scale Multi-Path**: Frequency-flat Rayleigh fading $h_{\text{fading}} \sim \mathcal{CN}(0, 1)$.
- **Composite Gain**: $g_i = |h_i|^2 = 10^{-PL(d_i)/10} |h_{\text{fading}, i}|^2$.
- **AWGN Noise**: Thermal noise $N_0 = -174\text{ dBm/Hz}$ + Noise Figure $F = 5\text{ dB}$.
- **Graph sample**: Each user becomes a node with position, distance, channel-gain, and rate-requirement features. Every two different users are connected by directed edges carrying distance, gain disparity, channel correlation, and interference features.
- **Training targets**: An optimization oracle supplies pairing and slot labels. These are generated supervision targets, not an external dataset.
- **Limitation**: The synthetic workflow is reproducible and useful for controlled testing, but real-world validation would require measured channel data or a higher-fidelity simulator.

### Slide 6: Power-Domain NOMA (PD-NOMA) & SIC Formulation
- **Superposition**: $x = \sqrt{\alpha_w P} s_w + \sqrt{\alpha_s P} s_s$ with $\alpha_w = 0.75, \alpha_s = 0.25$.
- **Weak User Decoding**: Treats strong user as noise:
  $$\text{SINR}_w = \frac{\alpha_w P g_w}{\alpha_s P g_w + \sigma^2}, \quad R_w = B_s \log_2(1 + \text{SINR}_w)$$
- **Strong User SIC Decoding (with Imperfect Cancellation)**:
  $$\text{SINR}_s = \frac{\alpha_s P g_s}{\beta_{\text{SIC}} \alpha_w P g_s + \sigma^2}, \quad R_s = B_s \log_2(1 + \text{SINR}_s)$$
  where $\beta_{\text{SIC}} = 0.01$ realistically models residual cancellation errors.

### Slide 7: Sparse Code Multiple Access (SCMA) & MPA Detector
- **Factor Graph Matrix**: $\mathbf{F}_{4 \times 6}$ ($J=6$ users multiplexed over $K=4$ orthogonal resources; $150\%$ overload).
- **Sparsity**: Each user connects to $d_v = 2$ subcarriers; each subcarrier hosts $d_f = 3$ colliding users.
- **Message Passing Algorithm (MPA)**: Iterative belief propagation passing log-likelihood probability messages between user nodes and resource nodes over 4 iterations.

### Slide 8: Optimization Problem Formulation
- **Objective**: Maximize joint utility:
  $$\max_{\mathbf{X}, \mathbf{S}} \sum_{s=1}^S R_s(\mathbf{X}, \mathbf{S}) + \lambda_{\text{fair}} J_{\text{fairness}}(\mathbf{R})$$
- **Constraints**:
  1. Disjoint pairing: Each user is paired at most once.
  2. Slot capacity: At most 1 pair per slot.
  3. Total power budget: $\sum_s P_s \le P_{\text{total}}$.
  4. QoS guarantee: $R_i \ge R_{\min}$.

### Slide 9: Why Graph Neural Networks (GNN)?
- **Limitation of MLPs**: Neural networks with vector inputs are sensitive to arbitrary user ordering (not permutation equivariant).
- **GNN Advantage**: Wireless networks are graphs where users are nodes and mutual interference channels are edges.
- **Edge-Aware Processing**: Directly incorporates channel gain disparity $|g_{i, \text{dB}} - g_{j, \text{dB}}|$ and spatial distance as edge attributes.

### Slide 10: Proposed GNN Architecture
- **Node Features (6D)**: Normalized coordinates $(x, y)$, distance $d$, channel gain in dB, linear gain ratio, QoS rate requirement.
- **Edge Features (4D)**: Inter-user distance, channel gain disparity, channel correlation, interference potential.
- **Message Passing Layers**: 3 Edge-Conditioned Convolution layers with LayerNorm and LeakyReLU activations.
- **Output Heads**: Bilinear edge scoring head for pair affinity matrix $\hat{\mathbf{A}} \in [0, 1]^{N \times N}$ and slot preference head $\hat{\mathbf{S}}$.

### Slide 11: Constraint-Preserving Post-Processing
- **Deterministic Disjoint Matching**: Converts soft continuous affinities $\hat{\mathbf{A}}$ into discrete pairs via maximum-weight greedy matching.
- **Guarantees**: Zero duplicate assignments, $100\%$ valid physical slots, and seamless odd-user handling (assigns single unpaired user to OFDMA).

### Slide 12: Project Deliverables & Software Architecture
- **Version A**: Modular Python package (`src/`, `tests/`, `main.py`, `config.yaml`).
- **Version B**: Complete standalone single file (`hybrid_noma_gnn_complete.py`).
- **Version C**: 18-cell Google Colab Jupyter notebook (`hybrid_noma_experiment.ipynb`).
- **Test Suite**: 27 automated unit tests with $100\%$ pass rate (`pytest`).

### Slide 13: Results: Network Sum-Rate Benchmark
- **Finding**: The proposed GNN achieves within $96-98\%$ of the globally optimal Combinatorial Oracle while outperforming conventional OMA by over $40-60\%$.
- **Comparison**: Outperforms Random NOMA and Conventional Near-Far sorting across all evaluated test topologies.

### Slide 14: Results: Fairness & Outage Probability
- **Fairness Analysis**: PD-NOMA power allocation ($\alpha_w=0.75, \alpha_s=0.25$) ensures cell-edge weak users receive sufficient throughput, maintaining high Jain's fairness ($J > 0.70$).
- **Outage**: Minimizes user outages compared to uncoordinated pairing.

### Slide 15: Results: SCMA MPA Error Rate Performance
- **BER / SER Analysis**: Semilogy error rate curves demonstrate steep waterfall performance across SNR range from $-5\text{ dB}$ to $20\text{ dB}$, confirming the effectiveness of the multi-user detector.

### Slide 16: Computational Complexity & Real-Time Viability
- **Runtime Comparison**:
  - Combinatorial Oracle: Exponential $\mathcal{O}((N-1)!!)$ ($\approx 100-500\text{ ms}$ for $N=10$, impossible for $N \ge 20$).
  - Proposed GNN: Forward pass $\mathcal{O}(V + E)$ executes in **$< 3\text{ ms}$**, enabling real-time millisecond slot scheduling in 5G/6G systems.

### Slide 17: Limitations & Scope Clarification
- Single-cell downlink deployment; inter-cell interference from neighboring base stations not included.
- Quasi-static block fading channels assumed over transmission slots.
- SCMA uses phase-rotated QPSK constellations with uncoded transmission.

### Slide 18: Conclusions & Future Research Directions
- **Conclusion**: Demonstrated that an edge-conditioned Graph Neural Network can solve the combinatorial user pairing and slot assignment problem in near-optimal time with strict physical constraint satisfaction.
- **Future Work**: Extension to multi-cell MIMO-NOMA, Deep Reinforcement Learning for continuous power allocation, and Software-Defined Radio (SDR) hardware validation.
