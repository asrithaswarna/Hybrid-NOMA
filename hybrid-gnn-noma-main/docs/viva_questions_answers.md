# Comprehensive Viva Voce Questions & Answers Guide

## Added ML Engineering Questions

### Q: Does the project require a real external dataset?
**Answer:** No. Synthetic wireless scenarios are the default and are generated reproducibly. A validated per-user scenario CSV can optionally be used. The CSV fields, units, missing-value rules, and duplicate-ID checks are documented in `data/README.md`.

### Q: What do the classical ML models predict?
**Answer:** Ridge Regression, Decision Tree Regression, and Random Forest Regression predict a continuous single-user achievable rate calculated from the same configured wireless simulation assumptions. This is a supervised simulation target, not a measured real-world label and not the GNN pairing target.

### Q: Why is scenario-level splitting important?
**Answer:** All users from one network realization stay in one split. This avoids putting highly related rows from the same scenario into both training and test sets.

### Q: Is SCMA integrated into the final scheduler?
**Answer:** No. The current scheduler evaluates PD-NOMA pairs and sequential slot assignment. SCMA has a working separate factor-graph and MPA detector experiment. Calling it fully integrated hybrid scheduling would overstate the implementation.

This document prepares project team members for viva voce examinations, technical interviews, and faculty cross-examination on **"Graph Neural Network-Based Dynamic User Pairing and Slot Assignment for Hybrid NOMA Networks"**.

---

## Category 1: Wireless Communications & NOMA Fundamentals

### Q1: What is the fundamental difference between OMA and NOMA?
**Answer**:
- **Orthogonal Multiple Access (OMA)**: Allocates strictly disjoint resources in time (TDMA), frequency (FDMA/OFDMA), or code (CDMA) so that user signals are mutually orthogonal, eliminating intra-cell interference at the expense of limited spectral efficiency and user connectivity bounded by the number of orthogonal channels.
- **Non-Orthogonal Multiple Access (NOMA)**: Intentionally multiplexes multiple users on the same frequency-time resource block simultaneously by exploiting power-domain differences (PD-NOMA) or multi-dimensional codebook sparsity (SCMA), using advanced multi-user detection (SIC or MPA) at the receiver to achieve higher spectral efficiency and massive connectivity.

### Q2: In downlink PD-NOMA, why must the weak user be allocated more transmit power than the strong user?
**Answer**:
Power allocation is deliberately inverse to channel quality ($\alpha_w > \alpha_s$):
1. **Fairness**: The weak user (cell-edge) suffers greater path loss. Allocating it high power ($\alpha_w = 0.75$) ensures its received signal power is sufficient to decode its message while treating the strong user's low-power signal ($\alpha_s = 0.25$) as negligible interference.
2. **SIC Feasibility**: The strong user (near-cell) has a superior channel ($g_s > g_w$). For the strong user to successfully decode and subtract the weak user's signal via SIC, its received SINR for decoding the weak signal must be at least as high as the weak user's own SINR ($\text{SINR}_{s \to w} \ge \text{SINR}_w$). This condition is mathematically guaranteed when $g_s \ge g_w$ and $\alpha_w > \alpha_s$.

### Q3: What is Successive Interference Cancellation (SIC), and what causes "Imperfect SIC"?
**Answer**:
- **SIC Mechanism**: The receiver decodes the superimposed signals sequentially in descending order of signal power. The strongest signal is decoded first, reconstructed, and subtracted from the total received signal before decoding the subsequent signal.
- **Imperfect SIC**: In theoretical models, cancellation is assumed to be perfect ($\beta_{\text{SIC}} = 0$). In physical systems, imperfect channel state information (CSI), quantization errors, carrier frequency offsets, and phase noise leave residual interference:
  $$I_{\text{residual}} = \beta_{\text{SIC}} \alpha_w P_s g_s$$
  where $\beta_{\text{SIC}} \in [0.01, 0.05]$ represents residual power leakage, creating an interference floor that limits the strong user's maximum achievable rate.

### Q4: What is Sparse Code Multiple Access (SCMA), and how does it differ from PD-NOMA?
**Answer**:
- **PD-NOMA**: Operates via power-domain superposition of 2 users per resource block, decoded through successive cancellation in the time/power domain.
- **SCMA**: A code-domain non-orthogonal scheme where bits are directly mapped to multidimensional sparse complex codewords selected from predefined codebooks. Signals from $J$ users are spread across $K$ subcarriers ($J > K$) according to a bipartite factor graph, and multi-user detection is performed simultaneously across all colliding subcarriers using the iterative Message Passing Algorithm (MPA).

### Q5: How is the overloading factor defined in SCMA?
**Answer**:
The overloading factor is the ratio of multiplexed user layers $J$ to the number of orthogonal resource elements $K$:
$$\lambda = \frac{J}{K} \times 100\%$$
In our canonical architecture, $J=6$ users are multiplexed over $K=4$ orthogonal resource subcarriers, yielding an overloading factor of $\lambda = 6/4 = 150\%$.

### Q6: How does the Message Passing Algorithm (MPA) achieve low-complexity detection in SCMA?
**Answer**:
Full maximum likelihood (ML) detection has exponential complexity $\mathcal{O}(M^J)$ where $M$ is codebook size. In contrast, SCMA codebooks are sparse: each user transmits on only $d_v = 2$ subcarriers, and each subcarrier experiences collisions from only $d_f = 3$ users.
MPA exploits this sparsity by passing probability messages along the edges of the factor graph between variable (user) nodes and factor (subcarrier) nodes, marginalizing over only $d_f - 1$ colliding users instead of all $J$ users. This reduces complexity to $\mathcal{O}(N_{\text{iter}} \cdot K \cdot d_f \cdot M^{d_f})$, enabling practical multi-user detection.

---

## Category 2: Graph Neural Networks & Deep Learning

### Q7: Why did you use a Graph Neural Network (GNN) instead of a standard Multilayer Perceptron (MLP)?
**Answer**:
1. **Permutation Equivariance**: A wireless network has no canonical ordering of users. If user indices are swapped, an MLP treats it as a completely new input vector and produces arbitrary, inconsistent outputs. A GNN operates on graphs where node and edge representations are invariant or equivariant under arbitrary node permutations.
2. **Scalability**: An MLP requires fixed-dimension input vectors ($N \times D$). A GNN with shared message passing weights can accept variable numbers of users without retraining.
3. **Relational Inductive Bias**: Pairing is inherently an edge-level relational decision between pairs of nodes. GNN edge-conditioned layers directly incorporate pairwise channel disparities.

### Q8: What features are fed to the GNN, and how did you prevent data leakage?
**Answer**:
- **Node Features (6D)**: Normalized spatial coordinates $(x/R, y/R)$, normalized distance $(d/R)$, standardized channel gain in dB, channel gain normalized by mean cell gain, and normalized QoS rate requirement.
- **Edge Features (4D)**: Normalized inter-user distance, channel gain disparity in dB $|g_{i, \text{dB}} - g_{j, \text{dB}}| / 40.0$, channel correlation $|\mathbf{h}_i^H \mathbf{h}_j| / (\|\mathbf{h}_i\|\|\mathbf{h}_j\|)$, and combined interference potential.
- **Zero Data Leakage**: Node and edge features are computed strictly from raw channel and geographic states before any scheduling decision is made. Optimization oracle labels (ground truth pairing and slot assignments) are used solely in the supervised loss function during training and are never exposed to the model inputs.

### Q9: How does the GNN model handle class imbalance in user pairing?
**Answer**:
In a network of $N=10$ users, there are $N(N-1) = 90$ directed edges in the complete graph, but an optimal disjoint pairing contains only $N = 10$ active directed edges (5 undirected pairs). Hence, $89\%$ of edges are negative.
We mitigate this by using a positive-class weighted Binary Cross-Entropy loss with:
$$w_{\text{pos}} = N - 2$$
giving higher loss penalty to false negatives and preventing the model from predicting trivial all-zero pairing matrices.

---

## Category 3: Optimization & System Engineering

### Q10: How does the Combinatorial Oracle generate ground truth labels?
**Answer**:
- For small networks ($N \le 10$): The oracle performs exhaustive enumeration over all distinct pairing partitions. For $N=10$, there are $(9 \times 7 \times 5 \times 3 \times 1) = 945$ possible pairings. The oracle evaluates each pairing under the exact sum-rate utility objective and selects the global mathematical optimum.
- For larger networks ($N > 10$): The oracle uses Edmonds' Blossom maximum-weight matching algorithm with $\mathcal{O}(V^3)$ polynomial complexity on a complete graph with edge weights set to candidate pair sum-rates.

### Q11: How do you guarantee that the GNN output never produces duplicate users or invalid schedules?
**Answer**:
The raw GNN output provides continuous edge affinity probabilities $\hat{A}_{ij} \in [0, 1]$. We pass these probabilities through a deterministic constraint-preserving post-processing algorithm:
1. Symmetrize the matrix: $\mathbf{A}_{\text{sym}} = \frac{1}{2}(\hat{\mathbf{A}} + \hat{\mathbf{A}}^T)$ and zero out the diagonal.
2. Sort candidate pairs by descending affinity.
3. Greedily match users while tracking an `active_users` set, strictly forbidding any user from being matched twice.
4. Any remaining user (in odd $N$ cases) is routed to single-user OFDMA mode.
This guarantees $100\%$ valid, collision-free schedules in every trial.

### Q12: What is Jain's Fairness Index, and why is it important in NOMA?
**Answer**:
Jain's Fairness Index is defined as:
$$J = \frac{\left(\sum_{i=1}^N R_i\right)^2}{N \sum_{i=1}^N R_i^2} \in \left[\frac{1}{N}, 1.0\right]$$
If an algorithm purely maximizes sum-rate, it may starve cell-edge weak users and allocate all resources to near-cell strong users with favorable channels. Jain's Fairness Index quantitatively evaluates whether all users receive an equitable share of throughput ($J \to 1.0$ indicates equal rates across all users).

### Q13: What is the computational complexity of GNN inference versus the Oracle?
**Answer**:
- Combinatorial Exhaustive Oracle: $\mathcal{O}((N-1)!!)$, which grows super-exponentially (e.g. $> 6.5 \times 10^{11}$ operations for $N=20$).
- Proposed GNN: Forward message passing requires $\mathcal{O}(L \cdot (|\mathcal{V}| d + |\mathcal{E}| d))$ operations, executing in under $3\text{ ms}$ on standard CPUs, making it practical for real-time 5G/6G slot scheduling.
