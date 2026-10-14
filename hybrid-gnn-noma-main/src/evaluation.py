"""
Evaluation and benchmarking module.
Evaluates OMA, Random NOMA, Near-Far NOMA, Greedy NOMA, Oracle, and GNN across
identical channel realizations for rigorous academic benchmarking.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any
import time
import numpy as np
import pandas as pd
import torch

from src.config import ProjectConfig
from src.channel_model import WirelessNetwork, UserChannelState
from src.graph_builder import GraphBuilder
from src.noma_pd import PDNOMAEngine
from src.noma_scma import SCMAEngine, SCMAResult
from src.pairing import UserPairingManager, PairingPlan
from src.slot_assignment import SlotAssignmentEngine, NetworkScheduleResult
from src.optimization import OptimizationOracle
from src.gnn_model import HybridNOMAGNN


@dataclass
class MethodBenchmarkMetrics:
    """Statistical summary of a pairing method over test trials."""
    method_name: str
    pairing_f1_score_mean: float
    sum_rate_mean_mbps: float
    sum_rate_std_mbps: float
    user_rate_mean_mbps: float
    user_rate_std_mbps: float
    min_user_rate_mean_mbps: float
    jains_fairness_mean: float
    jains_fairness_std: float
    outage_probability_mean: float
    slot_utilization_mean: float
    inference_time_ms: float
    validity_rate: float


class NetworkEvaluator:
    """Runs comprehensive comparative evaluations across all baseline and proposed algorithms."""

    def __init__(self, cfg: ProjectConfig, gnn_model: Optional[HybridNOMAGNN] = None):
        self.cfg = cfg
        self.gnn_model = gnn_model
        self.network = WirelessNetwork(cfg.network, cfg.channel, seed=cfg.system.seed + 999)
        self.pd_engine = PDNOMAEngine(cfg.noma_pd)
        self.scma_engine = SCMAEngine(cfg.noma_scma)
        self.slot_engine = SlotAssignmentEngine(
            num_slots=cfg.network.num_slots,
            total_bandwidth_hz=cfg.network.total_bandwidth,
            total_power_w=cfg.network.total_transmit_power_w,
            noise_power_per_slot=self.network.noise_power_per_slot,
            pd_engine=self.pd_engine,
            gain_ratio_threshold=cfg.noma_pd.gain_ratio_threshold,
        )
        self.pairing_mgr = UserPairingManager(self.pd_engine)
        self.oracle = OptimizationOracle(
            slot_engine=self.slot_engine,
            pd_engine=self.pd_engine,
            fairness_weight=cfg.optimization.fairness_weight,
        )
        self.graph_builder = GraphBuilder(max_cell_radius=cfg.network.cell_radius_max)
        self.method_ber_curves: Dict[str, List[Dict[str, float]]] = {}

    def _schedule_user_sinr(
        self,
        schedule: NetworkScheduleResult,
        users: List[UserChannelState],
        user_id: int,
        snr_db: float,
        reference_gain: float,
    ) -> float:
        """Get the scheduled user's effective SINR using the selected slot mode."""
        user = next((item for item in users if item.user_id == user_id), None)
        if user is None:
            return 0.0

        snr_linear = 10.0 ** (snr_db / 10.0)
        for record in schedule.slots:
            if user_id not in record.assigned_user_ids or record.mode == "IDLE":
                continue

            slot_gain = self.slot_engine._slot_quality_factor(record.slot_id)
            slot_noise = self.slot_engine._slot_noise_factor(record.slot_id)
            noise_power = record.power_w * reference_gain / snr_linear * slot_noise
            user_gain = user.channel_gain * slot_gain

            if record.mode == "PD-NOMA" and record.pd_pair_details is not None:
                pair = record.pd_pair_details
                if user_id == pair.weak_user_id:
                    desired = pair.alpha_weak * record.power_w * user_gain
                    interference = pair.alpha_strong * record.power_w * user_gain
                else:
                    desired = pair.alpha_strong * record.power_w * user_gain
                    interference = (
                        pair.sic_residual_factor * pair.alpha_weak
                        * record.power_w * user_gain
                    )
                return float(desired / max(interference + noise_power, 1e-30))

            if record.mode == "SCMA":
                other_id = next(
                    (assigned_id for assigned_id in record.assigned_user_ids if assigned_id != user_id),
                    None,
                )
                other = next((item for item in users if item.user_id == other_id), None)
                if other is None:
                    return 0.0
                pair_gain = user_gain + other.channel_gain * slot_gain
                interference = 0.18 * pair_gain
                return float(
                    record.power_w * pair_gain
                    / (2.0 * max(interference + noise_power, 1e-30))
                )

            return float(record.power_w * user_gain / max(noise_power, 1e-30))

        return 0.0

    def _schedule_ber_errors(
        self,
        schedule: NetworkScheduleResult,
        users: List[UserChannelState],
        snr_db: float,
        seed: int,
        num_bits: int = 4096,
    ) -> int:
        """Simulate uncoded BPSK over each scheduled link's effective AWGN SINR."""
        reference_gain = float(np.mean([user.channel_gain for user in users])) if users else 0.0
        total_errors = 0
        for user in users:
            sinr = self._schedule_user_sinr(
                schedule, users, user.user_id, snr_db, reference_gain
            )
            rng = np.random.default_rng(seed + user.user_id)
            bits = rng.integers(0, 2, size=num_bits)
            symbols = 1.0 - 2.0 * bits
            noise_std = np.sqrt(1.0 / (2.0 * max(sinr, 1e-12)))
            received = symbols + rng.normal(0.0, noise_std, size=num_bits)
            detected_bits = received < 0.0
            total_errors += int(np.count_nonzero(detected_bits != bits))
        return total_errors

    def evaluate_test_set(
        self,
        num_test_instances: int = 50,
        seed_offset: int = 2000,
        scenarios: Optional[List[List[UserChannelState]]] = None,
    ) -> Dict[str, MethodBenchmarkMetrics]:
        """
        Evaluate all methods on the exact same test channel realizations.
        """
        methods = [
            "Orthogonal Multiple Access (OMA)",
            "Random Pairing PD-NOMA",
            "Near-Far Channel Gain NOMA",
            "Greedy Sum-Rate NOMA",
            "Optimization Oracle",
        ]
        if self.gnn_model is not None:
            methods.append("Proposed GNN-Based NOMA")

        # Accumulator dictionaries
        sum_rates: Dict[str, List[float]] = {m: [] for m in methods}
        avg_rates: Dict[str, List[float]] = {m: [] for m in methods}
        min_rates: Dict[str, List[float]] = {m: [] for m in methods}
        fairness: Dict[str, List[float]] = {m: [] for m in methods}
        outages: Dict[str, List[float]] = {m: [] for m in methods}
        utilizations: Dict[str, List[float]] = {m: [] for m in methods}
        latencies: Dict[str, List[float]] = {m: [] for m in methods}
        validities: Dict[str, List[bool]] = {m: [] for m in methods}
        pairing_f1_scores: Dict[str, List[float]] = {m: [] for m in methods}
        ber_method_names = [
            "Random Pairing PD-NOMA",
            "Near-Far Channel Gain NOMA",
            "Greedy Sum-Rate NOMA",
            "Optimization Oracle",
            "Proposed GNN-Based NOMA",
        ]
        ber_method_names = [name for name in ber_method_names if name in methods]
        ber_snr_db = [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0]
        ber_error_counts = {name: np.zeros(len(ber_snr_db), dtype=np.int64) for name in ber_method_names}
        ber_total_bits = np.zeros(len(ber_snr_db), dtype=np.int64)

        evaluation_count = len(scenarios) if scenarios is not None else num_test_instances
        for idx in range(evaluation_count):
            if scenarios is None:
                self.network.set_seed(seed_offset + idx)
                users = self.network.generate_users(
                    num_users=self.cfg.network.num_users,
                    min_rate_bps=self.cfg.optimization.min_rate_threshold_bps,
                )
            else:
                users = scenarios[idx]
            n = len(users)
            ber_schedules: Dict[str, NetworkScheduleResult] = {}
            pairing_plans: Dict[str, PairingPlan] = {}

            # 1. OMA Baseline (One user per slot up to S slots)
            t0 = time.perf_counter()
            oma_plan = PairingPlan(pairs=[], unpaired_users=[u.user_id for u in users], method_name="OMA")
            pairing_plans["Orthogonal Multiple Access (OMA)"] = oma_plan
            oma_sched = self.slot_engine.evaluate_schedule(oma_plan, users)
            lat_oma = (time.perf_counter() - t0) * 1000.0
            self._record_metric("Orthogonal Multiple Access (OMA)", oma_sched, lat_oma, oma_plan.validate(n),
                                sum_rates, avg_rates, min_rates, fairness, outages, utilizations, latencies, validities)

            # 2. Random Pairing
            t0 = time.perf_counter()
            rand_plan = self.pairing_mgr.random_pairing(users)
            pairing_plans["Random Pairing PD-NOMA"] = rand_plan
            rand_sched = self.slot_engine.evaluate_schedule(rand_plan, users)
            lat_rand = (time.perf_counter() - t0) * 1000.0
            self._record_metric("Random Pairing PD-NOMA", rand_sched, lat_rand, rand_plan.validate(n),
                                sum_rates, avg_rates, min_rates, fairness, outages, utilizations, latencies, validities)
            ber_schedules["Random Pairing PD-NOMA"] = rand_sched

            # 3. Near-Far Channel Gain
            t0 = time.perf_counter()
            nf_plan = self.pairing_mgr.channel_gain_pairing(users, strategy="near_far")
            pairing_plans["Near-Far Channel Gain NOMA"] = nf_plan
            nf_sched = self.slot_engine.evaluate_schedule(nf_plan, users)
            lat_nf = (time.perf_counter() - t0) * 1000.0
            self._record_metric("Near-Far Channel Gain NOMA", nf_sched, lat_nf, nf_plan.validate(n),
                                sum_rates, avg_rates, min_rates, fairness, outages, utilizations, latencies, validities)
            ber_schedules["Near-Far Channel Gain NOMA"] = nf_sched

            # 4. Greedy Sum-Rate
            t0 = time.perf_counter()
            greedy_plan = self.pairing_mgr.greedy_sum_rate_pairing(
                users,
                power_w=self.slot_engine.slot_power,
                bandwidth_hz=self.slot_engine.slot_bandwidth,
                noise_power_w=self.slot_engine.noise_power_per_slot,
            )
            pairing_plans["Greedy Sum-Rate NOMA"] = greedy_plan
            greedy_sched = self.slot_engine.evaluate_schedule(greedy_plan, users)
            lat_greedy = (time.perf_counter() - t0) * 1000.0
            self._record_metric("Greedy Sum-Rate NOMA", greedy_sched, lat_greedy, greedy_plan.validate(n),
                                sum_rates, avg_rates, min_rates, fairness, outages, utilizations, latencies, validities)
            ber_schedules["Greedy Sum-Rate NOMA"] = greedy_sched

            # 5. Optimization Oracle
            t0 = time.perf_counter()
            if n <= 10:
                oracle_plan, oracle_sched = self.oracle.solve_exhaustive(users)
            else:
                oracle_plan, oracle_sched = self.oracle.solve_max_weight_matching(users)
            pairing_plans["Optimization Oracle"] = oracle_plan
            lat_oracle = (time.perf_counter() - t0) * 1000.0
            self._record_metric("Optimization Oracle", oracle_sched, lat_oracle, oracle_plan.validate(n),
                                sum_rates, avg_rates, min_rates, fairness, outages, utilizations, latencies, validities)
            ber_schedules["Optimization Oracle"] = oracle_sched

            # 6. Proposed GNN
            if self.gnn_model is not None:
                t0 = time.perf_counter()
                graph = self.graph_builder.build_graph(users)
                self.gnn_model.eval()
                with torch.no_grad():
                    aff_probs = self.gnn_model.predict_pairing_probabilities(
                        graph.node_features, graph.edge_index, graph.edge_features
                    ).cpu().numpy()
                    slot_logits = self.gnn_model.predict_slot_logits(
                        graph.node_features, graph.edge_index, graph.edge_features
                    ).cpu()
                gnn_plan = self.pairing_mgr.gnn_affinity_pairing(aff_probs, users)
                pairing_plans["Proposed GNN-Based NOMA"] = gnn_plan
                gnn_slot_mapping = self.slot_engine.resolve_gnn_slot_mapping(gnn_plan, users, slot_logits)
                gnn_sched = self.slot_engine.evaluate_schedule(gnn_plan, users, custom_slot_mapping=gnn_slot_mapping)
                lat_gnn = (time.perf_counter() - t0) * 1000.0
                self._record_metric("Proposed GNN-Based NOMA", gnn_sched, lat_gnn, gnn_plan.validate(n),
                                    sum_rates, avg_rates, min_rates, fairness, outages, utilizations, latencies, validities)
                ber_schedules["Proposed GNN-Based NOMA"] = gnn_sched

            for method_name, plan in pairing_plans.items():
                pairing_f1_scores[method_name].append(
                    self._pairing_f1_score(plan, oracle_plan)
                )

            for method_name, schedule in ber_schedules.items():
                for snr_idx, snr_db in enumerate(ber_snr_db):
                    ber_error_counts[method_name][snr_idx] += self._schedule_ber_errors(
                        schedule, users, snr_db, seed=seed_offset + idx * 1009 + snr_idx * 9176
                    )
            ber_total_bits += n * 4096

        # Aggregate statistics
        results: Dict[str, MethodBenchmarkMetrics] = {}
        for m in methods:
            results[m] = MethodBenchmarkMetrics(
                method_name=m,
                pairing_f1_score_mean=float(np.mean(pairing_f1_scores[m])),
                sum_rate_mean_mbps=float(np.mean(sum_rates[m])) / 1.0e6,
                sum_rate_std_mbps=float(np.std(sum_rates[m])) / 1.0e6,
                user_rate_mean_mbps=float(np.mean(avg_rates[m])) / 1.0e6,
                user_rate_std_mbps=float(np.std(avg_rates[m])) / 1.0e6,
                min_user_rate_mean_mbps=float(np.mean(min_rates[m])) / 1.0e6,
                jains_fairness_mean=float(np.mean(fairness[m])),
                jains_fairness_std=float(np.std(fairness[m])),
                outage_probability_mean=float(np.mean(outages[m])),
                slot_utilization_mean=float(np.mean(utilizations[m])) * 100.0,
                inference_time_ms=float(np.mean(latencies[m])),
                validity_rate=float(np.mean(validities[m])) * 100.0,
            )

        self.method_ber_curves = {
            method_name: [
                {
                    "snr_db": float(snr_db),
                    "ber": float(errors / max(int(ber_total_bits[snr_idx]), 1)),
                }
                for snr_idx, (snr_db, errors) in enumerate(zip(ber_snr_db, error_counts))
            ]
            for method_name, error_counts in ber_error_counts.items()
        }
        return results

    @staticmethod
    def _pairing_f1_score(predicted: PairingPlan, reference: PairingPlan) -> float:
        """Compute undirected pair F1 against the oracle pairing for one scenario."""
        predicted_pairs = {
            tuple(sorted((pair.user1_id, pair.user2_id))) for pair in predicted.pairs
        }
        reference_pairs = {
            tuple(sorted((pair.user1_id, pair.user2_id))) for pair in reference.pairs
        }
        true_positives = len(predicted_pairs & reference_pairs)
        denominator = len(predicted_pairs) + len(reference_pairs)
        return 2.0 * true_positives / denominator if denominator else 0.0

    def _record_metric(
        self,
        method: str,
        sched: NetworkScheduleResult,
        latency_ms: float,
        is_valid: bool,
        sum_rates: Dict[str, List[float]],
        avg_rates: Dict[str, List[float]],
        min_rates: Dict[str, List[float]],
        fairness: Dict[str, List[float]],
        outages: Dict[str, List[float]],
        utilizations: Dict[str, List[float]],
        latencies: Dict[str, List[float]],
        validities: Dict[str, List[bool]],
    ) -> None:
        user_rates_arr = list(sched.per_user_rates_bps.values())
        sum_rates[method].append(sched.total_sum_rate_bps)
        avg_rates[method].append(float(np.mean(user_rates_arr)) if user_rates_arr else 0.0)
        min_rates[method].append(float(np.min(user_rates_arr)) if user_rates_arr else 0.0)
        fairness[method].append(sched.jains_fairness)
        outages[method].append(sched.outage_probability)
        utilizations[method].append(sched.slot_utilization)
        latencies[method].append(latency_ms)
        validities[method].append(is_valid)

    def evaluate_scma_snr_sweep(
        self,
        snr_db_range: List[float] = [-5.0, 0.0, 5.0, 10.0, 15.0, 20.0],
        num_blocks: int = 100,
    ) -> List[SCMAResult]:
        """Evaluate SCMA MPA detector across SNR levels."""
        dummy_gains = np.ones(self.scma_engine.J)
        scma_results: List[SCMAResult] = []
        for snr in snr_db_range:
            res = self.scma_engine.simulate_transmission(
                channel_gains=dummy_gains,
                snr_db=snr,
                num_blocks=num_blocks,
                bandwidth_hz=self.cfg.network.total_bandwidth,
            )
            scma_results.append(res)
        return scma_results

    def to_dataframe(self, benchmark_results: Dict[str, MethodBenchmarkMetrics]) -> pd.DataFrame:
        """Convert benchmark metrics to structured DataFrame."""
        rows = [asdict(m) for m in benchmark_results.values()]
        df = pd.DataFrame(rows)
        return df
