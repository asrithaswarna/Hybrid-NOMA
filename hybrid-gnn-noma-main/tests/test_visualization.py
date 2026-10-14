import matplotlib.pyplot as plt

from src.training import TrainingHistory
from src.visualization import Visualizer
from src.noma_scma import SCMAResult
from src.evaluation import MethodBenchmarkMetrics


def test_visualizer_shows_plot_after_saving(tmp_path, monkeypatch):
    visualizer = Visualizer(output_dir=str(tmp_path))
    history = TrainingHistory(
        train_losses=[1.0, 0.5],
        val_losses=[1.1, 0.6],
        best_val_loss=0.6,
        epochs_trained=2,
    )

    calls = []
    monkeypatch.setattr(visualizer, "_show_if_possible", lambda fig, image_path=None: calls.append((fig, image_path)))

    output_path = visualizer.plot_training_history(history, filename="loss.png")

    assert output_path.endswith("loss.png")
    assert len(calls) == 1
    assert calls[0][1].endswith("loss.png")


def test_ber_plots_are_independent_and_labeled(tmp_path, monkeypatch):
    visualizer = Visualizer(output_dir=str(tmp_path))
    figures = []
    monkeypatch.setattr(
        visualizer,
        "_show_if_possible",
        lambda fig, image_path=None: figures.append((fig, image_path)),
    )
    snrs = [0.0, 5.0]
    scma_results = [
        SCMAResult(6, 4, 1.5, snr, 0.3, ber, 1.0, [], 4)
        for snr, ber in zip(snrs, [0.2, 0.1])
    ]
    visualizer.plot_scma_ber_curve(scma_results)

    methods = {
        "Random Pairing PD-NOMA": "Random Pairing vs BER",
        "Near-Far Channel Gain NOMA": "Near-Far vs BER",
        "Greedy Sum-Rate NOMA": "Greedy Sum-Rate vs BER",
        "Optimization Oracle": "Optimization vs BER",
        "Proposed GNN-Based NOMA": "GNN vs BER",
    }
    results = {
        method: [{"snr_db": snr, "ber": ber} for snr, ber in zip(snrs, [0.25, 0.125])]
        for method in methods
    }
    paths = visualizer.plot_method_ber_curves(results)

    assert len(figures) == 6
    assert len(paths) == 5
    for fig, _ in figures:
        ax = fig.axes[0]
        assert ax.get_xlabel() == "SNR (dB)"
        assert ax.get_ylabel() == "BER"
        assert len(ax.lines) == 1
    assert figures[0][0].axes[0].get_title() == "SCMA BER Curve"
    assert {
        fig.axes[0].get_title() for fig, _ in figures[1:]
    } == set(methods.values())
    for fig, _ in figures:
        plt.close(fig)


def test_benchmark_plot_includes_outage_probability_bars(tmp_path, monkeypatch):
    visualizer = Visualizer(output_dir=str(tmp_path))
    figures = []
    monkeypatch.setattr(
        visualizer,
        "_show_if_possible",
        lambda fig, image_path=None: figures.append(fig),
    )

    def metrics(method, fairness, outage):
        return MethodBenchmarkMetrics(
            method_name=method,
            pairing_f1_score_mean=0.5,
            sum_rate_mean_mbps=12.0,
            sum_rate_std_mbps=1.0,
            user_rate_mean_mbps=2.0,
            user_rate_std_mbps=0.2,
            min_user_rate_mean_mbps=0.5,
            jains_fairness_mean=fairness,
            jains_fairness_std=0.05,
            outage_probability_mean=outage,
            slot_utilization_mean=90.0,
            inference_time_ms=1.0,
            validity_rate=100.0,
        )

    visualizer.plot_benchmark_comparison({
        "Method A": metrics("Method A", 0.75, 0.25),
        "Method B": metrics("Method B", 0.60, 0.40),
    })

    fig = figures[0]
    assert len(fig.axes[0].patches) == 2
    secondary_bars = fig.axes[1].patches
    assert len(secondary_bars) == 4
    assert [bar.get_height() for bar in secondary_bars] == [0.75, 0.60, 0.25, 0.40]
    legend_labels = [text.get_text() for text in fig.axes[0].get_legend().get_texts()]
    assert "Outage Probability" in legend_labels
    plt.close(fig)
