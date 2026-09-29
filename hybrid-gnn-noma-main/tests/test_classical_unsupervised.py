from src.classical_models import compare_regressors, predict_rate
from src.config import ProjectConfig
from src.data_pipeline import generate_synthetic_scenarios, scenarios_to_frame
from src.unsupervised import run_clustering


def test_classical_models_and_clustering(tmp_path):
    cfg = ProjectConfig()
    cfg.network.num_users = 4
    scenarios = generate_synthetic_scenarios(cfg, num_samples=12)
    frame = scenarios_to_frame(scenarios, cfg)
    train = frame[frame["scenario_id"] < 8]
    test = frame[frame["scenario_id"] >= 8]
    result = compare_regressors(train, test, str(tmp_path / "models"))
    assert set(result["model"]) == {"Linear Regression", "Ridge Regression", "Lasso Regression", "Decision Tree", "Random Forest"}
    artifact = tmp_path / "models" / "random_forest.joblib"
    predictions = predict_rate(str(artifact), test)
    assert len(predictions) == len(test)
    clustering = run_clustering(frame, str(tmp_path / "clusters"))
    assert len(clustering["assignments"]) == len(frame)
    assert len(clustering["explained_variance"]) == 2
