import pytest

np = pytest.importorskip('numpy')
scipy = pytest.importorskip('scipy')
sklearn = pytest.importorskip('sklearn')

from evaluate_bvqa_features_regression import logistic_func, compute_metrics
from scipy.stats import spearmanr, kendalltau, pearsonr
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error


def test_logistic_func_basic():
    X = np.array([0.0, 1.0, 2.0])
    result = logistic_func(X, 1.0, 0.0, 0.5, 1.0)
    expected = 0.0 + (1.0 - 0.0) / (1 + np.exp(-(X - 0.5) / 1.0))
    assert np.allclose(result, expected)


def test_compute_metrics_values():
    y = np.array([1, 2, 3, 4, 5], dtype=float)
    y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1], dtype=float)

    expected_srcc = spearmanr(y, y_pred)[0]
    try:
        expected_krcc = kendalltau(y, y_pred)[0]
    except Exception:
        expected_krcc = kendalltau(y, y_pred, method='asymptotic')[0]
    beta_init = [np.max(y), np.min(y), np.mean(y_pred), 0.5]
    popt, _ = curve_fit(logistic_func, y_pred, y, p0=beta_init, maxfev=int(1e8))
    y_pred_logistic = logistic_func(y_pred, *popt)
    expected_plcc = pearsonr(y, y_pred_logistic)[0]
    expected_rmse = np.sqrt(mean_squared_error(y, y_pred_logistic))

    metrics, y_log = compute_metrics(y_pred, y)

    assert np.allclose(metrics[0], expected_srcc)
    assert np.allclose(metrics[1], expected_krcc)
    assert np.allclose(metrics[2], expected_plcc)
    assert np.allclose(metrics[3], expected_rmse)
    assert np.allclose(y_log, y_pred_logistic)
