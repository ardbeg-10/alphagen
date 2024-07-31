from typing import Optional, TypeVar, Callable, Optional, Tuple
import os
import pickle
import warnings
import pandas as pd
from pandas import DataFrame
from qlib.backtest import backtest, executor as exec
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.report.analysis_position import report_graph
from alphagen.data.expression import *

from alphagen_qlib.stock_data import StockData
from alphagen_generic.features import *
from alphagen_qlib.strategy import TopKSwapNStrategy

import json

_T = TypeVar("_T")

import numpy as np
from lightgbm import Booster
from alphagen_qlib.utils import load_alpha_pool_by_path, load_dt_model_by_path
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen.utils.pytorch_utils import normalize_by_day
from alphagen.data.calculator import AlphaCalculator
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr


def _create_parents(path: str) -> None:
    dir = os.path.dirname(path)
    if dir != "":
        os.makedirs(dir, exist_ok=True)


def write_all_text(path: str, text: str) -> None:
    _create_parents(path)
    with open(path, "w") as f:
        f.write(text)


def dump_pickle(path: str,
                factory: Callable[[], _T],
                invalidate_cache: bool = False) -> Optional[_T]:
    if invalidate_cache or not os.path.exists(path):
        _create_parents(path)
        obj = factory()
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        return obj


class BacktestResult(dict):
    sharpe: float
    annual_return: float
    max_drawdown: float
    information_ratio: float
    annual_excess_return: float
    excess_max_drawdown: float


class QlibBacktest:
    def __init__(
        self,
        benchmark: str = "SH000300",
        top_k: int = 30,
        n_drop: Optional[int] = None,
        deal: str = "close",
        open_cost: float = 0.0015,
        close_cost: float = 0.0015,
        min_cost: float = 5,
    ):
        self._benchmark = benchmark
        self._top_k = top_k
        self._n_drop = n_drop if n_drop is not None else top_k
        self._deal_price = deal
        self._open_cost = open_cost
        self._close_cost = close_cost
        self._min_cost = min_cost

    def run(
        self,
        prediction: pd.Series,
        output_prefix: Optional[str] = '/backtest',
        return_report: bool = False
    ) -> BacktestResult:
        prediction = prediction.sort_index()
        index: pd.MultiIndex = prediction.index.remove_unused_levels()  # type: ignore
        dates = index.levels[0]

        def backtest_impl(last: int = -1):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                strategy=TopKSwapNStrategy(
                    K=self._top_k,
                    n_swap=self._n_drop,
                    signal=prediction,
                    min_hold_days=1,
                    only_tradable=True,
                )
                executor=exec.SimulatorExecutor(
                    time_per_step="day",
                    generate_portfolio_metrics=True
                )
                return backtest(
                    strategy=strategy,
                    executor=executor,
                    start_time=dates[0],
                    end_time=dates[last],
                    account=100_000_000,
                    benchmark=self._benchmark,
                    exchange_kwargs={
                        "limit_threshold": 0.095,
                        "deal_price": self._deal_price,
                        "open_cost": self._open_cost,
                        "close_cost": self._close_cost,
                        "min_cost": self._min_cost,
                    }
                )[0]

        try:
            portfolio_metric = backtest_impl()
        except IndexError:
            print("Cannot backtest till the last day, trying again with one less day")
            portfolio_metric = backtest_impl(-2)

        report, _ = portfolio_metric["1day"]    # type: ignore
        result = self._analyze_report(report)
        graph = report_graph(report, show_notebook=False)[0]

        if output_prefix is not None:
            dump_pickle(output_prefix + "/report.pkl", lambda: report, True)
            dump_pickle(output_prefix + "/graph.pkl", lambda: graph, True)
            result_json = json.dumps(result, indent=4)
            write_all_text(output_prefix + "/result.json", result_json)

        print(report)
        print(result)
        return report if return_report else result

    def _analyze_report(self, report: pd.DataFrame) -> BacktestResult:
        excess = risk_analysis(report["return"] - report["bench"] - report["cost"])["risk"]
        returns = risk_analysis(report["return"] - report["cost"])["risk"]

        def loc(series: pd.Series, field: str) -> float:
            return series.loc[field]    # type: ignore

        return BacktestResult(
            sharpe=loc(returns, "information_ratio"),
            annual_return=loc(returns, "annualized_return"),
            max_drawdown=loc(returns, "max_drawdown"),
            information_ratio=loc(excess, "information_ratio"),
            annual_excess_return=loc(excess, "annualized_return"),
            excess_max_drawdown=loc(excess, "max_drawdown"),
        )

def make_ensemble_alpha(exprs: List[Expression], model: Booster) -> Tensor:
    n = len(exprs)
    return torch.from_numpy(predict(model, exprs)).to(data.device)

def predict(model: Booster, exprs: List[Expression]) -> np.ndarray:
    X = torch.stack([_calc_alpha(expr) for expr in exprs], dim=-1).cpu().numpy()
    X = X.reshape(-1, X.shape[-1])
    val = model.predict(X)
    return unstack(val)

def unstack(value: np.ndarray) -> np.ndarray:
    return value.reshape(data.n_days, data.n_stocks)

def _calc_alpha(expr: Expression) -> Tensor:
    return normalize_by_day(expr.evaluate(data))

def _calc_ICs(value1: Tensor, value2: Tensor) -> Tensor:
    return batch_pearsonr(value1, value2)

def _calc_IC(value1: Tensor, value2: Tensor) -> float:
    return batch_pearsonr(value1, value2).mean().item()

def _calc_IR(value1: Tensor, value2: Tensor) -> float:
    ICs = _calc_ICs(value1, value2)
    IC_mean = ICs.mean().item()
    IC_std = ICs.std().item()
    epsilon = 1e-10  # 防止除以零的小值
    IR = IC_mean / (IC_std - epsilon)
    return IR

def test_ensemble(exprs: List[Expression], model: Booster, calculator: AlphaCalculator) -> Tuple[float, float]:
    return calc_pool_all_ret(exprs, calculator.target_value, model)

def calc_pool_all_ret(exprs: List[Expression], target: Tensor, model: Booster) -> Tuple[float, float]:
    with torch.no_grad():
        ensemble_value = make_ensemble_alpha(exprs, model)
        return _calc_IC(ensemble_value, target), _calc_IR(ensemble_value, target)

from alphagen_qlib.utils import load_alpha_pool_by_path
from alphagen_qlib.calculator import QLibStockDataCalculator


if __name__ == "__main__":
    POOL_PATH = 'C:\Users\liush\Downloads\量化研究\alphagen\model\10240_steps_pool.json'
    DT_PATH = 'C:\Users\liush\Downloads\量化研究\alphagen\model\10240_steps_dt.txt'

    qlib_backtest = QlibBacktest()

    data = StockData(instrument='csi500',
                     start_time='2024-05-18',
                     end_time='2024-06-01')

    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -20) / close - 1

    calculator = QLibStockDataCalculator(data=data, target=target)
    exprs, _ = load_alpha_pool_by_path(POOL_PATH)
    booster = load_dt_model_by_path(DT_PATH)

    ensemble_alpha = make_ensemble_alpha(exprs, booster)
    df = data.make_dataframe(ensemble_alpha)

    print(test_ensemble(exprs, booster, calculator))

    qlib_backtest.run(df)


# set RUST_BACKTRACE=1