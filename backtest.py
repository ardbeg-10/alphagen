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

from alphagen_qlib.utils import load_alpha_pool_by_path
from alphagen_qlib.calculator import QLibStockDataCalculator

POOL_PATH = '/save/new_csi300_200_1_20240719010851/20480_steps_pool.json'

if __name__ == "__main__":
    qlib_backtest = QlibBacktest()

    data = StockData(instrument='csi300',
                     start_time='2023-01-01',
                     end_time='2024-06-01')

    calculator = QLibStockDataCalculator(data=data, target=None)
    exprs, weights = load_alpha_pool_by_path(POOL_PATH)

    ensemble_alpha = calculator.make_ensemble_alpha(exprs, weights)
    df = data.make_dataframe(ensemble_alpha)

    qlib_backtest.run(df)


# set RUST_BACKTRACE=1