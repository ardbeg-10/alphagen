from itertools import count
import math
from typing import List, Optional, Tuple, Set
from abc import ABCMeta, abstractmethod

import numpy as np
import torch
from lightgbm import LGBMRegressor
from torch import Tensor
from alphagen.data.calculator import AlphaCalculator

from alphagen.data.expression import Expression
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr
from alphagen.utils.pytorch_utils import masked_mean_std
from alphagen_qlib.stock_data import StockData


class AlphaPoolBase(metaclass=ABCMeta):
    def __init__(
        self,
        capacity: int,
        calculator: AlphaCalculator,
        device: torch.device = torch.device('cpu')
    ):
        self.capacity = capacity
        self.calculator = calculator
        self.device = device

    @abstractmethod
    def to_dict(self) -> dict: ...

    @abstractmethod
    def try_new_expr(self, expr: Expression) -> float: ...

    @abstractmethod
    def test_ensemble(self, calculator: AlphaCalculator) -> Tuple[float, float, float]: ...

    @abstractmethod
    def get_dt_model(self) -> LGBMRegressor: ...


class AlphaPool(AlphaPoolBase):
    def __init__(
        self,
        capacity: int,
        calculator: AlphaCalculator,
        ic_lower_bound: Optional[float] = None,
        l1_alpha: float = 5e-3,
        device: torch.device = torch.device('cpu')
    ):
        super().__init__(capacity, calculator, device)

        self.size: int = 0
        self.exprs: List[Optional[Expression]] = [None for _ in range(capacity + 1)]
        self.single_ics: np.ndarray = np.zeros(capacity + 1)
        # self.mutual_ics: np.ndarray = np.identity(capacity + 1)
        self.model: LGBMRegressor = None  # type: ignore
        self.best_ic_ret: float = -1.

        self.ic_lower_bound = ic_lower_bound or -1.
        self.l1_alpha = l1_alpha

        self.eval_cnt = 0

    @property
    def state(self) -> dict:
        return {
            "exprs": list(self.exprs[:self.size]),
            "ics_ret": list(self.single_ics[:self.size]),
            "feature_importances": list(self.model.feature_importances_),
            "best_ic_ret": self.best_ic_ret
        }

    def to_dict(self) -> dict:
        feature_importances = self.model.feature_importances_.tolist() if self.size > 0 else []
        return {
            "exprs": [str(expr) for expr in self.exprs[:self.size]],
            "feature_importances": feature_importances,
        }

    def get_dt_model(self) -> LGBMRegressor:
        return self.model

    def try_new_expr(self, expr: Expression) -> float:
        ic_ret, ic_mut = self._calc_ics(expr, ic_mut_threshold=0.99)
        if ic_ret is None or np.isnan(ic_ret):
            return 0.

        self._add_factor(expr, ic_ret)
        self.train_lgbm()
        if self.size > 1:
            self._pop()
            self.model = self.train_lgbm()

        new_ic_ret = self.evaluate_ensemble()
        increment = new_ic_ret - self.best_ic_ret
        if increment > 0:
            self.best_ic_ret = new_ic_ret
        self.eval_cnt += 1
        return new_ic_ret

    def train_lgbm(self) -> LGBMRegressor:
        self.model = self.calculator.train_lgbm(self.exprs[:self.size])
        return self.model

    def force_load_exprs(self, exprs: List[Expression]) -> None:
        for expr in exprs:
            ic_ret, ic_mut = self._calc_ics(expr, ic_mut_threshold=None)
            assert ic_ret is not None and ic_mut is not None
            self._add_factor(expr, ic_ret)
            assert self.size <= self.capacity
        self._optimize(alpha=self.l1_alpha, lr=5e-4, n_iter=500)

    def test_ensemble(self, calculator: AlphaCalculator) -> Tuple[float, float, float]:
        ic, rank_ic, ir = calculator.calc_pool_all_ret(self.exprs[:self.size], self.model)
        return ic, rank_ic, ir

    def evaluate_ensemble(self) -> float:
        ic = self.calculator.calc_pool_IR_ret(self.exprs[:self.size], self.model)
        return ic

    @property
    def _under_thres_alpha(self) -> bool:
        if self.ic_lower_bound is None or self.size > 1:
            return False
        return self.size == 0 or abs(self.single_ics[0]) < self.ic_lower_bound

    def _calc_ics(
        self,
        expr: Expression,
        ic_mut_threshold: Optional[float] = None
    ) -> Tuple[float, Optional[List[float]]]:
        single_ic = self.calculator.calc_single_IC_ret(expr)
        if not self._under_thres_alpha and single_ic < self.ic_lower_bound:
            return single_ic, None

        mutual_ics = []
        for i in range(self.size):
            mutual_ic = self.calculator.calc_mutual_IC(expr, self.exprs[i])
            if ic_mut_threshold is not None and mutual_ic > ic_mut_threshold:
                return single_ic, None
            mutual_ics.append(mutual_ic)

        return single_ic, mutual_ics

    def _add_factor(
        self,
        expr: Expression,
        ic_ret: float,
    ):
        if self._under_thres_alpha and self.size == 1:
            self._pop()
        n = self.size
        self.exprs[n] = expr
        self.single_ics[n] = ic_ret
        self.size += 1

    def _pop(self) -> None:
        if self.size <= self.capacity:
            return
        idx = np.argmin(np.abs(self.model.feature_importances_))
        self._swap_idx(idx, self.capacity)
        self.size = self.capacity

    def _swap_idx(self, i, j) -> None:
        if i == j:
            return
        self.exprs[i], self.exprs[j] = self.exprs[j], self.exprs[i]
        self.single_ics[i], self.single_ics[j] = self.single_ics[j], self.single_ics[i]
