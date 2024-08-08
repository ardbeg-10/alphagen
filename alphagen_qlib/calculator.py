import os
from typing import List, Optional, Tuple

from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
from sklearn.utils import column_or_1d
from torch import Tensor
import torch
from torch.nn.modules import fold

from alphagen.data.calculator import AlphaCalculator
from alphagen.data.expression import Expression
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr
from alphagen.utils.pytorch_utils import normalize_by_day
from alphagen_qlib.stock_data import StockData
import lightgbm as lgb
from lightgbm import LGBMRegressor

from sklearn.model_selection import train_test_split


class QLibStockDataCalculator(AlphaCalculator):
    def __init__(self, data: StockData, target: Optional[Expression]):
        self.data = data

        if target is None: # Combination-only mode
            self.target_value = None
        else:
            self.target_value = normalize_by_day(target.evaluate(self.data))

    def _calc_alpha(self, expr: Expression) -> Tensor:
        return normalize_by_day(expr.evaluate(self.data))

    def _calc_ICs(self, value1: Tensor, value2: Tensor) -> Tensor:
        return batch_pearsonr(value1, value2)

    def _calc_IR(self, value1: Tensor, value2: Tensor) -> float:
        ICs = self._calc_ICs(value1, value2)
        IC_mean = ICs.mean().item()
        IC_std = ICs.std().item()
        epsilon = 1e-10  # 防止除以零的小值
        IR = IC_mean / (IC_std - epsilon)
        return IR

    def _calc_IC(self, value1: Tensor, value2: Tensor) -> float:
        return batch_pearsonr(value1, value2).mean().item()

    def _calc_rIC(self, value1: Tensor, value2: Tensor) -> float:
        return batch_spearmanr(value1, value2).mean().item()

    def make_ensemble_alpha(self, exprs: List[Expression], model: LGBMRegressor) -> Tensor:
        n = len(exprs)
        return torch.from_numpy(self.predict(model, exprs)).to(self.data.device)

    def calc_single_IC_ret(self, expr: Expression) -> float:
        value = self._calc_alpha(expr)
        return self._calc_IC(value, self.target_value)

    def calc_single_rIC_ret(self, expr: Expression) -> float:
        value = self._calc_alpha(expr)
        return self._calc_rIC(value, self.target_value)

    def calc_single_all_ret(self, expr: Expression) -> Tuple[float, float]:
        value = self._calc_alpha(expr)
        return self._calc_IC(value, self.target_value), self._calc_rIC(value, self.target_value)

    def calc_mutual_IC(self, expr1: Expression, expr2: Expression) -> float:
        value1, value2 = self._calc_alpha(expr1), self._calc_alpha(expr2)
        return self._calc_IC(value1, value2)

    def calc_pool_IC_ret(self, exprs: List[Expression], model: LGBMRegressor) -> float:
        with torch.no_grad():
            ensemble_value = self.make_ensemble_alpha(exprs, model)
            val = self._calc_IC(ensemble_value, self.target_value)
            ir = self._calc_IR(ensemble_value, self.target_value)
            print(f"Combined IC: {val}")
            print(f"Combined IR: {ir}")
            print(model.feature_importances_)
            return val

    def calc_pool_IR_ret(self, exprs: List[Expression], model: LGBMRegressor) -> float:
        with torch.no_grad():
            ensemble_value = self.make_ensemble_alpha(exprs, model)
            val = self._calc_IC(ensemble_value, self.target_value)
            ir = self._calc_IR(ensemble_value, self.target_value)
            print(f"Combined IC: {val}")
            print(f"Combined IR: {ir}")
            print(model.feature_importances_)
            return ir

    def calc_pool_rIC_ret(self, exprs: List[Expression], model: LGBMRegressor) -> float:
        with torch.no_grad():
            ensemble_value = self.make_ensemble_alpha(exprs, model)
            return self._calc_rIC(ensemble_value, self.target_value)

    def calc_pool_all_ret(self, exprs: List[Expression], model: LGBMRegressor) -> Tuple[float, float, float]:
        with torch.no_grad():
            ensemble_value = self.make_ensemble_alpha(exprs, model)
            return self._calc_IC(ensemble_value, self.target_value), self._calc_rIC(ensemble_value, self.target_value), self._calc_IR(ensemble_value, self.target_value)

    def predict(self, model: LGBMRegressor, exprs: List[Expression]) -> np.ndarray:
        X = torch.stack([self._calc_alpha(expr) for expr in exprs], dim=-1).cpu().numpy()
        X = X.reshape(-1, X.shape[-1])
        val = model.booster_.predict(X)
        return self.unstack(val)

    def train_lgbm(self, exprs: List[Expression]) -> LGBMRegressor:
        X = torch.stack([self._calc_alpha(expr) for expr in exprs], dim=-1).cpu().numpy()
        X = X.reshape(-1, X.shape[-1])
        y = column_or_1d(self.target_value.cpu().numpy().reshape(-1, 1))

        threshold = 3
        X = np.where(X > threshold, threshold, X)
        X = np.where(X < -threshold, -threshold, X)

        model = LGBMRegressor(
            boosting_type='dart',
            metric="huber",
            num_iterations=20,
            learning_rate=0.3,
            num_leaves=31,
            max_depth=6,
            reg_alpha=0.005,
            subsample_freq=4,
            subsample=0.9,
            feature_fraction=0.45,
            min_child_samples=1000,
            drop_rate=0.45,
            max_drop=30,
            skip_drop=0.65,
            max_bin=65,
            extra_trees=True
        )

        model.fit(X, y, eval_metric='l1')

        # Calculate training error
        y_train_pred = model.predict(X)
        train_rmse = mean_squared_error(y, y_train_pred, squared=False)

        print(f'Train RMSE: {train_rmse} \n\n')

        return model

    def unstack(self, value: np.ndarray) -> np.ndarray:
        return value.reshape(self.data.n_days, self.data.n_stocks)
