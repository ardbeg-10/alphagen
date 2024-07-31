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
        n_splits = 2
        X = torch.stack([self._calc_alpha(expr) for expr in exprs], dim=-1).cpu().numpy()
        X = X.reshape(-1, X.shape[-1])
        y = column_or_1d(self.target_value.cpu().numpy().reshape(-1, 1))

        threshold = 3

        # z_scores = np.abs(stats.zscore(X))
        # # 设置阈值
        #
        #
        # # 将超过阈值的值设为阈值
        # X = np.where(z_scores > threshold, np.sign(X) * threshold, X)

        X = np.where(X > threshold, threshold, X)
        X = np.where(X < -threshold, -threshold, X)

        print("\n\n")

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        best_model = None
        best_score = float('inf')

        for train_index, valid_index in kf.split(X):
            X_train, X_valid = X[train_index], X[valid_index]
            y_train, y_valid = y[train_index], y[valid_index]

            model = LGBMRegressor(
                boosting_type='gbdt',
                num_leaves=31,
                learning_rate=0.02,
                n_estimators=500,
                max_depth=8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                min_split_gain=0.005,
                subsample_freq=5,
                colsample_bytree=0.8879,
                subsample=0.8789,
                min_child_samples=20
            )

            model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                eval_metric='l1',
            )

            score = model.best_score_['valid_0']['l1']
            # 计算训练误差
            y_train_pred = model.predict(X_train)
            train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)

            # 计算测试误差
            y_test_pred = model.predict(X_valid)
            test_rmse = mean_squared_error(y_valid, y_test_pred, squared=False)

            print('\n')
            print('\n')
            print(f'Train RMSE: {train_rmse}')
            print(f'Test RMSE: {test_rmse}')
            print('\n')
            print('\n')

            best_model = model

            if score < best_score:
                best_score = score
                best_model = model

        # model_dir = os.path.expanduser('~/models')
        # if not os.path.exists(model_dir):
        #     os.makedirs(model_dir)
        #
        # model_path = os.path.join(model_dir, 'model.txt')
        # best_model.booster_.save_model(model_path)

        return best_model

    def unstack(self, value: np.ndarray) -> np.ndarray:
        return value.reshape(self.data.n_days, self.data.n_stocks)
