from abc import ABCMeta, abstractmethod
from typing import List, Tuple

from torch import Tensor

from alphagen.data.expression import Expression
from lightgbm import LGBMRegressor

class AlphaCalculator(metaclass=ABCMeta):
    @abstractmethod
    def calc_single_IC_ret(self, expr: Expression) -> float:
        'Calculate IC between a single alpha and a predefined target.'

    @abstractmethod
    def calc_single_rIC_ret(self, expr: Expression) -> float:
        'Calculate Rank IC between a single alpha and a predefined target.'

    @abstractmethod
    def calc_single_all_ret(self, expr: Expression) -> Tuple[float, float]:
        'Calculate both IC and Rank IC between a single alpha and a predefined target.'

    @abstractmethod
    def calc_mutual_IC(self, expr1: Expression, expr2: Expression) -> float:
        'Calculate IC between two alphas.'

    @abstractmethod
    def calc_pool_IC_ret(self, exprs: List[Expression], model: LGBMRegressor) -> float:
        'First combine the alphas linearly,'
        'then Calculate IC between the linear combination and a predefined target.'

    @abstractmethod
    def calc_pool_rIC_ret(self, exprs: List[Expression], model: LGBMRegressor) -> float:
        'First combine the alphas linearly,'
        'then Calculate Rank IC between the linear combination and a predefined target.'

    @abstractmethod
    def calc_pool_all_ret(self, exprs: List[Expression], model: LGBMRegressor) -> Tuple[float, float, float]:
        'First combine the alphas linearly,'
        'then Calculate both IC, Rank IC, and ICIR between the linear combination and a predefined target.'

    @abstractmethod
    def train_lgbm(self, exprs: List[Expression]) -> LGBMRegressor:
        'Train a LightGBM model with the given parameters.'
        'Return the trained model.'