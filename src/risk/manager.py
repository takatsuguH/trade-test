"""リスク管理モジュール"""
from dataclasses import dataclass


@dataclass
class RiskManager:
    stop_loss_pct: float = 5.0      # 損切りライン（%）
    take_profit_pct: float = 10.0   # 利確ライン（%）
    max_position_pct: float = 100.0 # 最大投資割合（%）
    rebuy_dip_pct: float = 0.0      # 買い戻し最小下落率（%）0=無効

    @property
    def stop_loss(self) -> float:
        return self.stop_loss_pct / 100

    @property
    def take_profit(self) -> float:
        return self.take_profit_pct / 100

    @property
    def max_position(self) -> float:
        return self.max_position_pct / 100

    @property
    def rebuy_dip(self) -> float:
        return self.rebuy_dip_pct / 100
