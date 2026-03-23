from datamodel import OrderDepth, UserId, TradingState, Order, Trade
from typing import List, Dict, Tuple
import json
import collections

# --- 全局常量 ---
LIMIT = 80
PRODUCT = "TOMATOES"
LAMBDA_TIGHT = 0.35      # Tight regime alpha 系数
ALPHA_CAP = 3.5          # 偏离度上限 (Ticks)
TRADE_LOOKBACK = 10      # Trade VWAP 窗口
ADVERSE_THRESHOLD = 3    # 连续逆向成交阈值
COOLDOWN_TICKS = 10      # 冷却时长


class Trader:
    """
    TOMATOES Elite Market-Maker
    - 修复了函数定义 Bug
    - 集成 L2 Microprice Bias (Lambda-weighted)
    - 集成 Trade-to-Mid Dislocation
    - 集成 Adverse Selection 冷却机制
    """

    def __init__(self):
        # 用于计算 Trade Dislocation 的本地缓存
        self.trade_prices = collections.deque(maxlen=TRADE_LOOKBACK)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _extract_book(order_depth: OrderDepth):
        bids = sorted([(p, v) for p, v in order_depth.buy_orders.items()], key=lambda x: -x[0])
        asks = sorted([(p, abs(v)) for p, v in order_depth.sell_orders.items()], key=lambda x: x[0])
        return bids, asks

    @staticmethod
    def _compute_microprice_bias(bids, asks) -> float:
        """逻辑 2) 计算相对于 Mid 的偏置: bias = lambda * (mp2 - mid)"""
        bid1_p, bid1_v = bids[0]
        ask1_p, ask1_v = asks[0]
        mid = (bid1_p + ask1_p) / 2.0
        spread = ask1_p - bid1_p

        # L1 imbalance
        total_l1 = bid1_v + ask1_v
        l1_imb = (bid1_v - ask1_v) / total_l1 if total_l1 > 0 else 0.0

        # L2 Anchor (如果有 L2 则用 L2, 否则 fallback)
        if len(bids) >= 2 and len(asks) >= 2:
            l2_mid = (bids[1][0] + asks[1][0]) / 2.0
        else:
            l2_mid = mid

        microprice = l2_mid + l1_imb * (spread / 2.0)
        raw_bias = LAMBDA_TIGHT * (microprice - mid)
        return max(min(raw_bias, ALPHA_CAP), -ALPHA_CAP)

    def _get_trade_dislocation(self, mid, market_trades: List[Trade]) -> float:
        """逻辑 5) Trade-price-to-mid dislocation alpha"""
        for t in market_trades:
            self.trade_prices.append(t.price)
        if not self.trade_prices:
            return 0.0
        vwap = sum(self.trade_prices) / len(self.trade_prices)
        return vwap - mid

    @staticmethod
    def _inventory_params(position: int) -> Tuple[float, int, int, int, int]:
        # 维持原有的分层库存逻辑
        ap = abs(position)
        sign = 1 if position > 0 else -1
        if ap <= 40:
            return (0.0, 20, 20, 0, 0)
        elif ap <= 60:
            frac = (ap - 40) / 20.0
            skew = -sign * frac * 3.0
            if position > 0: return (skew, max(3, int(20 * (1 - frac))), 25, 0, 0)
            else: return (skew, 25, max(3, int(20 * (1 - frac))), 0, 0)
        elif ap <= 70:
            frac = (ap - 60) / 10.0
            skew = -sign * (3.0 + frac * 3.0)
            if position > 0: return (skew, 0, 30, 2, 0)
            else: return (skew, 30, 0, 0, 2)
        else:
            frac = min((ap - 70) / 10.0, 1.0)
            skew = -sign * (6.0 + frac * 4.0)
            if position > 0: return (skew, 0, 20, 4, 0)
            else: return (skew, 20, 0, 0, 4)

    # ------------------------------------------------------------------ run
    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0

        # 1. 状态恢复 (使用简写键名节省传输长度)
        trader_data = json.loads(state.traderData) if state.traderData else {}
        b_adv = trader_data.get("b_a", 0)
        s_adv = trader_data.get("s_a", 0)
        b_cd = trader_data.get("b_c", 0)
        s_cd = trader_data.get("s_c", 0)
        prev_m = trader_data.get("p_m", None)
        last_a = trader_data.get("l_a", None)

        if PRODUCT not in state.order_depths:
            return result, conversions, json.dumps(trader_data)

        order_depth = state.order_depths[PRODUCT]
        bids, asks = self._extract_book(order_depth)
        if not bids or not asks:
            return result, conversions, json.dumps(trader_data)

        curr_mid = (bids[0][0] + asks[0][0]) / 2.0
        position = state.position.get(PRODUCT, 0)

        # 2. Adverse Selection 冷却检测
        if prev_m is not None and last_a is not None:
            if last_a == "BUY" and curr_mid < prev_m: b_adv += 1
            elif last_a == "BUY" and curr_mid >= prev_m: b_adv = 0
            if last_a == "SELL" and curr_mid > prev_m: s_adv += 1
            elif last_a == "SELL" and curr_mid <= prev_m: s_adv = 0

        if b_adv >= ADVERSE_THRESHOLD: b_cd = COOLDOWN_TICKS; b_adv = 0
        if s_adv >= ADVERSE_THRESHOLD: s_cd = COOLDOWN_TICKS; s_adv = 0

        b_cd, s_cd = max(0, b_cd - 1), max(0, s_cd - 1)

        # 3. Alpha 偏置计算
        alpha_bias = self._compute_microprice_bias(bids, asks)
        trades_list = state.market_trades.get(PRODUCT, [])
        trade_dislo = self._get_trade_dislocation(curr_mid, trades_list)
        
        # 二级确认逻辑
        final_alpha = 0.0
        if (alpha_bias > 0 and trade_dislo > -0.2) or (alpha_bias < 0 and trade_dislo < 0.2):
            final_alpha = alpha_bias
        
        inv_skew, mm_buy_sz, mm_sell_sz, b_edge, s_edge = self._inventory_params(position)
        fair = curr_mid + final_alpha + inv_skew

        orders: List[Order] = []
        buy_cap = LIMIT - position
        sell_cap = LIMIT + position
        current_tick_act = None

        # PHASE 1 — TAKE (带冷却)
        tight_bonus = 1 if (asks[0][0] - bids[0][0]) < 10 else 0
        if b_cd == 0:
            for p, v in asks:
                if buy_cap <= 0: break
                if p <= (fair - b_edge + tight_bonus):
                    orders.append(Order(PRODUCT, p, min(v, buy_cap)))
                    buy_cap -= min(v, buy_cap); current_tick_act = "BUY"
        if s_cd == 0:
            for p, v in bids:
                if sell_cap <= 0: break
                if p >= (fair + s_edge - tight_bonus):
                    orders.append(Order(PRODUCT, p, -min(v, sell_cap)))
                    sell_cap -= min(v, sell_cap); current_tick_act = "SELL"

        # PHASE 2 — POST (偏置报价)
        b_p = int(min(bids[0][0] if final_alpha >= 0 else bids[0][0] - 1, fair - 0.5))
        a_p = int(max(asks[0][0] if final_alpha <= 0 else asks[0][0] + 1, fair + 0.5))

        if buy_cap > 0 and mm_buy_sz > 0 and b_cd == 0:
            orders.append(Order(PRODUCT, b_p, min(mm_buy_sz, buy_cap)))
        if sell_cap > 0 and mm_sell_sz > 0 and s_cd == 0:
            orders.append(Order(PRODUCT, a_p, -min(mm_sell_sz, sell_cap)))

        # PHASE 3 — EMERGENCY OFFLOAD
        abs_pos = abs(position)
        if abs_pos > 60:
            max_c = min(5, int(2 + (abs_pos - 70) * 0.3)) if abs_pos > 70 else 2
            off_p = int(round(fair))
            if position > 0 and (sell_cap + position) > 0:
                p_p = max(min(off_p, bids[0][0]) if abs_pos > 70 else off_p, int(fair) - max_c)
                orders.append(Order(PRODUCT, p_p, -min(abs_pos - 40, sell_cap + position)))
            elif position < 0 and (buy_cap + abs(position)) > 0:
                p_p = min(max(off_p, asks[0][0]) if abs_pos > 70 else off_p, int(fair) + max_c)
                orders.append(Order(PRODUCT, p_p, min(abs_pos - 40, buy_cap + abs(position))))

        # 4. 状态保存
        trader_data.update({"b_a": b_adv, "s_a": s_adv, "b_c": b_cd, "s_c": s_cd, "p_m": curr_mid, "l_a": current_tick_act})
        return {PRODUCT: orders}, conversions, json.dumps(trader_data)