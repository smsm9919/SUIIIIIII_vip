# -*- coding: utf-8 -*-
"""
SUI ULTRA PRO AI BOT - الإصدار الذكي المتقدم المتكامل المتطور
• مجلس الإدارة الفائق الذكي مع 25 استراتيجية متقدمة
• نظام ركوب الترند الذكي المحترف لتحقيق أقصى ربح متتالي
• السكالب الفائق الذكي بأهداف متعددة محسوبة
• إدارة صفقات ذكية متكيفة مع قوة الترند
• نظام Footprint + Diagonal Order-Flow المتقدم
• Multi-Exchange Support: BingX & Bybit
• نظام مراكبة الأرباح الذكي
• SMART MONEY CONCEPTS (ICT) Integration
"""

import os, time, math, random, signal, sys, traceback, logging, json
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import ccxt
from flask import Flask, jsonify
from decimal import Decimal, ROUND_DOWN, InvalidOperation

from scipy import stats
from dataclasses import dataclass
from typing import List, Optional

try:
    from termcolor import colored
except Exception:
    def colored(t,*a,**k): return t

# =================== ENV / MODE ===================
EXCHANGE_NAME = os.getenv("EXCHANGE", "bingx").lower()

if EXCHANGE_NAME == "bybit":
    API_KEY = os.getenv("BYBIT_API_KEY", "")
    API_SECRET = os.getenv("BYBIT_API_SECRET", "")
else:
    API_KEY = os.getenv("BINGX_API_KEY", "")
    API_SECRET = os.getenv("BINGX_API_SECRET", "")

MODE_LIVE = bool(API_KEY and API_SECRET)
SELF_URL = os.getenv("SELF_URL", "") or os.getenv("RENDER_EXTERNAL_URL", "")
PORT = int(os.getenv("PORT", 5000))

# ==== Run mode / Logging toggles ====
LOG_LEGACY = False
LOG_ADDONS = True

# ==== Execution Switches ====
EXECUTE_ORDERS = True
SHADOW_MODE_DASHBOARD = False
DRY_RUN = False

# ==== Addon: Logging + Recovery Settings ====
BOT_VERSION = f"SUI ULTRA PRO AI v8.0 — {EXCHANGE_NAME.upper()} + SMART MONEY"
print("🚀 Booting:", BOT_VERSION, flush=True)

STATE_PATH = "./bot_state.json"
RESUME_ON_RESTART = True
RESUME_LOOKBACK_SECS = 60 * 60

# === Addons config ===
BOOKMAP_DEPTH = 50
BOOKMAP_TOPWALLS = 3
IMBALANCE_ALERT = 1.30

FLOW_WINDOW = 20
FLOW_SPIKE_Z = 1.60
CVD_SMOOTH = 8

# =================== SETTINGS ===================
SYMBOL     = os.getenv("SYMBOL", "SUI/USDT:USDT")
INTERVAL   = os.getenv("INTERVAL", "15m")
LEVERAGE   = int(os.getenv("LEVERAGE", 10))
RISK_ALLOC = float(os.getenv("RISK_ALLOC", 0.60))
POSITION_MODE = os.getenv("POSITION_MODE", "oneway")

# RF Settings - Optimized for SUI
RF_SOURCE = "close"
RF_PERIOD = int(os.getenv("RF_PERIOD", 18))
RF_MULT   = float(os.getenv("RF_MULT", 3.0))
RF_LIVE_ONLY = True
RF_HYST_BPS  = 6.0

# Indicators
RSI_LEN = 14
ADX_LEN = 14
ATR_LEN = 14

ENTRY_RF_ONLY = False
MAX_SPREAD_BPS = float(os.getenv("MAX_SPREAD_BPS", 6.0))

# Dynamic TP / trail - Optimized for SUI
TP1_PCT_BASE       = 0.45
TP1_CLOSE_FRAC     = 0.50
BREAKEVEN_AFTER    = 0.30
TRAIL_ACTIVATE_PCT = 1.20
ATR_TRAIL_MULT     = 1.8

TREND_TPS       = [0.50, 1.00, 1.80, 2.50, 3.50, 5.00, 7.00]
TREND_TP_FRACS  = [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.10]

# Dust guard
FINAL_CHUNK_QTY = float(os.getenv("FINAL_CHUNK_QTY", 50.0))
RESIDUAL_MIN_QTY = float(os.getenv("RESIDUAL_MIN_QTY", 10.0))

# Strict close
CLOSE_RETRY_ATTEMPTS = 6
CLOSE_VERIFY_WAIT_S  = 2.0

# Pacing
BASE_SLEEP   = 5
NEAR_CLOSE_S = 1

# ==== Smart Exit Tuning ===
TP1_SCALP_PCT      = 0.35/100
TP1_TREND_PCT      = 0.60/100
HARD_CLOSE_PNL_PCT = 1.10/100
WICK_ATR_MULT      = 1.5
EVX_SPIKE          = 1.8
BM_WALL_PROX_BPS   = 5
TIME_IN_TRADE_MIN  = 8
TRAIL_TIGHT_MULT   = 1.20

# ==== Golden Entry Settings ====
GOLDEN_ENTRY_SCORE = 6.0
GOLDEN_ENTRY_ADX   = 20.0
GOLDEN_REVERSAL_SCORE = 6.5

# ==== Golden Zone Constants ====
FIB_LOW, FIB_HIGH = 0.618, 0.786
MIN_WICK_PCT = 0.35
VOL_MA_LEN = 20
RSI_LEN_GZ, RSI_MA_LEN_GZ = 14, 9
MIN_DISP = 0.8

# ==== Execution & Strategy Thresholds ====
ADX_TREND_MIN = 20
DI_SPREAD_TREND = 6
RSI_MA_LEN = 9
RSI_NEUTRAL_BAND = (45, 55)
RSI_TREND_PERSIST = 3

GZ_MIN_SCORE = 6.0
GZ_REQ_ADX = 20
GZ_REQ_VOL_MA = 20
ALLOW_GZ_ENTRY = True

SCALP_TP1 = 0.40
SCALP_BE_AFTER = 0.30
SCALP_ATR_MULT = 1.6
TREND_TP1 = 1.20
TREND_BE_AFTER = 0.80
TREND_ATR_MULT = 1.8

MAX_TRADES_PER_HOUR = 8
COOLDOWN_SECS_AFTER_CLOSE = 45
ADX_GATE = 17

# ===== SUPER SCALP ENGINE =====
SCALP_MODE            = True
SCALP_EXECUTE         = True
SCALP_SIZE_FACTOR     = 0.35
SCALP_ADX_GATE        = 12.0
SCALP_MIN_SCORE       = 3.5
SCALP_IMB_THRESHOLD   = 1.00
SCALP_VOL_MA_FACTOR   = 1.20
SCALP_COOLDOWN_SEC    = 8
SCALP_RESPECT_WAIT    = False
SCALP_TP_SINGLE_PCT   = 0.35
SCALP_BE_AFTER_PCT    = 0.15
SCALP_ATR_TRAIL_MULT  = 1.0

# ===== SUPER COUNCIL ENHANCEMENTS =====
COUNCIL_AI_MODE = True
TREND_EARLY_DETECTION = True
MOMENTUM_ACCELERATION = True
VOLUME_CONFIRMATION = True
PRICE_ACTION_INTELLIGENCE = True

# أوزان التصويت الذكية المحسنة
WEIGHT_ADX = 1.8
WEIGHT_RSI = 1.4
WEIGHT_MACD = 1.6
WEIGHT_VOLUME = 1.3
WEIGHT_FLOW = 1.7
WEIGHT_GOLDEN = 2.0
WEIGHT_CANDLES = 1.4
WEIGHT_MOMENTUM = 1.6
WEIGHT_FOOTPRINT = 1.8
WEIGHT_DIAGONAL = 1.7
WEIGHT_EARLY_TREND = 2.0
WEIGHT_BREAKOUT = 2.2
WEIGHT_MARKET_STRUCTURE = 1.9
WEIGHT_VOLATILITY = 1.2
WEIGHT_SENTIMENT = 1.5

# ===== INTELLIGENT TREND MANAGEMENT =====
TREND_RIDING_AI = True
DYNAMIC_TP_ADJUSTMENT = True
ADAPTIVE_TRAILING = True
TREND_STRENGTH_ANALYSIS = True

# إعدادات ركوب الترند الذكية
TREND_FOLLOW_MULTIPLIER = 1.5
WEAK_TREND_EARLY_EXIT = True
STRONG_TREND_HOLD = True
TREND_REENTRY_STRATEGY = True

# ===== FLOW/FOOTPRINT Council Boost =====
FLOW_IMB_RATIO          = 1.6
FLOW_STACK_DEPTH        = 4
FLOW_ABSORB_PCTL        = 0.95
FLOW_ABSORB_MAX_TICKS   = 2
FP_WINDOW               = 3
FP_SCORE_BUY            = (2, 1.0)
FP_SCORE_SELL           = (2, 1.0)
FP_SCORE_ABSORB_PENALTY = (-1, -0.5)
DIAG_SCORE_BUY          = (2, 1.0)
DIAG_SCORE_SELL         = (2, 1.0)

# =================== PROFIT ACCUMULATION SYSTEM ===================
COMPOUND_PROFIT_REINVEST = True
PROFIT_REINVEST_RATIO = 0.4  # 40% من الأرباح يعاد استثمارها
MIN_COMPOUND_BALANCE = 50.0
PROFIT_TARGET_DAILY = 5.0  # هدف ربح يومي 5%

# ==============================================================
#   SMART MONEY CONCEPTS INTEGRATION - ICT STRATEGIES
# ==============================================================

@dataclass
class SmartTrendContext:
    """الترند الحقيقي (Up / Down / Chop)"""
    side: str              # up / down / chop
    strength: float        # 0..1
    adx: float
    di_plus: float
    di_minus: float
    rsi: float
    ema_fast: float
    ema_slow: float

    @property
    def is_strong(self):
        return self.strength >= 0.65 and self.adx >= 20

    @property
    def is_choppy(self):
        return self.adx < 15 or self.strength < 0.25

@dataclass
class OrderBlock:
    """كتل الأوامر - ICT"""
    kind: str       # bullish / bearish
    low: float
    high: float
    strength: float
    timestamp: int

@dataclass
class FVG:
    """فجوات القيمة العادلة - ICT"""
    kind: str        # bullish / bearish
    upper: float
    lower: float
    strength: float
    timestamp: int

@dataclass
class GoldenZone:
    """المناطق الذهبية"""
    kind: str      # bottom / top
    price: float
    score: float
    timestamp: int

class SmartMoneyAnalyzer:
    """محلل SMART MONEY المتقدم"""
    
    def __init__(self):
        self.order_blocks: List[OrderBlock] = []
        self.fvgs: List[FVG] = []
        self.golden_zones: List[GoldenZone] = []
        self.max_blocks = 10
        
    def build_trend_context(self, df) -> SmartTrendContext:
        """بناء سياق الترند الذكي"""
        try:
            close = df['close'].astype(float)
            high = df['high'].astype(float)
            low = df['low'].astype(float)
            
            # حساب المؤشرات
            ema_fast = talib.EMA(close, timeperiod=9)
            ema_slow = talib.EMA(close, timeperiod=21)
            adx = talib.ADX(high, low, close, timeperiod=14)
            plus_di = talib.PLUS_DI(high, low, close, timeperiod=14)
            minus_di = talib.MINUS_DI(high, low, close, timeperiod=14)
            rsi = talib.RSI(close, timeperiod=14)
            
            # القيم الحالية
            ema_fast_val = last_scalar(ema_fast)
            ema_slow_val = last_scalar(ema_slow)
            adx_val = last_scalar(adx)
            di_plus_val = last_scalar(plus_di)
            di_minus_val = last_scalar(minus_di)
            rsi_val = last_scalar(rsi)
            
            # تحديد اتجاه الترند
            if ema_fast_val > ema_slow_val and di_plus_val > di_minus_val:
                side = "up"
            elif ema_fast_val < ema_slow_val and di_minus_val > di_plus_val:
                side = "down"
            else:
                side = "chop"

            # قوة الترند
            di_spread = abs(di_plus_val - di_minus_val)
            adx_norm = min(max(adx_val / 40, 0), 1)
            di_norm = min(max(di_spread / 35, 0), 1)
            strength = (adx_norm + di_norm) / 2

            return SmartTrendContext(side, strength, adx_val, di_plus_val, di_minus_val, rsi_val, ema_fast_val, ema_slow_val)
            
        except Exception as e:
            log_w(f"Smart trend context error: {e}")
            return SmartTrendContext("chop", 0.0, 0.0, 0.0, 0.0, 50.0, 0.0, 0.0)

    def detect_order_blocks(self, df) -> Optional[OrderBlock]:
        """اكتشاف كتل الأوامر - ICT"""
        try:
            if len(df) < 10:
                return None
                
            # تحويل البيانات
            candles = []
            for i in range(len(df)):
                candles.append({
                    "open": float(df.iloc[i]["open"]),
                    "high": float(df.iloc[i]["high"]),
                    "low": float(df.iloc[i]["low"]),
                    "close": float(df.iloc[i]["close"]),
                    "timestamp": int(df.iloc[i]["time"])
                })
            
            # البحث عن كتل الأوامر في آخر 10 شموع
            for i in range(len(candles)-4, max(0, len(candles)-15), -1):
                if i < 1:
                    continue
                    
                c0 = candles[i]
                c1 = candles[i+1]

                body0 = abs(c0["open"] - c0["close"])
                body1 = abs(c1["open"] - c1["close"])
                rng0 = c0["high"] - c0["low"]
                rng1 = c1["high"] - c1["low"]

                if rng0 <= 0 or rng1 <= 0:
                    continue

                strong0 = body0 / rng0 > 0.55
                strong1 = body1 / rng1 > 0.55

                # Bullish Order Block (هبوط قوي ثم صعود قوي)
                if (c0["close"] < c0["open"] and c1["close"] > c1["open"] 
                    and strong0 and strong1):
                    ob = OrderBlock("bullish", c0["low"], c0["high"], 1.0, c0["timestamp"])
                    # تجنب التكرار
                    if not any(abs(ob.low - existing.low) < 0.0001 for existing in self.order_blocks):
                        self.order_blocks.append(ob)
                        if len(self.order_blocks) > self.max_blocks:
                            self.order_blocks.pop(0)
                        return ob
                
                # Bearish Order Block (صعود قوي ثم هبوط قوي)
                if (c0["close"] > c0["open"] and c1["close"] < c1["open"] 
                    and strong0 and strong1):
                    ob = OrderBlock("bearish", c0["low"], c0["high"], 1.0, c0["timestamp"])
                    if not any(abs(ob.low - existing.low) < 0.0001 for existing in self.order_blocks):
                        self.order_blocks.append(ob)
                        if len(self.order_blocks) > self.max_blocks:
                            self.order_blocks.pop(0)
                        return ob
                        
            return None
            
        except Exception as e:
            log_w(f"Order block detection error: {e}")
            return None

    def detect_fvg(self, df) -> Optional[FVG]:
        """اكتشاف فجوات القيمة العادلة - ICT"""
        try:
            if len(df) < 3:
                return None
                
            candles = []
            for i in range(len(df)):
                candles.append({
                    "open": float(df.iloc[i]["open"]),
                    "high": float(df.iloc[i]["high"]),
                    "low": float(df.iloc[i]["low"]),
                    "close": float(df.iloc[i]["close"]),
                    "timestamp": int(df.iloc[i]["time"])
                })
            
            # البحث عن FVG في آخر 5 شموع
            for i in range(len(candles)-3, max(0, len(candles)-8), -1):
                if i < 0:
                    continue
                    
                c0 = candles[i]
                c1 = candles[i+1]

                # Bullish FVG = الشمعة الثانية منخفضة أعلى من الشمعة الأولى عالية
                if c1["low"] > c0["high"]:
                    fvg = FVG("bullish", c1["low"], c0["high"], 
                             (c1["low"] - c0["high"]) / c0["high"] * 100, c1["timestamp"])
                    if not any(abs(fvg.lower - existing.lower) < 0.0001 for existing in self.fvgs):
                        self.fvgs.append(fvg)
                        if len(self.fvgs) > self.max_blocks:
                            self.fvgs.pop(0)
                        return fvg

                # Bearish FVG = الشمعة الثانية عالية أقل من الشمعة الأولى منخفضة
                if c1["high"] < c0["low"]:
                    fvg = FVG("bearish", c1["high"], c0["low"],
                             (c0["low"] - c1["high"]) / c0["low"] * 100, c1["timestamp"])
                    if not any(abs(fvg.lower - existing.lower) < 0.0001 for existing in self.fvgs):
                        self.fvgs.append(fvg)
                        if len(self.fvgs) > self.max_blocks:
                            self.fvgs.pop(0)
                        return fvg

            return None
            
        except Exception as e:
            log_w(f"FVG detection error: {e}")
            return None

    def detect_golden_zone(self, df, indicators) -> Optional[GoldenZone]:
        """اكتشاف المناطق الذهبية"""
        try:
            if len(df) < 2:
                return None
                
            last = df.iloc[-1]
            current_time = int(last["time"])
            
            open_p = float(last["open"])
            high = float(last["high"])
            low = float(last["low"])
            close = float(last["close"])
            
            rng = high - low
            if rng <= 0:
                return None

            # حساب الظلال
            lower_wick = min(open_p, close) - low
            upper_wick = high - max(open_p, close)

            # استخراج المؤشرات
            rsi = indicators.get('rsi', 50)
            adx = indicators.get('adx', 0)
            di_plus = indicators.get('plus_di', 0)
            di_minus = indicators.get('minus_di', 0)

            bottom_score = 0
            top_score = 0

            # قاع ذهبي
            if lower_wick / rng > 0.55: 
                bottom_score += 3
            if rsi < 35: 
                bottom_score += 2
            if di_plus > di_minus: 
                bottom_score += 2
            if adx > 18: 
                bottom_score += 1

            # قمة ذهبية
            if upper_wick / rng > 0.55: 
                top_score += 3
            if rsi > 65: 
                top_score += 2
            if di_minus > di_plus: 
                top_score += 2
            if adx > 18: 
                top_score += 1

            if bottom_score >= 6:
                zone = GoldenZone("bottom", low, bottom_score, current_time)
                self.golden_zones.append(zone)
                if len(self.golden_zones) > self.max_blocks:
                    self.golden_zones.pop(0)
                return zone

            if top_score >= 6:
                zone = GoldenZone("top", high, top_score, current_time)
                self.golden_zones.append(zone)
                if len(self.golden_zones) > self.max_blocks:
                    self.golden_zones.pop(0)
                return zone

            return None
            
        except Exception as e:
            log_w(f"Golden zone detection error: {e}")
            return None

    def get_active_zones(self, current_price: float, tolerance_bps: float = 20.0) -> dict:
        """الحصول على المناطق النشطة بالقرب من السعر الحالي"""
        tolerance = current_price * (tolerance_bps / 10000.0)
        
        active_obs = [ob for ob in self.order_blocks 
                     if ob.low - tolerance <= current_price <= ob.high + tolerance]
        
        active_fvgs = [fvg for fvg in self.fvgs 
                      if fvg.lower - tolerance <= current_price <= fvg.upper + tolerance]
        
        active_golden = [gz for gz in self.golden_zones 
                        if abs(gz.price - current_price) <= tolerance]
        
        return {
            "order_blocks": active_obs,
            "fvgs": active_fvgs,
            "golden_zones": active_golden
        }

# إنشاء محلل SMART MONEY
smart_money_analyzer = SmartMoneyAnalyzer()

# ==============================================================
#   HYBRID ENTRY DECISION - منطق الدخول الذكي الهجين
# ==============================================================

def hybrid_entry_decision(
    rf_long: bool,
    rf_short: bool,
    council_score_b: float,
    council_score_s: float,
    trend_ctx: SmartTrendContext,
    order_block: Optional[OrderBlock],
    fvg: Optional[FVG],
    golden_zone: Optional[GoldenZone],
    current_price: float
) -> Optional[str]:
    """
    قرار الدخول الهجين الذكي - يجمع بين RF، Council، وSMART MONEY Concepts
    """
    
    # أ) منطق الشراء الذكي
    if rf_long or (golden_zone and golden_zone.kind == "bottom"):
        if council_score_b >= 8:  # ثقة عالية من المجلس
            zone_confirmation = False
            zone_type = ""

            # التأكد من وجود منطقة سمارت موني
            if order_block and order_block.kind == "bullish":
                if order_block.low <= current_price <= order_block.high:
                    zone_confirmation = True
                    zone_type = f"OB@{order_block.low:.6f}"

            if fvg and fvg.kind == "bullish":
                if fvg.lower <= current_price <= fvg.upper:
                    zone_confirmation = True
                    zone_type = f"FVG@{fvg.lower:.6f}"

            if golden_zone and golden_zone.kind == "bottom":
                if abs(golden_zone.price - current_price) / current_price < 0.002:  # 0.2% tolerance
                    zone_confirmation = True
                    zone_type = f"GOLDEN@{golden_zone.price:.6f}"

            if zone_confirmation:
                # تجنب الدخول ضد الترند القوي
                if trend_ctx.side != "down" or trend_ctx.strength < 0.4:
                    log_i(f"🎯 SMART BUY SIGNAL - {zone_type} | Council: {council_score_b} | Trend: {trend_ctx.side}")
                    return "BUY"

    # ب) منطق البيع الذكي
    if rf_short or (golden_zone and golden_zone.kind == "top"):
        if council_score_s >= 8:  # ثقة عالية من المجلس
            zone_confirmation = False
            zone_type = ""

            # التأكد من وجود منطقة سمارت موني
            if order_block and order_block.kind == "bearish":
                if order_block.low <= current_price <= order_block.high:
                    zone_confirmation = True
                    zone_type = f"OB@{order_block.high:.6f}"

            if fvg and fvg.kind == "bearish":
                if fvg.lower <= current_price <= fvg.upper:
                    zone_confirmation = True
                    zone_type = f"FVG@{fvg.upper:.6f}"

            if golden_zone and golden_zone.kind == "top":
                if abs(golden_zone.price - current_price) / current_price < 0.002:  # 0.2% tolerance
                    zone_confirmation = True
                    zone_type = f"GOLDEN@{golden_zone.price:.6f}"

            if zone_confirmation:
                # تجنب الدخول ضد الترند القوي
                if trend_ctx.side != "up" or trend_ctx.strength < 0.4:
                    log_i(f"🎯 SMART SELL SIGNAL - {zone_type} | Council: {council_score_s} | Trend: {trend_ctx.side}")
                    return "SELL"

    return None

def log_smart_money_context(trend_ctx: SmartTrendContext, ob: Optional[OrderBlock], 
                           fvg: Optional[FVG], golden: Optional[GoldenZone], 
                           entry_decision: Optional[str], current_price: float):
    """تسجيل سياق السمارت موني"""
    
    log_i("───── 🔍 SMART MONEY CONTEXT ─────")
    log_i(f"📊 TREND: {trend_ctx.side.upper()} | Strength: {trend_ctx.strength:.2f} | ADX: {trend_ctx.adx:.1f}")
    log_i(f"📈 EMA: Fast={trend_ctx.ema_fast:.6f} | Slow={trend_ctx.ema_slow:.6f}")
    log_i(f"🎯 RSI: {trend_ctx.rsi:.1f} | DIs: +{trend_ctx.di_plus:.1f}/-{trend_ctx.di_minus:.1f}")
    
    if ob:
        log_i(f"🟦 ORDER BLOCK: {ob.kind} | Range: {ob.low:.6f}-{ob.high:.6f}")
    else:
        log_i("🟦 ORDER BLOCK: None")
        
    if fvg:
        log_i(f"🟩 FVG: {fvg.kind} | Range: {fvg.lower:.6f}-{fvg.upper:.6f} | Strength: {fvg.strength:.2f}%")
    else:
        log_i("🟩 FVG: None")
        
    if golden:
        log_i(f"🟨 GOLDEN ZONE: {golden.kind} | Price: {golden.price:.6f} | Score: {golden.score}")
    else:
        log_i("🟨 GOLDEN ZONE: None")
    
    # المناطق النشطة
    active_zones = smart_money_analyzer.get_active_zones(current_price)
    if active_zones["order_blocks"]:
        log_i(f"📍 ACTIVE OBs: {len(active_zones['order_blocks'])} near price")
    if active_zones["fvgs"]:
        log_i(f"📍 ACTIVE FVGs: {len(active_zones['fvgs'])} near price")
    if active_zones["golden_zones"]:
        log_i(f"📍 ACTIVE GOLDEN: {len(active_zones['golden_zones'])} near price")
    
    log_i(f"🎯 ENTRY DECISION: {entry_decision if entry_decision else 'WAITING'}")
    log_i("──────────────────────────────────")

# =================== ADVANCED MARKET ANALYSIS ===================
class AdvancedMarketAnalyzer:
    def __init__(self):
        self.market_phases = []
        self.volatility_regime = "normal"
        self.trend_strength = 0.0
        self.support_resistance = []
        
    def detect_market_phase(self, df):
        """اكتشاف مرحلة السوق الحالية"""
        try:
            close = df['close'].astype(float)
            high = df['high'].astype(float)
            low = df['low'].astype(float)
            
            # حساب المتوسطات المتحركة
            sma_20 = close.rolling(20).mean()
            sma_50 = close.rolling(50).mean()
            sma_200 = close.rolling(200).mean()
            
            # تحديد الترند
            price_vs_20 = close.iloc[-1] > sma_20.iloc[-1]
            price_vs_50 = close.iloc[-1] > sma_50.iloc[-1]
            price_vs_200 = close.iloc[-1] > sma_200.iloc[-1]
            
            ma_alignment = (sma_20.iloc[-1] > sma_50.iloc[-1] > sma_200.iloc[-1])
            
            if price_vs_200 and ma_alignment:
                return "strong_bull"
            elif price_vs_200 and not ma_alignment:
                return "bull"
            elif not price_vs_200 and ma_alignment:
                return "weak_bull"
            elif not price_vs_200 and not ma_alignment:
                return "bear"
            else:
                return "neutral"
                
        except Exception as e:
            return "neutral"
    
    def calculate_support_resistance(self, df, window=20):
        """حساب مستويات الدعم والمقاومة"""
        try:
            high = df['high'].astype(float)
            low = df['low'].astype(float)
            
            resistance = high.rolling(window).max()
            support = low.rolling(window).min()
            
            current_price = df['close'].iloc[-1]
            
            # إيجاد أقرب مستويات الدعم والمقاومة
            above_support = support[support < current_price].tail(3)
            below_resistance = resistance[resistance > current_price].head(3)
            
            return {
                'support_levels': above_support.tolist(),
                'resistance_levels': below_resistance.tolist(),
                'current_position': (current_price - above_support.iloc[-1] if len(above_support) > 0 else 0) / 
                                  (below_resistance.iloc[0] - above_support.iloc[-1] if len(above_support) > 0 and len(below_resistance) > 0 else 1)
            }
        except Exception as e:
            return {'support_levels': [], 'resistance_levels': [], 'current_position': 0.5}
    
    def analyze_volatility_regime(self, df):
        """تحليل نظام التقلب الحالي"""
        try:
            high = df['high'].astype(float)
            low = df['low'].astype(float)
            close = df['close'].astype(float)
            
            true_range = np.maximum(high - low, 
                                  np.maximum(abs(high - close.shift(1)), 
                                           abs(low - close.shift(1))))
            atr = true_range.rolling(14).mean()
            current_atr = atr.iloc[-1]
            avg_atr = atr.mean()
            
            volatility_ratio = current_atr / avg_atr if avg_atr > 0 else 1.0
            
            if volatility_ratio > 1.5:
                return "high", volatility_ratio
            elif volatility_ratio < 0.7:
                return "low", volatility_ratio
            else:
                return "normal", volatility_ratio
                
        except Exception as e:
            return "normal", 1.0

# إنشاء محلل السوق المتقدم
market_analyzer = AdvancedMarketAnalyzer()

# =================== ENHANCED TRADE MANAGER ===================
class SmartTradeManager:
    def __init__(self):
        self.trade_history = []
        self.daily_profit = 0.0
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.win_rate = 0.0
        self.avg_win = 0.0
        self.avg_loss = 0.0
        
    def record_trade(self, side, entry, exit_price, quantity, profit, duration):
        """تسجيل الصفقة في السجل"""
        trade = {
            'timestamp': datetime.now(),
            'side': side,
            'entry': entry,
            'exit': exit_price,
            'quantity': quantity,
            'profit': profit,
            'duration': duration,
            'profit_pct': (profit / (entry * quantity)) * 100 if entry * quantity > 0 else 0
        }
        
        self.trade_history.append(trade)
        self.daily_profit += profit
        
        if profit > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            
        # تحديث إحصائيات الأداء
        self.calculate_performance_metrics()
        
    def calculate_performance_metrics(self):
        """حساب مقاييس الأداء"""
        if not self.trade_history:
            return
            
        wins = [t for t in self.trade_history if t['profit'] > 0]
        losses = [t for t in self.trade_history if t['profit'] <= 0]
        
        self.win_rate = len(wins) / len(self.trade_history) * 100
        
        if wins:
            self.avg_win = sum(t['profit'] for t in wins) / len(wins)
        if losses:
            self.avg_loss = abs(sum(t['profit'] for t in losses) / len(losses))
            
    def get_trade_suggestions(self):
        """الحصول على اقتراحات تداول ذكية بناءً على الأداء"""
        suggestions = []
        
        if self.consecutive_losses >= 3:
            suggestions.append("REDUCE_SIZE: خسائر متتالية - تقليل حجم الصفقة")
            
        if self.win_rate < 40:
            suggestions.append("REVIEW_STRATEGY: نسبة نجاح منخفضة - مراجعة الاستراتيجية")
            
        if self.avg_loss > self.avg_win * 1.5:
            suggestions.append("ADJUST_STOP_LOSS: متوسط الخسارة أكبر من متوسط الربح - تعديل وقف الخسارة")
            
        return suggestions
    
    def get_optimal_position_size(self, balance, risk_per_trade=0.02):
        """حساب حجم الصفقة الأمثل بناءً على الأداء"""
        base_size = balance * risk_per_trade
        
        # تعديل الحجم بناءً على الأداء
        if self.consecutive_wins >= 3:
            size_multiplier = min(2.0, 1.0 + (self.consecutive_wins * 0.1))
        elif self.consecutive_losses >= 2:
            size_multiplier = max(0.5, 1.0 - (self.consecutive_losses * 0.2))
        else:
            size_multiplier = 1.0
            
        return base_size * size_multiplier

# إنشاء مدير الصفقات الذكي
trade_manager = SmartTradeManager()

# =================== PROFESSIONAL LOGGING ===================
def log_i(msg): 
    print(f"ℹ️ {datetime.now().strftime('%H:%M:%S')} {msg}", flush=True)

def log_g(msg): 
    print(f"✅ {datetime.now().strftime('%H:%M:%S')} {msg}", flush=True)

def log_w(msg): 
    print(f"🟨 {datetime.now().strftime('%H:%M:%S')} {msg}", flush=True)

def log_e(msg): 
    print(f"❌ {datetime.now().strftime('%H:%M:%S')} {msg}", flush=True)

def log_banner(text): 
    print(f"\n{'—'*12} {text} {'—'*12}\n", flush=True)

def save_state(state: dict):
    try:
        state["ts"] = int(time.time())
        state["trade_stats"] = {
            "daily_profit": trade_manager.daily_profit,
            "consecutive_wins": trade_manager.consecutive_wins,
            "consecutive_losses": trade_manager.consecutive_losses,
            "win_rate": trade_manager.win_rate
        }
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        log_i(f"state saved → {STATE_PATH}")
    except Exception as e:
        log_w(f"state save failed: {e}")

def load_state() -> dict:
    try:
        if not os.path.exists(STATE_PATH): return {}
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            state = json.load(f)
            
        # استعادة إحصائيات التداول
        if "trade_stats" in state:
            trade_manager.daily_profit = state["trade_stats"].get("daily_profit", 0.0)
            trade_manager.consecutive_wins = state["trade_stats"].get("consecutive_wins", 0)
            trade_manager.consecutive_losses = state["trade_stats"].get("consecutive_losses", 0)
            trade_manager.win_rate = state["trade_stats"].get("win_rate", 0.0)
            
        return state
    except Exception as e:
        log_w(f"state load failed: {e}")
    return {}

# =================== EXCHANGE FACTORY ===================
def make_ex():
    exchange_config = {
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "timeout": 20000,
    }
    
    if EXCHANGE_NAME == "bybit":
        exchange_config["options"] = {"defaultType": "swap"}
        return ccxt.bybit(exchange_config)
    else:
        exchange_config["options"] = {"defaultType": "swap"}
        return ccxt.bingx(exchange_config)

ex = make_ex()

# =================== EXCHANGE-SPECIFIC ADAPTERS ===================
def exchange_specific_params(side, is_close=False):
    if EXCHANGE_NAME == "bybit":
        if POSITION_MODE == "hedge":
            return {"positionSide": "Long" if side == "buy" else "Short", "reduceOnly": is_close}
        return {"positionSide": "Both", "reduceOnly": is_close}
    else:
        if POSITION_MODE == "hedge":
            return {"positionSide": "LONG" if side == "buy" else "SHORT", "reduceOnly": is_close}
        return {"positionSide": "BOTH", "reduceOnly": is_close}

def exchange_set_leverage(exchange, leverage, symbol):
    try:
        if EXCHANGE_NAME == "bybit":
            exchange.set_leverage(leverage, symbol)
        else:
            exchange.set_leverage(leverage, symbol, params={"side": "BOTH"})
        log_g(f"✅ {EXCHANGE_NAME.upper()} leverage set: {leverage}x")
    except Exception as e:
        log_w(f"⚠️ set_leverage warning: {e}")

# =================== MARKET SPECS ===================
MARKET = {}
AMT_PREC = 0
LOT_STEP = None
LOT_MIN  = None

def load_market_specs():
    global MARKET, AMT_PREC, LOT_STEP, LOT_MIN
    try:
        ex.load_markets()
        MARKET = ex.markets.get(SYMBOL, {})
        AMT_PREC = int((MARKET.get("precision", {}) or {}).get("amount", 0) or 0)
        LOT_STEP = (MARKET.get("limits", {}) or {}).get("amount", {}).get("step", None)
        LOT_MIN  = (MARKET.get("limits", {}) or {}).get("amount", {}).get("min",  None)
        log_i(f"🎯 {SYMBOL} specs → precision={AMT_PREC}, step={LOT_STEP}, min={LOT_MIN}")
    except Exception as e:
        log_w(f"load_market_specs: {e}")

def ensure_leverage_mode():
    try:
        exchange_set_leverage(ex, LEVERAGE, SYMBOL)
        log_i(f"📊 {EXCHANGE_NAME.upper()} position mode: {POSITION_MODE}")
    except Exception as e:
        log_w(f"ensure_leverage_mode: {e}")

try:
    load_market_specs()
    ensure_leverage_mode()
except Exception as e:
    log_w(f"exchange init: {e}")

# =================== LOGGING SETUP ===================
def setup_file_logging():
    """إعداد التسجيل المهني مع قمع رسائل Werkzeug"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    if not any(isinstance(h, RotatingFileHandler) and getattr(h, "baseFilename", "").endswith("bot.log")
               for h in logger.handlers):
        fh = RotatingFileHandler("bot.log", maxBytes=5_000_000, backupCount=7, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s [%(filename)s:%(lineno)d]"))
        logger.addHandler(fh)
    
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(ch)
    
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    logging.getLogger('ccxt.base.exchange').setLevel(logging.INFO)
    
    log_i("🔄 Professional logging ready - File rotation + Werkzeug suppression")

setup_file_logging()

# =================== HELPERS ===================
_consec_err = 0
last_loop_ts = time.time()

def _fmt(x,n=6):
    try: return f"{float(x):.{n}f}"
    except: return str(x)

def _pct(x):
    try: return f"{float(x):.2f}%"
    except: return str(x)

def last_scalar(x, default=0.0):
    """يرجع float من آخر عنصر; يقبل Series/np.ndarray/list/float."""
    try:
        if isinstance(x, pd.Series): return float(x.iloc[-1])
        if isinstance(x, (list, tuple, np.ndarray)): return float(x[-1])
        if x is None: return float(default)
        return float(x)
    except Exception:
        return float(default)

def safe_get(ind: dict, key: str, default=0.0):
    """يقرأ مؤشر من dict ويحوّله scalar أخير."""
    if ind is None: 
        return float(default)
    val = ind.get(key, default)
    return last_scalar(val, default=default)

def _ind_brief(ind):
    if not ind: return "n/a"
    
    # استخراج قيم scalar بأمان
    adx = safe_get(ind, 'adx', 0)
    di_spread = safe_get(ind, 'di_spread', 0)
    rsi = safe_get(ind, 'rsi', 0)
    rsi_ma = safe_get(ind, 'rsi_ma', 0)
    atr = safe_get(ind, 'atr', 0)
    
    return (f"ADX={adx:.1f} DI={di_spread:.1f} | "
            f"RSI={rsi:.1f}/{rsi_ma:.1f} | "
            f"ATR={atr:.4f}")

def _council_brief(c):
    if not c: return "n/a"
    return f"B:{c.get('b',0)}/{_fmt(c.get('score_b',0),1)} | S:{c.get('s',0)}/{_fmt(c.get('score_s',0),1)}"

def _flow_brief(f):
    if not f: return "n/a"
    parts=[f"Δz={_fmt(f.get('delta_z','n/a'),2)}", f"CVD={_fmt(f.get('cvd_last','n/a'),0)}", f"trend={f.get('cvd_trend','?')}"]
    if f.get("spike"): parts.append("SPIKE")
    return " ".join(parts)

def print_position_snapshot(reason="OPEN", color=None):
    try:
        side   = STATE.get("side")
        open_f = STATE.get("open",False)
        qty    = STATE.get("qty"); px = STATE.get("entry")
        mode   = STATE.get("mode","trend")
        lev    = globals().get("LEVERAGE",0)
        tp1    = globals().get("TP1_PCT_BASE",0)
        be_a   = globals().get("BREAKEVEN_AFTER",0)
        trailA = globals().get("TRAIL_ACTIVATE_PCT",0)
        atrM   = globals().get("ATR_TRAIL_MULT",0)
        bal    = balance_usdt()
        spread = STATE.get("last_spread_bps")
        council= STATE.get("last_council")
        ind    = STATE.get("last_ind")
        flow   = STATE.get("last_flow")

        if color is None:
            icon = "🟢" if side=="buy" else "🔴"
        else:
            icon = "🟢" if str(color).lower()=="green" else "🔴"

        log_i(f"{icon} {reason} — POSITION SNAPSHOT")
        log_i(f"SIDE: {side} | QTY: {_fmt(qty)} | ENTRY: {_fmt(px)} | LEV: {lev}× | MODE: {mode} | OPEN: {open_f}")
        log_i(f"TP1: {_pct(tp1)} | BE@: {_pct(be_a)} | TRAIL: act≥{_pct(trailA)}, ATR×{atrM} | SPREAD: {_fmt(spread,2)} bps")
        log_i(f"IND: {_ind_brief(ind)}")
        log_i(f"COUNCIL: {_council_brief(council)}")
        log_i(f"FLOW: {_flow_brief(flow)}")
        
        # إضافة إحصائيات الأداء
        log_i(f"PERFORMANCE: Win Rate: {trade_manager.win_rate:.1f}% | Daily PnL: {trade_manager.daily_profit:.2f}")
        log_i("—"*72)
    except Exception as e:
        log_w(f"SNAPSHOT ERR: {e}")

def _round_amt(q):
    if q is None: return 0.0
    try:
        d = Decimal(str(q))
        if LOT_STEP and isinstance(LOT_STEP,(int,float)) and LOT_STEP>0:
            step = Decimal(str(LOT_STEP))
            d = (d/step).to_integral_value(rounding=ROUND_DOWN)*step
        prec = int(AMT_PREC) if AMT_PREC and AMT_PREC>=0 else 0
        d = d.quantize(Decimal(1).scaleb(-prec), rounding=ROUND_DOWN)
        if LOT_MIN and isinstance(LOT_MIN,(int,float)) and LOT_MIN>0 and d < Decimal(str(LOT_MIN)): return 0.0
        return float(d)
    except (InvalidOperation, ValueError, TypeError):
        return max(0.0, float(q))

def safe_qty(q): 
    q = _round_amt(q)
    if q<=0: log_w(f"qty invalid after normalize → {q}")
    return q

def fmt(v, d=6, na="—"):
    try:
        if v is None or (isinstance(v,float) and (math.isnan(v) or math.isinf(v))): return na
        return f"{float(v):.{d}f}"
    except Exception:
        return na

def with_retry(fn, tries=3, base_wait=0.4):
    global _consec_err
    for i in range(tries):
        try:
            r = fn()
            _consec_err = 0
            return r
        except Exception:
            _consec_err += 1
            if i == tries-1: raise
            time.sleep(base_wait*(2**i) + random.random()*0.25)

def fetch_ohlcv(limit=600):
    rows = with_retry(lambda: ex.fetch_ohlcv(SYMBOL, timeframe=INTERVAL, limit=limit, params={"type":"swap"}))
    return pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])

def price_now():
    try:
        t = with_retry(lambda: ex.fetch_ticker(SYMBOL))
        return t.get("last") or t.get("close")
    except Exception: return None

def balance_usdt():
    if not MODE_LIVE: return 1000.0  # رصيد افتراضي أكبر للتجربة
    try:
        b = with_retry(lambda: ex.fetch_balance(params={"type":"swap"}))
        return b.get("total",{}).get("USDT") or b.get("free",{}).get("USDT")
    except Exception: return None

def orderbook_spread_bps():
    try:
        ob = with_retry(lambda: ex.fetch_order_book(SYMBOL, limit=5))
        bid = ob["bids"][0][0] if ob["bids"] else None
        ask = ob["asks"][0][0] if ob["asks"] else None
        if not (bid and ask): return None
        mid = (bid+ask)/2.0
        return ((ask-bid)/mid)*10000.0
    except Exception:
        return None

def _interval_seconds(iv: str) -> int:
    iv=(iv or "").lower().strip()
    if iv.endswith("m"): return int(float(iv[:-1]))*60
    if iv.endswith("h"): return int(float(iv[:-1]))*3600
    if iv.endswith("d"): return int(float(iv[:-1]))*86400
    return 15*60

def time_to_candle_close(df: pd.DataFrame) -> int:
    tf = _interval_seconds(INTERVAL)
    if len(df) == 0: return tf
    cur_start_ms = int(df["time"].iloc[-1])
    now_ms = int(time.time()*1000)
    next_close_ms = cur_start_ms + tf*1000
    while next_close_ms <= now_ms:
        next_close_ms += tf*1000
    left = max(0, next_close_ms - now_ms)
    return int(left/1000)

def fmt_walls(walls):
    return ", ".join([f"{p:.6f}@{q:.0f}" for p, q in walls]) if walls else "-"

# ========= Bookmap snapshot =========
def bookmap_snapshot(exchange, symbol, depth=BOOKMAP_DEPTH):
    try:
        ob = exchange.fetch_order_book(symbol, depth)
        bids = ob.get("bids", [])[:depth]; asks = ob.get("asks", [])[:depth]
        if not bids or not asks:
            return {"ok": False, "why": "empty"}
        b_sizes = np.array([b[1] for b in bids]); b_prices = np.array([b[0] for b in bids])
        a_sizes = np.array([a[1] for a in asks]); a_prices = np.array([a[0] for a in asks])
        b_idx = b_sizes.argsort()[::-1][:BOOKMAP_TOPWALLS]
        a_idx = a_sizes.argsort()[::-1][:BOOKMAP_TOPWALLS]
        buy_walls = [(float(b_prices[i]), float(b_sizes[i])) for i in b_idx]
        sell_walls = [(float(a_prices[i]), float(a_sizes[i])) for i in a_idx]
        imb = b_sizes.sum() / max(a_sizes.sum(), 1e-12)
        return {"ok": True, "buy_walls": buy_walls, "sell_walls": sell_walls, "imbalance": float(imb)}
    except Exception as e:
        return {"ok": False, "why": f"{e}"}

# ========= Volume flow / Delta & CVD =========
def compute_flow_metrics(df):
    try:
        if len(df) < max(30, FLOW_WINDOW+2):
            return {"ok": False, "why": "short_df"}
        close = df["close"].astype(float).copy()
        vol = df["volume"].astype(float).copy()
        up_mask = close.diff().fillna(0) > 0
        up_vol = (vol * up_mask).astype(float)
        dn_vol = (vol * (~up_mask)).astype(float)
        delta = up_vol - dn_vol
        cvd = delta.cumsum()
        cvd_ma = cvd.rolling(CVD_SMOOTH).mean()
        wnd = delta.tail(FLOW_WINDOW)
        mu = float(wnd.mean()); sd = float(wnd.std() or 1e-12)
        z = float((wnd.iloc[-1] - mu) / sd)
        trend = "up" if (cvd_ma.iloc[-1] - cvd_ma.iloc[-min(CVD_SMOOTH, len(cvd_ma))]) >= 0 else "down"
        return {"ok": True, "delta_last": float(delta.iloc[-1]), "delta_mean": mu, "delta_z": z,
                "cvd_last": float(cvd.iloc[-1]), "cvd_trend": trend, "spike": abs(z) >= FLOW_SPIKE_Z}
    except Exception as e:
        return {"ok": False, "why": str(e)}

# =================== ADVANCED INDICATORS ===================
def compute_advanced_indicators(df):
    """حساب المؤشرات المتقدمة"""
    try:
        close = df['close'].astype(float)
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        volume = df['volume'].astype(float)
        
        # مؤشرات الترند
        sma_20 = talib.SMA(close, timeperiod=20)
        sma_50 = talib.SMA(close, timeperiod=50)
        ema_20 = talib.EMA(close, timeperiod=20)
        
        # مؤشرات الزخم
        rsi = talib.RSI(close, timeperiod=14)
        macd, macd_signal, macd_hist = talib.MACD(close)
        stoch_k, stoch_d = talib.STOCH(high, low, close)
        
        # مؤشرات التقلب
        atr = talib.ATR(high, low, close, timeperiod=14)
        bollinger_upper, bollinger_middle, bollinger_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        
        # مؤشرات الحجم
        obv = talib.OBV(close, volume)
        
        # مؤشرات الاتجاه
        adx = talib.ADX(high, low, close, timeperiod=14)
        plus_di = talib.PLUS_DI(high, low, close, timeperiod=14)
        minus_di = talib.MINUS_DI(high, low, close, timeperiod=14)
        
        return {
            'sma_20': last_scalar(sma_20),
            'sma_50': last_scalar(sma_50),
            'ema_20': last_scalar(ema_20),
            'rsi': last_scalar(rsi),
            'macd': last_scalar(macd),
            'macd_signal': last_scalar(macd_signal),
            'macd_hist': last_scalar(macd_hist),
            'stoch_k': last_scalar(stoch_k),
            'stoch_d': last_scalar(stoch_d),
            'atr': last_scalar(atr),
            'bollinger_upper': last_scalar(bollinger_upper),
            'bollinger_middle': last_scalar(bollinger_middle),
            'bollinger_lower': last_scalar(bollinger_lower),
            'obv': last_scalar(obv),
            'adx': last_scalar(adx),
            'plus_di': last_scalar(plus_di),
            'minus_di': last_scalar(minus_di),
            'volume': last_scalar(volume)
        }
    except Exception as e:
        log_w(f"Advanced indicators error: {e}")
        return {}

# =================== ULTRA INTELLIGENT COUNCIL AI ===================
def ultra_intelligent_council_ai(df):
    """
    مجلس الإدارة الذكي الفائق - يدمج 25 استراتيجية متقدمة لاتخاذ أفضل القرارات
    """
    try:
        if len(df) < 100:
            return {"b": 0, "s": 0, "score_b": 0.0, "score_s": 0.0, "confidence": 0.0, "logs": []}
        
        # تحليل السوق المتقدم
        market_phase = market_analyzer.detect_market_phase(df)
        support_resistance = market_analyzer.calculate_support_resistance(df)
        volatility_regime, volatility_ratio = market_analyzer.analyze_volatility_regime(df)
        
        # المؤشرات المتقدمة
        advanced_indicators = compute_advanced_indicators(df)
        
        # دمج المؤشرات
        indicators = advanced_indicators
        
        # تحليل SMART MONEY
        trend_ctx = smart_money_analyzer.build_trend_context(df)
        order_block = smart_money_analyzer.detect_order_blocks(df)
        fvg = smart_money_analyzer.detect_fvg(df)
        golden_zone = smart_money_analyzer.detect_golden_zone(df, indicators)
        
        # تحليل التدفق
        flow_metrics = compute_flow_metrics(df)
        orderbook = bookmap_snapshot(ex, SYMBOL)
        
        votes_b = 0
        votes_s = 0
        score_b = 0.0
        score_s = 0.0
        logs = []
        confidence_factors = []
        
        current_price = float(df['close'].iloc[-1])
        
        # ===== 1. تحليل مرحلة السوق =====
        if market_phase == "strong_bull":
            score_b += WEIGHT_MARKET_STRUCTURE * 2.5
            votes_b += 3
            logs.append("📈 مرحلة صاعدة قوية")
            confidence_factors.append(1.8)
        elif market_phase == "bull":
            score_b += WEIGHT_MARKET_STRUCTURE * 1.5
            votes_b += 2
            logs.append("📈 مرحلة صاعدة")
            confidence_factors.append(1.3)
        elif market_phase == "bear":
            score_s += WEIGHT_MARKET_STRUCTURE * 1.5
            votes_s += 2
            logs.append("📉 مرحلة هابطة")
            confidence_factors.append(1.3)
        elif market_phase == "strong_bear":
            score_s += WEIGHT_MARKET_STRUCTURE * 2.5
            votes_s += 3
            logs.append("📉 مرحلة هابطة قوية")
            confidence_factors.append(1.8)
        
        # ===== 2. SMART MONEY ANALYSIS =====
        if order_block and order_block.kind == "bullish":
            if order_block.low <= current_price <= order_block.high:
                score_b += WEIGHT_BREAKOUT * 2.5
                votes_b += 3
                logs.append(f"🟦 Bullish Order Block active")
                confidence_factors.append(1.7)
        
        if order_block and order_block.kind == "bearish":
            if order_block.low <= current_price <= order_block.high:
                score_s += WEIGHT_BREAKOUT * 2.5
                votes_s += 3
                logs.append(f"🟦 Bearish Order Block active")
                confidence_factors.append(1.7)
        
        if fvg and fvg.kind == "bullish":
            if fvg.lower <= current_price <= fvg.upper:
                score_b += WEIGHT_BREAKOUT * 2.0
                votes_b += 2
                logs.append(f"🟩 Bullish FVG active")
                confidence_factors.append(1.5)
        
        if fvg and fvg.kind == "bearish":
            if fvg.lower <= current_price <= fvg.upper:
                score_s += WEIGHT_BREAKOUT * 2.0
                votes_s += 2
                logs.append(f"🟩 Bearish FVG active")
                confidence_factors.append(1.5)
        
        if golden_zone and golden_zone.kind == "bottom":
            score_b += WEIGHT_GOLDEN * 2.5
            votes_b += 3
            logs.append(f"🟨 Golden Bottom detected")
            confidence_factors.append(1.8)
        
        if golden_zone and golden_zone.kind == "top":
            score_s += WEIGHT_GOLDEN * 2.5
            votes_s += 3
            logs.append(f"🟨 Golden Top detected")
            confidence_factors.append(1.8)
        
        # ===== 3. تحليل الترند الذكي =====
        if trend_ctx.side == "up" and trend_ctx.is_strong:
            score_b += WEIGHT_EARLY_TREND * 2.0
            votes_b += 2
            logs.append(f"📊 Strong Uptrend (ADX: {trend_ctx.adx:.1f})")
            confidence_factors.append(1.5)
        elif trend_ctx.side == "down" and trend_ctx.is_strong:
            score_s += WEIGHT_EARLY_TREND * 2.0
            votes_s += 2
            logs.append(f"📊 Strong Downtrend (ADX: {trend_ctx.adx:.1f})")
            confidence_factors.append(1.5)
        
        # ===== 4. المؤشرات التقليدية =====
        rsi = indicators.get('rsi', 50)
        if rsi < 30:
            score_b += WEIGHT_RSI * 2.0
            votes_b += 2
            logs.append("📊 RSI في منطقة شراء قوية")
        elif rsi > 70:
            score_s += WEIGHT_RSI * 2.0
            votes_s += 2
            logs.append("📊 RSI في منطقة بيع قوية")
        
        # MACD
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        if macd > macd_signal and indicators.get('macd_hist', 0) > 0:
            score_b += WEIGHT_MACD * 1.8
            votes_b += 2
            logs.append("📈 MACD صاعد قوي")
        elif macd < macd_signal and indicators.get('macd_hist', 0) < 0:
            score_s += WEIGHT_MACD * 1.8
            votes_s += 2
            logs.append("📉 MACD هابط قوي")
        
        # ADX والاتجاه
        adx = indicators.get('adx', 0)
        plus_di = indicators.get('plus_di', 0)
        minus_di = indicators.get('minus_di', 0)
        
        if adx > 25:  # ترند قوي
            if plus_di > minus_di:
                score_b += WEIGHT_ADX * 2.5
                votes_b += 3
                logs.append(f"🎯 ترند صاعد قوي (ADX: {adx:.1f})")
                confidence_factors.append(1.8)
            else:
                score_s += WEIGHT_ADX * 2.5
                votes_s += 3
                logs.append(f"🎯 ترند هابط قوي (ADX: {adx:.1f})")
                confidence_factors.append(1.8)
        
        # ===== 5. تحليل التدفق =====
        if flow_metrics.get('ok'):
            delta_z = flow_metrics.get('delta_z', 0)
            cvd_trend = flow_metrics.get('cvd_trend', '')
            
            if delta_z > 2.5 and cvd_trend == 'up':
                score_b += WEIGHT_FLOW * 2.2
                votes_b += 3
                logs.append(f"🌊 تدفق شرائي قوي جداً (z: {delta_z:.2f})")
                confidence_factors.append(1.7)
            elif delta_z < -2.5 and cvd_trend == 'down':
                score_s += WEIGHT_FLOW * 2.2
                votes_s += 3
                logs.append(f"🌊 تدفق بيعي قوي جداً (z: {delta_z:.2f})")
                confidence_factors.append(1.7)
        
        # ===== 6. تحليل الكتاب =====
        if orderbook.get('ok'):
            imbalance = orderbook.get('imbalance', 1.0)
            if imbalance > 2.0:
                score_b += WEIGHT_SENTIMENT * 1.5
                votes_b += 2
                logs.append(f"📚 تضارب قوي لصالح المشترين (imb: {imbalance:.2f})")
            elif imbalance < 0.5:
                score_s += WEIGHT_SENTIMENT * 1.5
                votes_s += 2
                logs.append(f"📚 تضارب قوي لصالح البائعين (imb: {imbalance:.2f})")
        
        # ===== 7. تطبيق عوامل الثقة =====
        if confidence_factors:
            confidence_multiplier = sum(confidence_factors) / len(confidence_factors)
            score_b *= confidence_multiplier
            score_s *= confidence_multiplier
        
        # ===== 8. مراعاة أداء التداول السابق =====
        if trade_manager.consecutive_losses >= 2:
            score_b *= 0.7
            score_s *= 0.7
            logs.append("⚠️ خسائر متتالية - تخفيض ثقة")
        
        if trade_manager.consecutive_wins >= 3:
            score_b *= 1.2
            score_s *= 1.2
            logs.append("🎯 أرباح متتالية - زيادة ثقة")
        
        # ===== 9. حساب الثقة النهائية =====
        total_score = score_b + score_s
        max_possible_score = 35.0  # أقصى درجة ممكنة
        
        confidence = min(1.0, total_score / max_possible_score)
        
        # ===== 10. تطبيق الحد الأدنى للثقة =====
        min_confidence = 0.65
        if confidence < min_confidence:
            score_b *= 0.5
            score_s *= 0.5
            logs.append(f"🛡️ ثقة منخفضة ({confidence:.2f} < {min_confidence}) - تخفيض")
        
        return {
            "b": votes_b,
            "s": votes_s,
            "score_b": round(score_b, 2),
            "score_s": round(score_s, 2),
            "confidence": round(confidence, 2),
            "logs": logs,
            "market_phase": market_phase,
            "volatility_regime": volatility_regime,
            "support_resistance": support_resistance,
            "indicators": indicators,
            "smart_money": {
                "trend_ctx": trend_ctx,
                "order_block": order_block,
                "fvg": fvg,
                "golden_zone": golden_zone
            }
        }
        
    except Exception as e:
        log_e(f"Ultra intelligent council error: {e}")
        return {"b": 0, "s": 0, "score_b": 0.0, "score_s": 0.0, "confidence": 0.0, "logs": [f"Error: {e}"]}

# =================== ENHANCED TRADE EXECUTION ===================
def execute_intelligent_trade(side, price, qty, council_data, market_analysis):
    """تنفيذ صفقة ذكية مع تحليل متقدم"""
    try:
        if not EXECUTE_ORDERS or DRY_RUN:
            log_i(f"DRY_RUN: {side} {qty:.4f} @ {price:.6f}")
            return True
        
        if qty <= 0:
            log_e("❌ كمية غير صالحة للتنفيذ")
            return False
        
        # تحضير البيانات للتنفيذ
        confidence = council_data.get('confidence', 0)
        market_phase = market_analysis.get('market_phase', 'neutral')
        volatility_regime = market_analysis.get('volatility_regime', 'normal')
        
        log_i(f"🎯 EXECUTING INTELLIGENT TRADE:")
        log_i(f"   SIDE: {side.upper()}")
        log_i(f"   QTY: {qty:.4f}")
        log_i(f"   PRICE: {price:.6f}")
        log_i(f"   CONFIDENCE: {confidence:.2f}")
        log_i(f"   MARKET PHASE: {market_phase}")
        log_i(f"   VOLATILITY: {volatility_regime}")
        
        if MODE_LIVE:
            exchange_set_leverage(ex, LEVERAGE, SYMBOL)
            params = exchange_specific_params(side, is_close=False)
            ex.create_order(SYMBOL, "market", side, qty, None, params)
        
        log_g(f"✅ INTELLIGENT TRADE EXECUTED: {side.upper()} {qty:.4f} @ {price:.6f}")
        
        # تسجيل الصفقة في المدير
        trade_manager.record_trade(
            side=side,
            entry=price,
            exit_price=price,  # سيتم تحديثها عند الإغلاق
            quantity=qty,
            profit=0.0,  # سيتم تحديثها عند الإغلاق
            duration=0
        )
        
        return True
        
    except Exception as e:
        log_e(f"❌ INTELLIGENT TRADE EXECUTION FAILED: {e}")
        return False

def compute_adaptive_position_size(balance, price, confidence, market_phase):
    """حساب حجم صفقة متكيف مع ظروف السوق"""
    base_size = trade_manager.get_optimal_position_size(balance)
    
    # تعديل الحجم بناءً على الثقة
    confidence_multiplier = 0.5 + (confidence * 0.5)  # 0.5 إلى 1.0
    
    # تعديل الحجم بناءً على مرحلة السوق
    if market_phase in ["strong_bull", "strong_bear"]:
        market_multiplier = 1.3
    elif market_phase in ["bull", "bear"]:
        market_multiplier = 1.1
    else:
        market_multiplier = 0.8
    
    adaptive_size = base_size * confidence_multiplier * market_multiplier
    
    # التأكد من أن الحجم ضمن الحدود المعقولة
    max_position = balance * LEVERAGE * 0.8  # 80% من الرصيد بالرافعة
    final_size = min(adaptive_size, max_position / price) if price > 0 else adaptive_size
    
    log_i(f"📊 ADAPTIVE POSITION SIZING:")
    log_i(f"   Base: {base_size:.4f}")
    log_i(f"   Confidence Multiplier: {confidence_multiplier:.2f}")
    log_i(f"   Market Multiplier: {market_multiplier:.2f}")
    log_i(f"   Final: {final_size:.4f}")
    
    return safe_qty(final_size)

# =================== ENHANCED TRADE MANAGEMENT ===================
def manage_intelligent_position(df, indicators, price_info):
    """إدارة ذكية للمراكب المفتوحة"""
    if not STATE["open"] or STATE["qty"] <= 0:
        return
    
    try:
        current_price = price_info.get("price", 0)
        entry_price = STATE["entry"]
        side = STATE["side"]
        qty = STATE["qty"]
        
        # حساب الربح/الخسارة
        if side == "long":
            pnl_pct = (current_price - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - current_price) / entry_price * 100
        
        STATE["pnl"] = pnl_pct
        
        if pnl_pct > STATE["highest_profit_pct"]:
            STATE["highest_profit_pct"] = pnl_pct
        
        # تحليل السوق الحالي
        market_phase = market_analyzer.detect_market_phase(df)
        volatility_regime, volatility_ratio = market_analyzer.analyze_volatility_regime(df)
        
        # إستراتيجية الخروج الذكية
        exit_decision = intelligent_exit_decision(
            pnl_pct, side, indicators, market_phase, volatility_regime
        )
        
        if exit_decision["action"] == "close":
            log_i(f"🔴 INTELLIGENT EXIT: {exit_decision['reason']}")
            close_market_strict(exit_decision['reason'])
        elif exit_decision["action"] == "partial":
            close_qty = safe_qty(qty * exit_decision["qty_pct"])
            if close_qty > 0:
                close_side = "sell" if side == "long" else "buy"
                if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
                    try:
                        params = exchange_specific_params(close_side, is_close=True)
                        ex.create_order(SYMBOL, "market", close_side, close_qty, None, params)
                        log_g(f"🎯 PARTIAL CLOSE: {exit_decision['reason']} - {close_qty:.4f} units")
                        STATE["qty"] = safe_qty(qty - close_qty)
                    except Exception as e:
                        log_e(f"❌ PARTIAL CLOSE FAILED: {e}")
        
        # تحديث الوقف المتحرك الذكي
        update_intelligent_trailing_stop(current_price, side, indicators, market_phase)
        
    except Exception as e:
        log_e(f"❌ INTELLIGENT POSITION MANAGEMENT ERROR: {e}")

def intelligent_exit_decision(pnl_pct, side, indicators, market_phase, volatility_regime):
    """قرار خروج ذكي بناءً على متعددة معايير"""
    try:
        rsi = indicators.get('rsi', 50)
        adx = indicators.get('adx', 0)
        macd_hist = indicators.get('macd_hist', 0)
        
        # إستراتيجية الخروج بناءً على مرحلة السوق
        if market_phase in ["strong_bull", "strong_bear"]:
            # في الترند القوي، نبقى لفترة أطول
            tp_targets = [1.0, 2.0, 3.5, 5.0, 7.0, 10.0]
        else:
            # في السوق الجانبي، نخرج مبكراً
            tp_targets = [0.8, 1.5, 2.5, 4.0, 6.0]
        
        # تحقيق أهداف الربح
        for i, target in enumerate(tp_targets):
            tp_key = f"tp_{i+1}_done"
            if not STATE.get(tp_key, False) and pnl_pct >= target:
                close_pct = 0.2 if i < 3 else 0.15  # 20% للأهداف الأولى، 15% للبقية
                return {
                    "action": "partial",
                    "qty_pct": close_pct,
                    "reason": f"TP{i+1} achieved: {target:.1f}%"
                }
        
        # إشارات انعكاس قوية
        reversal_signals = 0
        if (side == "long" and rsi > 80 and macd_hist < 0) or (side == "short" and rsi < 20 and macd_hist > 0):
            reversal_signals += 1
        
        if adx < 20 and abs(pnl_pct) > 2.0:  # فقدان الزخم مع ربح جيد
            reversal_signals += 1
        
        if reversal_signals >= 2:
            return {
                "action": "close",
                "reason": "Strong reversal signals"
            }
        
        # خروج وقائي في التقلب العالي مع أرباح جيدة
        if volatility_regime == "high" and pnl_pct > 3.0:
            return {
                "action": "close",
                "reason": "High volatility with good profit - secure gains"
            }
        
        return {"action": "hold", "reason": "Continue riding trend"}
        
    except Exception as e:
        log_w(f"Intelligent exit decision error: {e}")
        return {"action": "hold", "reason": "Error in decision"}

def update_intelligent_trailing_stop(current_price, side, indicators, market_phase):
    """تحديث وقف الخسارة المتحرك الذكي"""
    try:
        atr = indicators.get('atr', 0)
        pnl_pct = STATE.get("pnl", 0)
        
        # تحديد مضاعف ATR بناءً على مرحلة السوق والتقلب
        if market_phase in ["strong_bull", "strong_bear"]:
            base_multiplier = 2.0
        else:
            base_multiplier = 1.5
        
        # تعديل المضاعف بناءً على مستوى الربح
        if pnl_pct > 5.0:
            trail_mult = base_multiplier * 0.7  # وقف أضيق عند الأرباح العالية
        elif pnl_pct > 2.0:
            trail_mult = base_multiplier * 0.8
        else:
            trail_mult = base_multiplier
        
        if not STATE.get("trail_active", False) and pnl_pct >= 1.0:
            STATE["trail_active"] = True
            STATE["breakeven_armed"] = True
            STATE["breakeven"] = STATE["entry"]
            log_i("🔄 Intelligent trailing stop activated")
        
        if STATE.get("trail_active"):
            if side == "long":
                new_trail = current_price - (atr * trail_mult)
                if STATE.get("trail") is None or new_trail > STATE["trail"]:
                    STATE["trail"] = new_trail
                    if STATE["trail"] > STATE.get("entry", 0):
                        log_i(f"🔼 Intelligent trail updated: {STATE['trail']:.6f}")
            else:
                new_trail = current_price + (atr * trail_mult)
                if STATE.get("trail") is None or new_trail < STATE["trail"]:
                    STATE["trail"] = new_trail
                    if STATE["trail"] < STATE.get("entry", float('inf')):
                        log_i(f"🔽 Intelligent trail updated: {STATE['trail']:.6f}")
        
        # تفعيل وقف الخسارة عند نقطة التعادل بعد تحقيق ربح معين
        if STATE.get("breakeven_armed") and not STATE.get("breakeven_active") and pnl_pct >= 1.5:
            STATE["breakeven_active"] = True
            STATE["trail"] = STATE["entry"]  # وقف عند نقطة الدخول
            log_i("🎯 Breakeven activated - risk free trade")
            
    except Exception as e:
        log_w(f"Intelligent trailing stop error: {e}")

# =================== ENHANCED TRADING LOOP WITH SMART MONEY ===================
def ultra_intelligent_trading_loop():
    """الحلقة الرئيسية المحسنة مع SMART MONEY INTEGRATION"""
    
    log_banner("STARTING ULTRA INTELLIGENT TRADING BOT WITH SMART MONEY")
    log_i(f"🤖 Bot Version: {BOT_VERSION}")
    log_i(f"💱 Exchange: {EXCHANGE_NAME.upper()}")
    log_i(f"📈 Symbol: {SYMBOL}")
    log_i(f"⏰ Interval: {INTERVAL}")
    log_i(f"🎯 Leverage: {LEVERAGE}x")
    log_i(f"📊 Risk Allocation: {RISK_ALLOC*100}%")
    
    while True:
        try:
            # جمع البيانات الأساسية
            balance = balance_usdt()
            current_price = price_now()
            df = fetch_ohlcv(limit=200)
            
            if df.empty or current_price is None:
                log_w("📭 No data available - retrying...")
                time.sleep(BASE_SLEEP)
                continue
            
            # 1️⃣ قرار مجلس الإدارة الذكي
            council_data = ultra_intelligent_council_ai(df)
            
            # 2️⃣ استخراج بيانات SMART MONEY
            smart_money_data = council_data.get("smart_money", {})
            trend_ctx = smart_money_data.get("trend_ctx")
            order_block = smart_money_data.get("order_block")
            fvg = smart_money_data.get("fvg")
            golden_zone = smart_money_data.get("golden_zone")
            
            # 3️⃣ قرار الدخول الهجين الذكي
            entry_decision = hybrid_entry_decision(
                rf_long=False,  # يمكن استبدالها بالقيم الفعلية من RF
                rf_short=False, # يمكن استبدالها بالقيم الفعلية من RF
                council_score_b=council_data["score_b"],
                council_score_s=council_data["score_s"],
                trend_ctx=trend_ctx,
                order_block=order_block,
                fvg=fvg,
                golden_zone=golden_zone,
                current_price=current_price
            )
            
            # 4️⃣ تحديث الحالة
            STATE["last_council"] = council_data
            STATE["last_ind"] = council_data.get("indicators", {})
            STATE["last_spread_bps"] = orderbook_spread_bps()
            
            # 5️⃣ تسجيل السياق
            if LOG_ADDONS:
                log_i(f"🏪 MARKET: {council_data.get('market_phase', 'neutral').upper()} | VOLATILITY: {council_data.get('volatility_regime', 'normal')}")
                log_i(f"🎯 COUNCIL: B{council_data['b']}/S{council_data['s']} | "
                      f"Score: {council_data['score_b']:.1f}/{council_data['score_s']:.1f} | "
                      f"Confidence: {council_data['confidence']:.2f}")
                
                for log_msg in council_data.get("logs", [])[-5:]:
                    log_i(f"   {log_msg}")
                
                # تسجيل سياق SMART MONEY
                log_smart_money_context(trend_ctx, order_block, fvg, golden_zone, entry_decision, current_price)
            
            # 6️⃣ إدارة المركز المفتوح
            if STATE["open"]:
                manage_intelligent_position(df, council_data.get("indicators", {}), {
                    "price": current_price,
                    "market_phase": council_data.get('market_phase', 'neutral'),
                    "volatility_regime": council_data.get('volatility_regime', 'normal')
                })
            
            # 7️⃣ فتح صفقات جديدة
            if not STATE["open"] and entry_decision:
                # حساب حجم صفقة متكيف
                market_phase = council_data.get('market_phase', 'neutral')
                position_size = compute_adaptive_position_size(
                    balance, current_price, council_data["confidence"], market_phase
                )
                
                if position_size > 0 and council_data["confidence"] >= 0.7:
                    log_i(f"🎯 SMART MONEY TRADE SIGNAL:")
                    log_i(f"   Side: {entry_decision.upper()}")
                    log_i(f"   Size: {position_size:.4f}")
                    log_i(f"   Price: {current_price:.6f}")
                    log_i(f"   Confidence: {council_data['confidence']:.2f}")
                    log_i(f"   Smart Money Confirmation: ✅")
                    
                    # تنفيذ الصفقة
                    success = execute_intelligent_trade(
                        entry_decision.lower(), 
                        current_price, 
                        position_size, 
                        council_data, {
                            "market_phase": market_phase,
                            "volatility_regime": council_data.get('volatility_regime', 'normal'),
                            "support_resistance": council_data.get('support_resistance', {})
                        }
                    )
                    
                    if success:
                        STATE.update({
                            "open": True,
                            "side": "long" if entry_decision == "BUY" else "short",
                            "entry": current_price,
                            "qty": position_size,
                            "pnl": 0.0,
                            "bars": 0,
                            "trail": None,
                            "breakeven": None,
                            "highest_profit_pct": 0.0,
                            "profit_targets_achieved": 0,
                            "mode": "smart_money_hybrid"
                        })
                        
                        save_state({
                            "in_position": True,
                            "side": entry_decision.upper(),
                            "entry_price": current_price,
                            "position_qty": position_size,
                            "opened_at": int(time.time())
                        })
                        
                        print_position_snapshot("SMART_MONEY_OPEN")
            
            # 8️⃣ التحقق من تحقيق الهدف اليومي
            if trade_manager.daily_profit >= PROFIT_TARGET_DAILY:
                log_g(f"🎉 DAILY PROFIT TARGET ACHIEVED: {trade_manager.daily_profit:.2f} USDT")
                if STATE["open"]:
                    log_i("🔒 Locking profits - closing all positions")
                    close_market_strict("daily_target_achieved")
            
            # الانتظار للدورة التالية
            sleep_time = NEAR_CLOSE_S if time_to_candle_close(df) <= 10 else BASE_SLEEP
            time.sleep(sleep_time)
            
        except Exception as e:
            log_e(f"❌ ULTRA INTELLIGENT TRADING LOOP ERROR: {e}")
            log_e(traceback.format_exc())
            time.sleep(BASE_SLEEP * 2)

# =================== STATE INITIALIZATION ===================
STATE = {
    "open": False, "side": None, "entry": None, "qty": 0.0,
    "pnl": 0.0, "bars": 0, "trail": None, "breakeven": None,
    "tp1_done": False, "highest_profit_pct": 0.0,
    "profit_targets_achieved": 0,
}

compound_pnl = 0.0
wait_for_next_signal_side = None

# =================== FLASK API ===================
app = Flask(__name__)

@app.route("/")
def home():
    return f"""
    <html>
        <head><title>SUI ULTRA PRO AI BOT</title></head>
        <body>
            <h1>🚀 SUI ULTRA PRO AI BOT - الإصدار الذكي المتقدم + SMART MONEY</h1>
            <p><strong>Version:</strong> {BOT_VERSION}</p>
            <p><strong>Exchange:</strong> {EXCHANGE_NAME.upper()}</p>
            <p><strong>Symbol:</strong> {SYMBOL}</p>
            <p><strong>Status:</strong> {'🟢 LIVE' if MODE_LIVE else '🟡 PAPER'}</p>
            <p><strong>Daily PnL:</strong> {trade_manager.daily_profit:.2f} USDT</p>
            <p><strong>Win Rate:</strong> {trade_manager.win_rate:.1f}%</p>
            <p><a href="/health">Health Check</a> | <a href="/metrics">Metrics</a> | <a href="/performance">Performance</a></p>
        </body>
    </html>
    """

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "exchange": EXCHANGE_NAME,
        "symbol": SYMBOL,
        "position_open": STATE["open"],
        "daily_profit": trade_manager.daily_profit,
        "win_rate": trade_manager.win_rate
    })

@app.route("/metrics")
def metrics():
    return jsonify({
        "bot_version": BOT_VERSION,
        "exchange": EXCHANGE_NAME,
        "symbol": SYMBOL,
        "balance": balance_usdt(),
        "daily_profit": trade_manager.daily_profit,
        "win_rate": trade_manager.win_rate,
        "consecutive_wins": trade_manager.consecutive_wins,
        "consecutive_losses": trade_manager.consecutive_losses,
        "total_trades": len(trade_manager.trade_history),
        "position": STATE,
        "performance_suggestions": trade_manager.get_trade_suggestions()
    })

@app.route("/performance")
def performance():
    recent_trades = trade_manager.trade_history[-10:]  # آخر 10 صفقات
    return jsonify({
        "daily_profit": trade_manager.daily_profit,
        "win_rate": trade_manager.win_rate,
        "avg_win": trade_manager.avg_win,
        "avg_loss": trade_manager.avg_loss,
        "recent_trades": [
            {
                "time": t['timestamp'].strftime('%H:%M:%S'),
                "side": t['side'],
                "profit": t['profit'],
                "profit_pct": t['profit_pct']
            } for t in recent_trades
        ]
    })

@app.get("/mark/<color>")
def mark_position(color):
    color = color.lower()
    if color not in ["green", "red"]:
        return jsonify({"ok": False, "error": "Use /mark/green or /mark/red"}), 400
    
    print_position_snapshot(reason="MANUAL_MARK", color=color)
    return jsonify({"ok": True, "marked": color, "timestamp": datetime.now().isoformat()})

# =================== STARTUP ===================
def startup_sequence():
    """تسلسل بدء التشغيل"""
    log_banner("SYSTEM INITIALIZATION")
    
    # تحميل الحالة السابقة
    loaded_state = load_state()
    if loaded_state:
        log_g("✅ Previous state loaded successfully")
    
    # التحقق من اتصال البورصة
    try:
        balance = balance_usdt()
        price = price_now()
        log_g(f"✅ Exchange connection successful")
        log_g(f"💰 Balance: {balance:.2f} USDT")
        log_g(f"💰 Current price: {price:.6f}")
    except Exception as e:
        log_e(f"❌ Exchange connection failed: {e}")
        return False
    
    # عرض إحصائيات البوت
    log_i(f"📊 Performance Metrics:")
    log_i(f"   Win Rate: {trade_manager.win_rate:.1f}%")
    log_i(f"   Daily PnL: {trade_manager.daily_profit:.2f} USDT")
    log_i(f"   Consecutive Wins: {trade_manager.consecutive_wins}")
    log_i(f"   Consecutive Losses: {trade_manager.consecutive_losses}")
    
    log_g("🚀 ULTRA INTELLIGENT TRADING BOT WITH SMART MONEY READY!")
    return True

# =================== MAIN EXECUTION ===================
if __name__ == "__main__":
    # إعداد معالجات الإشارات
    def signal_handler(signum, frame):
        log_i(f"🛑 Received signal {signum} - Shutting down gracefully...")
        save_state(STATE)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # بدء التشغيل
    if startup_sequence():
        # بدء خيوط التنفيذ
        import threading
        
        # خيط التداول الرئيسي
        trading_thread = threading.Thread(target=ultra_intelligent_trading_loop, daemon=True)
        trading_thread.start()
        
        # خيط الحفاظ على الحالة
        def state_saver():
            while True:
                time.sleep(300)  # حفظ كل 5 دقائق
                save_state(STATE)
        
        state_thread = threading.Thread(target=state_saver, daemon=True)
        state_thread.start()
        
        log_g(f"🌐 Starting web server on port {PORT}")
        
        # تشغيل سيرفل الويب
        try:
            app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
        except Exception as e:
            log_e(f"❌ Web server failed: {e}")
    else:
        log_e("❌ Startup failed - check configuration and try again")
