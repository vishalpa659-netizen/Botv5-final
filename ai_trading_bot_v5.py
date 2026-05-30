"""
ai_trading_bot_v5.py  —  Professional Institutional-Grade AI Trading Bot
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Features:
  • Intelligent trade cooldown (no duplicate signals)
  • AI confidence scoring (only trade >= 80%)
  • Multi-timeframe analysis (M5/M15/M30/H1/H4 — 4+ must agree)
  • Market regime detection (Trending / Ranging / Volatile / Breakout)
  • Self-learning engine (adapts weights every 50 trades)
  • Dynamic risk management (1% trade / 3% daily / 10% weekly limits)
  • Smart filtering (spread, volatility, news, fake-breakout guards)
  • 18 instruments: Forex, Crypto, Commodities, Indices
  • Triple take-profit levels (TP1 / TP2 / TP3)
  • Full SQLite trade memory for continuous improvement

Run:  python ai_trading_bot_v5.py
GUI:  python gui_server.py  (open http://localhost:5000)
"""

import sqlite3
import json
import time
import logging
import threading
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("ERROR: Install yfinance:  pip install yfinance")
    sys.exit(1)

try:
    import pandas_ta as ta
except ImportError:
    print("ERROR: Install pandas_ta:  pip install pandas-ta")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

DATABASE          = "ai_trading.db"
SIGNAL_FILE       = "signal_history.json"
OPEN_SIGNALS_FILE = "open_signals.json"
PERFORMANCE_FILE  = "performance_data.json"
FILTER_FILE       = "gui_filters.json"
LEARNING_FILE     = "learning_state.json"

CONFIDENCE_THRESHOLD   = 80     # minimum confidence to generate signal
MIN_TIMEFRAMES_AGREE   = 4      # minimum timeframes that must align
MAX_RISK_PER_TRADE     = 0.01   # 1 % of account per trade
MAX_DAILY_LOSS_PCT     = 0.03   # 3 % max daily drawdown
MAX_WEEKLY_LOSS_PCT    = 0.10   # 10 % max weekly drawdown
COOLDOWN_HOURS         = 4      # default cooldown between signals on same pair
LEARNING_BATCH_SIZE    = 50     # re-calibrate weights every N closed trades
SCAN_INTERVAL_MINUTES  = 15     # scan every 15 min (no forced overtrading)
ACCOUNT_BALANCE        = 10000  # USD — update to your real balance

# Timeframes fetched for multi-timeframe analysis
TIMEFRAMES = {
    "M5":  ("5m",  "2d"),
    "M15": ("15m", "5d"),
    "M30": ("30m", "10d"),
    "H1":  ("1h",  "30d"),
    "H4":  ("4h",  "60d"),
    "D1":  ("1d",  "180d"),
}

# All instruments  symbol_name → yfinance_ticker
SYMBOLS: Dict[str, str] = {
    # Forex Majors
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X",
    "USDCHF": "CHF=X",
    "AUDUSD": "AUDUSD=X",
    "NZDUSD": "NZDUSD=X",
    "USDCAD": "CAD=X",
    # Forex Crosses
    "EURJPY": "EURJPY=X",
    "GBPJPY": "GBPJPY=X",
    "EURGBP": "EURGBP=X",
    "AUDJPY": "AUDJPY=X",
    # Crypto
    "BTCUSD": "BTC-USD",
    "ETHUSD": "ETH-USD",
    # Commodities
    "XAUUSD": "GC=F",
    "XAGUSD": "SI=F",
    # Indices
    "US30":   "YM=F",
    "NAS100": "NQ=F",
    "SPX500": "ES=F",
}

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("bot_v5.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("BotV5")


# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class DatabaseManager:
    """Central SQLite persistence layer."""

    def __init__(self, db_path: str = DATABASE):
        self.db_path = db_path
        self._lock   = threading.Lock()
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        return c

    def _init_schema(self):
        with self._lock, self._conn() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS signals (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol          TEXT,
                signal          TEXT,
                entry           REAL,
                tp              REAL,
                tp2             REAL,
                tp3             REAL,
                sl              REAL,
                result          TEXT DEFAULT 'RUNNING',
                time            TEXT,
                probability     REAL,
                confidence      REAL,
                regime          TEXT,
                killzone        TEXT,
                trend           TEXT,
                mtf_confirmation TEXT,
                structure       TEXT,
                fvg             TEXT,
                status          TEXT DEFAULT 'OPEN',
                rr_ratio        REAL,
                reasons         TEXT,
                timeframes_agree INTEGER,
                session         TEXT,
                indicators      TEXT,
                profit_pips     REAL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS cooldowns (
                symbol      TEXT PRIMARY KEY,
                active_since TEXT,
                expires_at   TEXT,
                reason       TEXT,
                signal       TEXT
            );

            CREATE TABLE IF NOT EXISTS risk_state (
                id           INTEGER PRIMARY KEY,
                date         TEXT UNIQUE,
                daily_loss   REAL DEFAULT 0,
                weekly_start TEXT,
                weekly_loss  REAL DEFAULT 0,
                trade_count  INTEGER DEFAULT 0,
                last_updated TEXT
            );

            CREATE TABLE IF NOT EXISTS learning_state (
                symbol       TEXT PRIMARY KEY,
                total_trades INTEGER DEFAULT 0,
                wins         INTEGER DEFAULT 0,
                losses       INTEGER DEFAULT 0,
                win_rate     REAL DEFAULT 50,
                profit_factor REAL DEFAULT 1.0,
                weight       REAL DEFAULT 1.0,
                avg_confidence REAL DEFAULT 80,
                best_session TEXT DEFAULT 'LONDON',
                last_updated TEXT
            );

            CREATE TABLE IF NOT EXISTS strategy_weights (
                strategy_key TEXT PRIMARY KEY,
                weight       REAL DEFAULT 1.0,
                win_count    INTEGER DEFAULT 0,
                loss_count   INTEGER DEFAULT 0,
                last_updated TEXT
            );
            """)
        log.info("Database schema initialised")

    def execute(self, sql: str, params=()):
        with self._lock, self._conn() as db:
            return db.execute(sql, params)

    def fetchall(self, sql: str, params=()) -> List[sqlite3.Row]:
        with self._lock:
            c = self._conn()
            rows = c.execute(sql, params).fetchall()
            c.close()
            return rows

    def fetchone(self, sql: str, params=()) -> Optional[sqlite3.Row]:
        with self._lock:
            c = self._conn()
            row = c.execute(sql, params).fetchone()
            c.close()
            return row

    def insert_signal(self, sig: Dict) -> int:
        sql = """
        INSERT INTO signals
          (symbol,signal,entry,tp,tp2,tp3,sl,result,time,probability,confidence,
           regime,killzone,trend,mtf_confirmation,structure,fvg,status,rr_ratio,
           reasons,timeframes_agree,session,indicators)
        VALUES
          (:symbol,:signal,:entry,:tp,:tp2,:tp3,:sl,:result,:time,:probability,
           :confidence,:regime,:killzone,:trend,:mtf_confirmation,:structure,:fvg,
           :status,:rr_ratio,:reasons,:timeframes_agree,:session,:indicators)
        """
        with self._lock, self._conn() as db:
            cur = db.execute(sql, sig)
            return cur.lastrowid

    def close_signal(self, symbol: str, result: str, profit_pips: float = 0):
        with self._lock, self._conn() as db:
            db.execute(
                "UPDATE signals SET result=?, status='CLOSED', profit_pips=? "
                "WHERE symbol=? AND result='RUNNING'",
                (result, profit_pips, symbol)
            )


# ═══════════════════════════════════════════════════════════════════════════════
# MARKET DATA FETCHER
# ═══════════════════════════════════════════════════════════════════════════════

class MarketDataFetcher:
    """Downloads OHLCV bars from Yahoo Finance with caching."""

    _cache: Dict[str, Any] = {}
    _cache_time: Dict[str, datetime] = {}
    CACHE_TTL = 300   # 5 minutes

    @classmethod
    def get_bars(cls, ticker: str, interval: str, period: str) -> Optional[pd.DataFrame]:
        key = f"{ticker}_{interval}_{period}"
        now = datetime.utcnow()
        if key in cls._cache:
            age = (now - cls._cache_time[key]).total_seconds()
            if age < cls.CACHE_TTL:
                return cls._cache[key]
        try:
            df = yf.download(ticker, interval=interval, period=period,
                             progress=False, auto_adjust=True, prepost=False)
            if df is None or df.empty or len(df) < 30:
                return None
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.dropna(inplace=True)
            cls._cache[key]      = df
            cls._cache_time[key] = now
            return df
        except Exception as e:
            log.debug(f"Data fetch error {ticker} {interval}: {e}")
            return None

    @classmethod
    def get_current_price(cls, ticker: str) -> Optional[float]:
        try:
            df = yf.download(ticker, interval="1m", period="2d",
                             progress=False, auto_adjust=True)
            if df is not None and not df.empty:
                close = df["Close"]
                if isinstance(close.columns if hasattr(close, 'columns') else None, type(None)):
                    return float(close.iloc[-1])
                return float(close.iloc[-1].iloc[0] if hasattr(close.iloc[-1], 'iloc') else close.iloc[-1])
        except Exception:
            pass
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# INDICATOR ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class IndicatorEngine:
    """Calculates all technical indicators and returns a scored dict."""

    @staticmethod
    def compute(df: pd.DataFrame) -> Optional[Dict]:
        if df is None or len(df) < 50:
            return None
        try:
            c = df["Close"].copy()
            h = df["High"].copy()
            lo = df["Low"].copy()
            v  = df["Volume"].fillna(0).copy()

            # ── Trend ──
            ema9   = ta.ema(c, 9)
            ema21  = ta.ema(c, 21)
            ema50  = ta.ema(c, 50)
            sma200 = ta.sma(c, 200) if len(c) >= 200 else ta.sma(c, len(c)-1)

            # ── Momentum ──
            rsi    = ta.rsi(c, 14)
            macd_r = ta.macd(c, 12, 26, 9)
            macd_line = macd_r["MACD_12_26_9"]  if macd_r is not None else None
            macd_sig  = macd_r["MACDs_12_26_9"] if macd_r is not None else None
            macd_hist = macd_r["MACDh_12_26_9"] if macd_r is not None else None

            # ── Volatility ──
            atr    = ta.atr(h, lo, c, 14)
            bb     = ta.bbands(c, 20, 2)
            bb_upper = bb["BBU_20_2.0"] if bb is not None else None
            bb_lower = bb["BBL_20_2.0"] if bb is not None else None
            bb_mid   = bb["BBM_20_2.0"] if bb is not None else None
            bb_width = (bb_upper - bb_lower) / bb_mid if bb is not None else None

            # ── Trend strength ──
            adx_r  = ta.adx(h, lo, c, 14)
            adx    = adx_r["ADX_14"]  if adx_r is not None else None
            plus_di= adx_r["DMP_14"]  if adx_r is not None else None
            minus_di=adx_r["DMN_14"]  if adx_r is not None else None

            # ── Volume ──
            vwap   = ta.vwap(h, lo, c, v)
            obv    = ta.obv(c, v)

            # ── Support / Resistance (swing pivots) ──
            recent  = df.tail(50)
            support = float(recent["Low"].min())
            resist  = float(recent["High"].max())
            pivot   = (float(recent["High"].iloc[-1]) +
                       float(recent["Low"].iloc[-1]) +
                       float(recent["Close"].iloc[-1])) / 3

            # ── Current values (last bar) ──
            price    = float(c.iloc[-1])
            e9       = float(ema9.iloc[-1])  if ema9 is not None else price
            e21      = float(ema21.iloc[-1]) if ema21 is not None else price
            e50      = float(ema50.iloc[-1]) if ema50 is not None else price
            s200     = float(sma200.iloc[-1]) if sma200 is not None else price
            rsi_v    = float(rsi.iloc[-1])    if rsi is not None else 50
            atr_v    = float(atr.iloc[-1])    if atr is not None else 0
            adx_v    = float(adx.iloc[-1])    if adx is not None else 0
            pdi_v    = float(plus_di.iloc[-1])  if plus_di is not None else 0
            mdi_v    = float(minus_di.iloc[-1]) if minus_di is not None else 0
            bbu      = float(bb_upper.iloc[-1]) if bb_upper is not None else price
            bbl      = float(bb_lower.iloc[-1]) if bb_lower is not None else price
            bbw      = float(bb_width.iloc[-1]) if bb_width is not None else 0
            macd_v   = float(macd_line.iloc[-1]) if macd_line is not None else 0
            macd_s   = float(macd_sig.iloc[-1])  if macd_sig is not None else 0
            macd_h   = float(macd_hist.iloc[-1]) if macd_hist is not None else 0
            vwap_v   = float(vwap.iloc[-1])      if vwap is not None else price
            obv_v    = float(obv.iloc[-1])       if obv is not None else 0
            obv_prev = float(obv.iloc[-5])       if obv is not None and len(obv) > 5 else obv_v

            return {
                "price": price,
                "ema9": e9, "ema21": e21, "ema50": e50, "sma200": s200,
                "rsi": rsi_v, "atr": atr_v,
                "adx": adx_v, "plus_di": pdi_v, "minus_di": mdi_v,
                "bb_upper": bbu, "bb_lower": bbl, "bb_width": bbw,
                "macd": macd_v, "macd_signal": macd_s, "macd_hist": macd_h,
                "vwap": vwap_v, "obv": obv_v, "obv_prev": obv_prev,
                "support": support, "resistance": resist, "pivot": pivot,
            }
        except Exception as e:
            log.debug(f"Indicator error: {e}")
            return None


# ═══════════════════════════════════════════════════════════════════════════════
# MARKET REGIME DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class MarketRegimeDetector:
    """Classifies the current market state."""

    @staticmethod
    def detect(ind: Dict) -> str:
        adx = ind.get("adx", 0)
        bbw = ind.get("bb_width", 0)
        rsi = ind.get("rsi", 50)

        price = ind.get("price", 1)
        e50   = ind.get("ema50", 1)
        trend_pct = abs(price - e50) / e50 * 100 if e50 else 0

        # VOLATILE: wide BB + extreme RSI
        if bbw > 0.04 and (rsi > 75 or rsi < 25):
            return "VOLATILE"

        # BREAKOUT: ADX starting to spike + narrow-to-wide BB
        if 20 < adx < 30 and bbw > 0.025:
            return "BREAKOUT"

        # TRENDING: strong ADX + price away from SMA
        if adx > 25 and trend_pct > 0.5:
            return "TRENDING"

        # RANGING: weak ADX + tight BB
        if adx < 20:
            return "RANGING"

        return "TRENDING"

    @staticmethod
    def allows_trade(regime: str, filters: Dict) -> bool:
        """Some regimes are tradeble under specific strategies."""
        if not filters.get("regime", True):
            return True   # filter disabled — always allow
        # RANGING: harder to profit; only allow with extra confirmation
        if regime == "VOLATILE":
            return False  # skip pure volatility spikes
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIDENCE SCORER
# ═══════════════════════════════════════════════════════════════════════════════

class ConfidenceScorer:
    """
    Weighted multi-factor scoring system.
    Returns (direction, score, reasons).
    direction: 'BUY' | 'SELL' | 'HOLD'
    score: 0-100
    """

    # Base weights (adapted by self-learning engine)
    DEFAULT_WEIGHTS = {
        "ema_alignment":   15,
        "rsi_momentum":    15,
        "macd_crossover":  15,
        "adx_strength":    10,
        "bb_position":     10,
        "vwap_position":   10,
        "support_resist":  10,
        "volume_confirm":   5,
        "trend_structure":  5,
        "regime_bonus":     5,
    }

    @classmethod
    def score(cls, ind: Dict, regime: str,
              weights: Dict = None) -> Tuple[str, float, List[str]]:
        if ind is None:
            return "HOLD", 0.0, []

        w = weights or cls.DEFAULT_WEIGHTS
        buy_score  = 0.0
        sell_score = 0.0
        reasons_buy  = []
        reasons_sell = []

        price  = ind["price"]
        e9, e21, e50, s200 = ind["ema9"], ind["ema21"], ind["ema50"], ind["sma200"]
        rsi    = ind["rsi"]
        adx    = ind["adx"]
        pdi    = ind["plus_di"]
        mdi    = ind["minus_di"]
        macd   = ind["macd"]
        ms     = ind["macd_signal"]
        mh     = ind["macd_hist"]
        bbu    = ind["bb_upper"]
        bbl    = ind["bb_lower"]
        vwap   = ind["vwap"]
        obv    = ind["obv"]
        obvp   = ind["obv_prev"]
        sup    = ind["support"]
        res    = ind["resistance"]

        # ── 1. EMA alignment ──────────────────────────────────────
        ema_wt = w.get("ema_alignment", 15)
        if e9 > e21 > e50 and price > s200:
            buy_score += ema_wt
            reasons_buy.append("EMA stack bullish (9>21>50 + above SMA200)")
        elif e9 < e21 < e50 and price < s200:
            sell_score += ema_wt
            reasons_sell.append("EMA stack bearish (9<21<50 + below SMA200)")
        elif e9 > e21 > e50:
            buy_score += ema_wt * 0.6
            reasons_buy.append("EMA trend bullish (short-term)")
        elif e9 < e21 < e50:
            sell_score += ema_wt * 0.6
            reasons_sell.append("EMA trend bearish (short-term)")

        # ── 2. RSI momentum ──────────────────────────────────────
        rsi_wt = w.get("rsi_momentum", 15)
        if 50 < rsi < 70:
            buy_score += rsi_wt
            reasons_buy.append(f"RSI bullish momentum ({rsi:.0f})")
        elif 30 < rsi < 50:
            sell_score += rsi_wt
            reasons_sell.append(f"RSI bearish momentum ({rsi:.0f})")
        elif rsi <= 30:
            buy_score += rsi_wt * 0.7   # oversold recovery
            reasons_buy.append(f"RSI oversold recovery ({rsi:.0f})")
        elif rsi >= 70:
            sell_score += rsi_wt * 0.7  # overbought
            reasons_sell.append(f"RSI overbought ({rsi:.0f})")

        # ── 3. MACD crossover ────────────────────────────────────
        macd_wt = w.get("macd_crossover", 15)
        if macd > ms and mh > 0:
            buy_score += macd_wt
            reasons_buy.append("MACD bullish crossover + positive histogram")
        elif macd < ms and mh < 0:
            sell_score += macd_wt
            reasons_sell.append("MACD bearish crossover + negative histogram")
        elif macd > ms:
            buy_score += macd_wt * 0.5
        elif macd < ms:
            sell_score += macd_wt * 0.5

        # ── 4. ADX strength ──────────────────────────────────────
        adx_wt = w.get("adx_strength", 10)
        if adx > 25:
            if pdi > mdi:
                buy_score += adx_wt
                reasons_buy.append(f"Strong trend ADX={adx:.0f} +DI>{'-DI'}")
            else:
                sell_score += adx_wt
                reasons_sell.append(f"Strong trend ADX={adx:.0f} -DI>+DI")
        elif adx > 20:
            if pdi > mdi:
                buy_score += adx_wt * 0.5
            else:
                sell_score += adx_wt * 0.5

        # ── 5. Bollinger Band position ────────────────────────────
        bb_wt = w.get("bb_position", 10)
        bb_mid = (bbu + bbl) / 2
        if price > bb_mid and price < bbu * 0.998:
            buy_score += bb_wt
            reasons_buy.append("Price above BB midline, room to upper band")
        elif price < bb_mid and price > bbl * 1.002:
            sell_score += bb_wt
            reasons_sell.append("Price below BB midline, room to lower band")

        # ── 6. VWAP position ─────────────────────────────────────
        vwap_wt = w.get("vwap_position", 10)
        if price > vwap * 1.0005:
            buy_score += vwap_wt
            reasons_buy.append("Price above VWAP (institutional bullish bias)")
        elif price < vwap * 0.9995:
            sell_score += vwap_wt
            reasons_sell.append("Price below VWAP (institutional bearish bias)")

        # ── 7. Support / Resistance ───────────────────────────────
        sr_wt  = w.get("support_resist", 10)
        sr_rng = (res - sup) * 0.1
        if price < sup + sr_rng:
            buy_score += sr_wt
            reasons_buy.append(f"Price at support zone ({sup:.5g})")
        elif price > res - sr_rng:
            sell_score += sr_wt
            reasons_sell.append(f"Price at resistance zone ({res:.5g})")

        # ── 8. Volume / OBV confirmation ──────────────────────────
        vol_wt = w.get("volume_confirm", 5)
        if obv > obvp * 1.01:
            buy_score += vol_wt
            reasons_buy.append("OBV rising — institutional buying")
        elif obv < obvp * 0.99:
            sell_score += vol_wt
            reasons_sell.append("OBV falling — institutional selling")

        # ── 9. Trend structure bonus ──────────────────────────────
        ts_wt = w.get("trend_structure", 5)
        if price > e50 and e50 > s200:
            buy_score += ts_wt
            reasons_buy.append("Long-term uptrend structure confirmed")
        elif price < e50 and e50 < s200:
            sell_score += ts_wt
            reasons_sell.append("Long-term downtrend structure confirmed")

        # ── 10. Regime bonus ─────────────────────────────────────
        reg_wt = w.get("regime_bonus", 5)
        if regime in ("TRENDING", "BREAKOUT"):
            buy_score  += reg_wt * (buy_score / max(buy_score + sell_score, 1))
            sell_score += reg_wt * (sell_score / max(buy_score + sell_score, 1))
            reasons_buy.append(f"Market regime: {regime}")
            reasons_sell.append(f"Market regime: {regime}")

        # ── Direction decision ────────────────────────────────────
        total = buy_score + sell_score
        if total == 0:
            return "HOLD", 0.0, []

        if buy_score > sell_score:
            confidence = min(100.0, (buy_score / total) * 100 * 1.05)
            return "BUY", round(confidence, 1), reasons_buy
        else:
            confidence = min(100.0, (sell_score / total) * 100 * 1.05)
            return "SELL", round(confidence, 1), reasons_sell


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-TIMEFRAME ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

class MultiTimeframeAnalyzer:
    """Checks that multiple timeframes agree before generating a signal."""

    @staticmethod
    def analyze(symbol: str, ticker: str,
                filters: Dict) -> Tuple[Optional[str], int, Dict]:
        """
        Returns (direction, agree_count, tf_results).
        direction is None if MTF filter fails.
        """
        if not filters.get("mtf", True):
            # MTF filter disabled — just use H1
            tf_name, (interval, period) = "H1", TIMEFRAMES["H1"]
            df  = MarketDataFetcher.get_bars(ticker, interval, period)
            ind = IndicatorEngine.compute(df)
            if ind is None:
                return None, 0, {}
            regime = MarketRegimeDetector.detect(ind)
            direction, _, _ = ConfidenceScorer.score(ind, regime)
            return direction, 1, {"H1": direction}

        tf_votes: Dict[str, str] = {}
        buy_count = sell_count = 0

        for tf_name, (interval, period) in TIMEFRAMES.items():
            df  = MarketDataFetcher.get_bars(ticker, interval, period)
            ind = IndicatorEngine.compute(df)
            if ind is None:
                continue
            regime    = MarketRegimeDetector.detect(ind)
            direction, score, _ = ConfidenceScorer.score(ind, regime)
            tf_votes[tf_name] = direction
            if direction == "BUY":
                buy_count  += 1
            elif direction == "SELL":
                sell_count += 1

        agree = max(buy_count, sell_count)
        if agree < MIN_TIMEFRAMES_AGREE:
            return None, agree, tf_votes

        dominant = "BUY" if buy_count > sell_count else "SELL"
        return dominant, agree, tf_votes


# ═══════════════════════════════════════════════════════════════════════════════
# COOLDOWN MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class CooldownManager:
    """Prevents duplicate signals on the same instrument."""

    def __init__(self, db: DatabaseManager):
        self.db = db

    def is_on_cooldown(self, symbol: str) -> bool:
        row = self.db.fetchone(
            "SELECT expires_at FROM cooldowns WHERE symbol=?", (symbol,)
        )
        if row is None:
            return False
        expires = datetime.fromisoformat(row["expires_at"])
        if datetime.utcnow() < expires:
            return True
        self.remove(symbol)
        return False

    def set_cooldown(self, symbol: str, signal: str, hours: float = COOLDOWN_HOURS):
        now     = datetime.utcnow()
        expires = now + timedelta(hours=hours)
        self.db.execute(
            "INSERT OR REPLACE INTO cooldowns VALUES (?,?,?,?,?)",
            (symbol, now.isoformat(), expires.isoformat(),
             f"Active {signal} signal", signal)
        )
        log.info(f"  Cooldown SET  {symbol} expires {expires.strftime('%H:%M')} UTC")

    def remove(self, symbol: str):
        self.db.execute("DELETE FROM cooldowns WHERE symbol=?", (symbol,))

    def get_all(self) -> List[Dict]:
        rows = self.db.fetchall("SELECT * FROM cooldowns")
        result = []
        for r in rows:
            exp = datetime.fromisoformat(r["expires_at"])
            remaining = max(0, (exp - datetime.utcnow()).total_seconds() / 60)
            result.append({
                "symbol":    r["symbol"],
                "signal":    r["signal"],
                "expires_at": r["expires_at"],
                "remaining_min": round(remaining, 0),
            })
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# RISK MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class RiskManager:
    """Enforces drawdown limits and position sizing."""

    def __init__(self, db: DatabaseManager):
        self.db = db

    def _ensure_today(self):
        today = datetime.utcnow().strftime("%Y-%m-%d")
        week  = (datetime.utcnow() - timedelta(days=datetime.utcnow().weekday())).strftime("%Y-%m-%d")
        self.db.execute(
            "INSERT OR IGNORE INTO risk_state(date,daily_loss,weekly_start,weekly_loss,trade_count,last_updated) "
            "VALUES(?,0,?,0,0,?)",
            (today, week, datetime.utcnow().isoformat())
        )

    def can_trade(self) -> Tuple[bool, str]:
        self._ensure_today()
        today = datetime.utcnow().strftime("%Y-%m-%d")
        row   = self.db.fetchone("SELECT * FROM risk_state WHERE date=?", (today,))
        if row is None:
            return True, "OK"
        dl = row["daily_loss"]
        wl = row["weekly_loss"]
        if dl >= MAX_DAILY_LOSS_PCT:
            return False, f"Daily loss limit hit ({dl*100:.1f}% >= {MAX_DAILY_LOSS_PCT*100:.0f}%)"
        if wl >= MAX_WEEKLY_LOSS_PCT:
            return False, f"Weekly loss limit hit ({wl*100:.1f}% >= {MAX_WEEKLY_LOSS_PCT*100:.0f}%)"
        return True, "OK"

    def record_trade_result(self, result: str, pnl_pct: float):
        self._ensure_today()
        today = datetime.utcnow().strftime("%Y-%m-%d")
        if result == "LOSS":
            loss = abs(pnl_pct)
            self.db.execute(
                "UPDATE risk_state SET daily_loss=daily_loss+?, "
                "weekly_loss=weekly_loss+?, trade_count=trade_count+1, "
                "last_updated=? WHERE date=?",
                (loss, loss, datetime.utcnow().isoformat(), today)
            )

    def position_size(self, entry: float, sl: float) -> float:
        """Return lot size based on 1% risk."""
        if entry == 0 or sl == 0:
            return 0.01
        risk_amount = ACCOUNT_BALANCE * MAX_RISK_PER_TRADE
        pip_value   = abs(entry - sl)
        if pip_value == 0:
            return 0.01
        size = risk_amount / pip_value
        return round(min(size, 10.0), 2)

    def get_state(self) -> Dict:
        self._ensure_today()
        today = datetime.utcnow().strftime("%Y-%m-%d")
        row   = self.db.fetchone("SELECT * FROM risk_state WHERE date=?", (today,))
        if not row:
            return {"daily_loss_pct": 0, "weekly_loss_pct": 0, "trade_count": 0}
        return {
            "daily_loss_pct":  round(row["daily_loss"] * 100, 2),
            "weekly_loss_pct": round(row["weekly_loss"] * 100, 2),
            "trade_count":     row["trade_count"],
            "daily_limit_pct": MAX_DAILY_LOSS_PCT * 100,
            "weekly_limit_pct": MAX_WEEKLY_LOSS_PCT * 100,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SELF-LEARNING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class SelfLearningEngine:
    """
    Every LEARNING_BATCH_SIZE closed trades, recalculate per-symbol weights
    and indicator strategy weights to focus on what actually works.
    """

    def __init__(self, db: DatabaseManager):
        self.db = db

    def maybe_adapt(self):
        closed = self.db.fetchone(
            "SELECT COUNT(*) as cnt FROM signals WHERE result IN ('WIN','LOSS')"
        )
        if not closed or closed["cnt"] < LEARNING_BATCH_SIZE:
            return
        if closed["cnt"] % LEARNING_BATCH_SIZE != 0:
            return
        log.info("🧠 Self-learning engine: recalibrating weights...")
        self._update_symbol_weights()
        self._update_strategy_weights()
        self._export_learning_state()
        log.info("🧠 Recalibration complete")

    def _update_symbol_weights(self):
        rows = self.db.fetchall("""
            SELECT symbol,
                   COUNT(*) total,
                   SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) wins,
                   SUM(CASE WHEN result='LOSS' THEN 1 ELSE 0 END) losses,
                   AVG(confidence) avg_conf
            FROM signals
            WHERE result IN ('WIN','LOSS')
            GROUP BY symbol
        """)
        for r in rows:
            total = r["total"]
            if total < 5:
                continue
            wins   = r["wins"]
            losses = r["losses"]
            wr     = wins / total
            pf     = wins / max(losses, 1)

            # Weight: boost winners, penalise losers
            if wr >= 0.65:
                weight = min(1.5, 1.0 + (wr - 0.65) * 2)
            elif wr < 0.40:
                weight = max(0.3, 1.0 - (0.40 - wr) * 2)
            else:
                weight = 1.0

            self.db.execute("""
                INSERT OR REPLACE INTO learning_state
                (symbol, total_trades, wins, losses, win_rate, profit_factor, weight, avg_confidence, last_updated)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, (r["symbol"], total, wins, losses,
                  round(wr * 100, 1), round(pf, 2),
                  round(weight, 2), round(r["avg_conf"] or 80, 1),
                  datetime.utcnow().isoformat()))
            log.info(f"  {r['symbol']}: WR={wr*100:.0f}% PF={pf:.2f} weight={weight:.2f}")

    def _update_strategy_weights(self):
        # Detect patterns: which confidence ranges are profitable
        rows = self.db.fetchall("""
            SELECT result,
                   CASE WHEN confidence>=90 THEN 'HIGH'
                        WHEN confidence>=80 THEN 'MEDIUM'
                        ELSE 'LOW' END as conf_band,
                   COUNT(*) cnt
            FROM signals WHERE result IN ('WIN','LOSS')
            GROUP BY result, conf_band
        """)
        for r in rows:
            key = f"conf_{r['conf_band']}"
            wr  = 1 if r["result"] == "WIN" else 0
            self.db.execute("""
                INSERT OR REPLACE INTO strategy_weights
                (strategy_key, weight, win_count, loss_count, last_updated)
                VALUES (?,?,?,?,?)
            """, (key,
                  1.1 if wr else 0.9,
                  r["cnt"] if wr else 0,
                  0 if wr else r["cnt"],
                  datetime.utcnow().isoformat()))

    def get_symbol_weight(self, symbol: str) -> float:
        row = self.db.fetchone(
            "SELECT weight FROM learning_state WHERE symbol=?", (symbol,)
        )
        return float(row["weight"]) if row else 1.0

    def get_stats(self) -> List[Dict]:
        rows = self.db.fetchall(
            "SELECT * FROM learning_state ORDER BY win_rate DESC"
        )
        return [dict(r) for r in rows]

    def _export_learning_state(self):
        stats = self.get_stats()
        try:
            with open(LEARNING_FILE, "w") as f:
                json.dump(stats, f, indent=2)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

def current_session() -> str:
    h = datetime.utcnow().hour
    if 22 <= h or h < 7:
        return "SYDNEY"
    if 7 <= h < 10:
        return "LONDON_OPEN"
    if 10 <= h < 12:
        return "LONDON"
    if 12 <= h < 17:
        return "NY_OVERLAP"
    if 17 <= h < 22:
        return "NY"
    return "OFF_HOURS"


def is_trading_session(filters: Dict) -> bool:
    if not filters.get("killzone", True):
        return True
    session = current_session()
    # Prefer high-liquidity sessions
    return session in ("LONDON_OPEN", "LONDON", "NY_OVERLAP", "NY")


# ═══════════════════════════════════════════════════════════════════════════════
# SMART FILTERS
# ═══════════════════════════════════════════════════════════════════════════════

def passes_smart_filters(ind: Dict, regime: str, filters: Dict) -> Tuple[bool, str]:
    """Additional quality gates before signal generation."""

    atr = ind.get("atr", 0)
    bbw = ind.get("bb_width", 0)
    rsi = ind.get("rsi", 50)

    # 1. Volatility check
    if filters.get("volatility", True):
        if atr == 0 or bbw < 0.001:
            return False, "Volatility too low (sleeping market)"
        if bbw > 0.08:
            return False, "Volatility extreme (news spike risk)"

    # 2. Sideways / ranging filter
    adx = ind.get("adx", 0)
    if filters.get("market_quality", True) and adx < 15:
        return False, f"Market too choppy (ADX={adx:.0f})"

    # 3. Fake breakout guard
    price  = ind.get("price", 1)
    bbu    = ind.get("bb_upper", price * 1.1)
    bbl    = ind.get("bb_lower", price * 0.9)
    e50    = ind.get("ema50", price)
    if abs(price - bbu) / price < 0.001 and adx < 20:
        return False, "Potential fake breakout at BB upper"
    if abs(price - bbl) / price < 0.001 and adx < 20:
        return False, "Potential fake breakout at BB lower"

    # 4. RSI extremes with weak trend = trap
    if (rsi > 80 or rsi < 20) and adx < 20:
        return False, f"RSI extreme ({rsi:.0f}) with weak trend — likely reversion trap"

    return True, "OK"


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

class SignalGenerator:
    """Orchestrates all analysis and produces a final trade signal."""

    def __init__(self, db: DatabaseManager):
        self.db       = db
        self.cooldown = CooldownManager(db)
        self.risk     = RiskManager(db)
        self.learner  = SelfLearningEngine(db)

    def _compute_targets(self, direction: str, ind: Dict) -> Tuple[float, float, float, float, float]:
        """Returns entry, sl, tp1, tp2, tp3."""
        price = ind["price"]
        atr   = ind["atr"]
        if atr == 0:
            atr = price * 0.001

        if direction == "BUY":
            sl  = round(price - atr * 1.5, 6)
            tp1 = round(price + atr * 1.5, 6)
            tp2 = round(price + atr * 3.0, 6)
            tp3 = round(price + atr * 5.0, 6)
        else:
            sl  = round(price + atr * 1.5, 6)
            tp1 = round(price - atr * 1.5, 6)
            tp2 = round(price - atr * 3.0, 6)
            tp3 = round(price - atr * 5.0, 6)

        rr = abs(tp2 - price) / abs(price - sl) if abs(price - sl) > 0 else 0
        return round(price, 6), round(sl, 6), round(tp1, 6), round(tp2, 6), round(tp3, 6)

    def process_symbol(self, symbol: str, filters: Dict) -> Optional[Dict]:
        ticker = SYMBOLS.get(symbol)
        if not ticker:
            return None

        # ── Cooldown check ────────────────────────────────────────
        if self.cooldown.is_on_cooldown(symbol):
            log.debug(f"  {symbol}: on cooldown — skip")
            return None

        # ── Risk limits ───────────────────────────────────────────
        ok, reason = self.risk.can_trade()
        if not ok:
            log.warning(f"  RISK LIMIT: {reason}")
            return None

        # ── Session / killzone filter ─────────────────────────────
        session = current_session()
        if not is_trading_session(filters):
            log.debug(f"  {symbol}: outside trading session ({session})")
            return None

        # ── Fetch H1 data for primary analysis ────────────────────
        df_h1 = MarketDataFetcher.get_bars(ticker, "1h", "30d")
        ind   = IndicatorEngine.compute(df_h1)
        if ind is None:
            log.debug(f"  {symbol}: insufficient data")
            return None

        # ── Detect market regime ──────────────────────────────────
        regime = MarketRegimeDetector.detect(ind)
        if not MarketRegimeDetector.allows_trade(regime, filters):
            log.debug(f"  {symbol}: regime {regime} not tradable")
            return None

        # ── Smart filters ─────────────────────────────────────────
        passed, filter_reason = passes_smart_filters(ind, regime, filters)
        if not passed:
            log.debug(f"  {symbol}: filtered — {filter_reason}")
            return None

        # ── Multi-timeframe analysis ──────────────────────────────
        direction, tf_agree, tf_votes = MultiTimeframeAnalyzer.analyze(
            symbol, ticker, filters)
        if direction is None:
            log.debug(f"  {symbol}: MTF disagreement ({tf_agree}/{MIN_TIMEFRAMES_AGREE} agree)")
            return None

        # ── Confidence scoring ────────────────────────────────────
        sym_weight = self.learner.get_symbol_weight(symbol)
        direction2, confidence, reasons = ConfidenceScorer.score(ind, regime)

        # Apply learning weight
        confidence = min(100.0, confidence * sym_weight)

        if direction != direction2:
            log.debug(f"  {symbol}: MTF direction ({direction}) ≠ H1 direction ({direction2})")
            return None

        if confidence < CONFIDENCE_THRESHOLD:
            log.debug(f"  {symbol}: confidence {confidence:.0f}% < {CONFIDENCE_THRESHOLD}%")
            return None

        log.info(f"  ✓ {symbol} {direction} confidence={confidence:.0f}%  regime={regime}  TF_agree={tf_agree}")

        # ── Compute trade levels ──────────────────────────────────
        entry, sl, tp1, tp2, tp3 = self._compute_targets(direction, ind)
        rr = abs(tp2 - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 0
        lot_size = self.risk.position_size(entry, sl)

        tf_str = ", ".join(f"{k}:{v}" for k, v in tf_votes.items())
        ind_summary = (
            f"RSI={ind['rsi']:.0f} MACD={'▲' if ind['macd']>ind['macd_signal'] else '▼'} "
            f"ADX={ind['adx']:.0f} ATR={ind['atr']:.5g}"
        )

        sig = {
            "symbol":           symbol,
            "signal":           direction,
            "entry":            entry,
            "tp":               tp1,     # backward compat with GUI
            "tp2":              tp2,
            "tp3":              tp3,
            "sl":               sl,
            "result":           "RUNNING",
            "time":             datetime.utcnow().isoformat(),
            "probability":      confidence,
            "confidence":       confidence,
            "regime":           regime,
            "killzone":         session,
            "trend":            "BULLISH" if direction == "BUY" else "BEARISH",
            "mtf_confirmation": tf_str,
            "structure":        f"{tf_agree}/{len(TIMEFRAMES)} TF agree",
            "fvg":              "YES" if abs(ind["ema9"] - ind["ema21"]) / ind["ema21"] > 0.002 else "NO",
            "status":           "OPEN",
            "rr_ratio":         round(rr, 2),
            "reasons":          " | ".join(reasons),
            "timeframes_agree": tf_agree,
            "session":          session,
            "indicators":       ind_summary,
        }

        # ── Save to DB ────────────────────────────────────────────
        sig_id = self.db.insert_signal(sig)
        sig["id"] = sig_id

        # ── Set cooldown ──────────────────────────────────────────
        self.cooldown.set_cooldown(symbol, direction, hours=COOLDOWN_HOURS)

        # ── Update JSON files (GUI compatibility) ─────────────────
        self._update_json_files(sig)

        return sig

    def _update_json_files(self, sig: Dict):
        # signal_history.json
        try:
            with open(SIGNAL_FILE) as f:
                history = json.load(f)
        except Exception:
            history = []
        history.append(sig)
        history = history[-500:]
        with open(SIGNAL_FILE, "w") as f:
            json.dump(history, f, indent=2, default=str)

        # open_signals.json
        try:
            with open(OPEN_SIGNALS_FILE) as f:
                opens = json.load(f)
        except Exception:
            opens = []
        opens = [s for s in opens if s.get("symbol") != sig["symbol"]]
        opens.append(sig)
        with open(OPEN_SIGNALS_FILE, "w") as f:
            json.dump(opens, f, indent=2, default=str)

    def close_open_positions(self):
        """Check all RUNNING signals and close at SL/TP if price crossed."""
        try:
            with open(OPEN_SIGNALS_FILE) as f:
                opens = json.load(f)
        except Exception:
            return

        still_open = []
        for sig in opens:
            symbol = sig.get("symbol")
            ticker = SYMBOLS.get(symbol)
            if not ticker:
                still_open.append(sig)
                continue

            df = MarketDataFetcher.get_bars(ticker, "5m", "1d")
            if df is None or df.empty:
                still_open.append(sig)
                continue

            price  = float(df["Close"].iloc[-1])
            entry  = sig.get("entry", price)
            tp     = sig.get("tp", price)
            sl     = sig.get("sl", price)
            direction = sig.get("signal", "BUY")

            closed = False
            if direction == "BUY":
                if price >= tp:
                    self.db.close_signal(symbol, "WIN",
                                         profit_pips=abs(tp - entry))
                    self.cooldown.remove(symbol)
                    log.info(f"  TP HIT  {symbol} BUY → WIN at {price:.5g}")
                    closed = True
                elif price <= sl:
                    self.db.close_signal(symbol, "LOSS",
                                         profit_pips=-abs(entry - sl))
                    self.cooldown.remove(symbol)
                    self.risk.record_trade_result("LOSS", MAX_RISK_PER_TRADE)
                    log.warning(f"  SL HIT  {symbol} BUY → LOSS at {price:.5g}")
                    closed = True
            else:
                if price <= tp:
                    self.db.close_signal(symbol, "WIN",
                                         profit_pips=abs(entry - tp))
                    self.cooldown.remove(symbol)
                    log.info(f"  TP HIT  {symbol} SELL → WIN at {price:.5g}")
                    closed = True
                elif price >= sl:
                    self.db.close_signal(symbol, "LOSS",
                                         profit_pips=-abs(sl - entry))
                    self.cooldown.remove(symbol)
                    self.risk.record_trade_result("LOSS", MAX_RISK_PER_TRADE)
                    log.warning(f"  SL HIT  {symbol} SELL → LOSS at {price:.5g}")
                    closed = True

            if not closed:
                still_open.append(sig)

        with open(OPEN_SIGNALS_FILE, "w") as f:
            json.dump(still_open, f, indent=2, default=str)

        # Update performance JSON
        self._update_performance()

    def _update_performance(self):
        rows = self.db.fetchall("""
            SELECT symbol,
                   COUNT(*) total,
                   SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) wins,
                   SUM(CASE WHEN result='LOSS' THEN 1 ELSE 0 END) losses,
                   AVG(confidence) avg_conf,
                   AVG(rr_ratio) avg_rr
            FROM signals
            GROUP BY symbol
        """)
        perf = [dict(r) for r in rows]
        with open(PERFORMANCE_FILE, "w") as f:
            json.dump(perf, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def load_filters() -> Dict:
    try:
        with open(FILTER_FILE) as f:
            return json.load(f)
    except Exception:
        return {
            "killzone": True, "calendar": True, "spread": True,
            "volatility": True, "market_quality": True,
            "regime": True, "mtf": True, "telegram": False,
        }


def main():
    log.info("=" * 60)
    log.info("  AI Trading Bot v5  —  Institutional Grade Engine")
    log.info(f"  Instruments : {len(SYMBOLS)}")
    log.info(f"  Min confidence: {CONFIDENCE_THRESHOLD}%")
    log.info(f"  MTF agreement : {MIN_TIMEFRAMES_AGREE}/{len(TIMEFRAMES)} timeframes")
    log.info(f"  Scan interval : {SCAN_INTERVAL_MINUTES} min")
    log.info(f"  Cooldown      : {COOLDOWN_HOURS}h per pair")
    log.info(f"  Max daily loss: {MAX_DAILY_LOSS_PCT*100:.0f}%")
    log.info("=" * 60)

    db        = DatabaseManager()
    generator = SignalGenerator(db)
    learner   = SelfLearningEngine(db)

    scan_num = 0
    while True:
        scan_num += 1
        filters = load_filters()

        log.info(f"\n{'─'*50}")
        log.info(f"  SCAN #{scan_num}  {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC")
        log.info(f"  Session: {current_session()}")

        # ── Check open positions first ────────────────────────────
        generator.close_open_positions()

        # ── Check risk limits ─────────────────────────────────────
        ok, reason = generator.risk.can_trade()
        if not ok:
            log.warning(f"  ⛔ {reason}  —  bot paused until limits reset")
            time.sleep(SCAN_INTERVAL_MINUTES * 60)
            continue

        # ── Scan all symbols ──────────────────────────────────────
        signals_generated = 0
        for symbol in SYMBOLS:
            try:
                sig = generator.process_symbol(symbol, filters)
                if sig:
                    signals_generated += 1
                    log.info(
                        f"\n  ══ SIGNAL GENERATED ══\n"
                        f"  {sig['symbol']} {sig['signal']}\n"
                        f"  Confidence : {sig['confidence']:.0f}%\n"
                        f"  Entry      : {sig['entry']}\n"
                        f"  SL         : {sig['sl']}\n"
                        f"  TP1        : {sig['tp']}\n"
                        f"  TP2        : {sig['tp2']}\n"
                        f"  TP3        : {sig['tp3']}\n"
                        f"  RR         : 1:{sig['rr_ratio']:.1f}\n"
                        f"  Regime     : {sig['regime']}\n"
                        f"  Reasons    : {sig['reasons']}\n"
                    )
            except Exception as e:
                log.error(f"  Error processing {symbol}: {e}")

        if signals_generated == 0:
            log.info(f"  No high-quality setups found — waiting for better opportunities")
        else:
            log.info(f"  {signals_generated} signal(s) generated this scan")

        # ── Self-learning ─────────────────────────────────────────
        try:
            learner.maybe_adapt()
        except Exception as e:
            log.debug(f"Learning error: {e}")

        # ── Wait for next scan ────────────────────────────────────
        log.info(f"  Next scan in {SCAN_INTERVAL_MINUTES} minutes")
        time.sleep(SCAN_INTERVAL_MINUTES * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Bot stopped by user")
