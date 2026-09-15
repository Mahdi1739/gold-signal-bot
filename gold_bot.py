# ============================================
# ربات تحلیل طلا + هشدار تلگرام
# نسخه: 3.0 - با فیلتر سیگنال
# ============================================

import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, List
from enum import Enum


# ========== تنظیمات ==========
TWELVEDATA_KEY = os.environ.get("TWELVEDATA_KEY")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

SYMBOL = "XAU/USD"
ACCOUNT_BALANCE = 10000
RISK_PERCENT = 1.0

# ========== تنظیمات فیلتر سیگنال ==========
MIN_SCORE = 75  # حداقل امتیاز برای ارسال پیام


# ========== ENUMS ==========
class Direction(Enum):
    BUY = "BUY"
    SELL = "SELL"
    NONE = "NONE"

class Decision(Enum):
    STRONG_ENTRY = "🟢 ورود قوی (Full Position)"
    CAUTIOUS_ENTRY = "🟡 ورود محتاطانه (نصف حجم)"
    WAIT = "🟠 صبر کن"
    NO_ENTRY = "🔴 وارد نشو"
    VETOED = "⛔ وتو شد"


@dataclass
class Condition:
    name: str
    passed: bool
    detail: str = ""
    weight: float = 1.0


@dataclass
class LayerResult:
    name: str
    layer_weight: float
    conditions: List[Condition] = field(default_factory=list)
    sl_price: float = 0.0
    tp_price: float = 0.0
    rr_ratio: float = 0.0
    position_size: float = 0.0

    @property
    def score(self):
        if not self.conditions:
            return 0.0
        tw = sum(c.weight for c in self.conditions)
        e = sum(c.weight for c in self.conditions if c.passed)
        return (e / tw) * 100 if tw > 0 else 0.0

    @property
    def passed_count(self):
        return sum(1 for c in self.conditions if c.passed)

    @property
    def total_count(self):
        return len(self.conditions)


# ========== اندیکاتورها ==========
class Indicators:
    @staticmethod
    def ema(s, p):
        return s.ewm(span=p, adjust=False).mean()

    @staticmethod
    def rsi(s, p=14):
        d = s.diff()
        g = d.where(d > 0, 0).rolling(p).mean()
        l = -d.where(d < 0, 0).rolling(p).mean()
        rs = g / l.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(s, f=12, sl=26, sig=9):
        ef = s.ewm(span=f, adjust=False).mean()
        es = s.ewm(span=sl, adjust=False).mean()
        m = ef - es
        sg = m.ewm(span=sig, adjust=False).mean()
        return m, sg, m - sg

    @staticmethod
    def atr(df, p=14):
        hl = df['high'] - df['low']
        hc = (df['high'] - df['close'].shift()).abs()
        lc = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr.rolling(p).mean()

    @staticmethod
    def swings(df, lb=5):
        h = df['high'].rolling(lb*2+1, center=True).max() == df['high']
        l = df['low'].rolling(lb*2+1, center=True).min() == df['low']
        return h, l


# ========== دریافت داده ==========
def fetch_data(interval, size=500):
    url = "https://api.twelvedata.com/time_series"
    params = {
        'symbol': SYMBOL,
        'interval': interval,
        'outputsize': size,
        'apikey': TWELVEDATA_KEY,
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        data = r.json()
        if 'values' not in data:
            print(f"⚠️ خطا: {data.get('message', data)}")
            return None
        df = pd.DataFrame(data['values'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime').sort_index()
        for c in ['open', 'high', 'low', 'close']:
            df[c] = pd.to_numeric(df[c])
        df['volume'] = (df['high'] - df['low']) * 1000
        return df
    except Exception as e:
        print(f"⚠️ خطا: {e}")
        return None


# ========== ارسال تلگرام ==========
def send_telegram(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': text,
        'parse_mode': 'HTML',
    }
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code == 200:
            print("📤 پیام تلگرام ارسال شد ✅")
            return True
        else:
            print(f"⚠️ خطای تلگرام: {r.text}")
            return False
    except Exception as e:
        print(f"⚠️ خطای اتصال تلگرام: {e}")
        return False


# ========== ربات ==========
class GoldBot:
    LAYER_WEIGHTS = {'HTF': 0.35, 'MTF': 0.30, 'LTF': 0.20, 'RISK': 0.15}

    def __init__(self):
        self.bias = Direction.NONE

    def check_session(self):
        h = datetime.now(timezone.utc).hour
        if 12 <= h < 16: return True, "🔥 همپوشانی لندن+نیویورک"
        if 8 <= h < 12: return True, "✅ سشن لندن"
        if 16 <= h < 21: return True, "✅ سشن نیویورک"
        if 0 <= h < 8: return False, "❌ سشن آسیا"
        return False, "❌ بعد از نیویورک"

    def analyze_htf(self, df, direction):
        r = LayerResult("HTF (H4)", self.LAYER_WEIGHTS['HTF'])
        if len(df) < 200:
            r.conditions.append(Condition("داده کافی", False, f"{len(df)} کندل", 2))
            return r
        c = df['close']
        p = c.iloc[-1]
        e200 = Indicators.ema(c, 200).iloc[-1]
        e50 = Indicators.ema(c, 50).iloc[-1]

        cond1 = (p > e200) if direction == Direction.BUY else (p < e200)
        r.conditions.append(Condition("روند HTF", cond1, f"قیمت={p:.2f}, EMA200={e200:.2f}", 2))

        cond2 = (p > e50 > e200) if direction == Direction.BUY else (p < e50 < e200)
        r.conditions.append(Condition("ترتیب میانگین‌ها", cond2, f"EMA50={e50:.2f}", 1.5))

        h, l = Indicators.swings(df)
        rh = df.loc[h, 'high'].tail(3).values
        rl = df.loc[l, 'low'].tail(3).values
        if len(rh) >= 2 and len(rl) >= 2:
            if direction == Direction.BUY:
                cond3 = rh[-1] > rh[-2] and rl[-1] > rl[-2]
            else:
                cond3 = rh[-1] < rh[-2] and rl[-1] < rl[-2]
        else:
            cond3 = False
        r.conditions.append(Condition("ساختار بازار", cond3, "HH/HL" if direction == Direction.BUY else "LH/LL", 1.5))

        cs, ds = self.check_session()
        r.conditions.append(Condition("سشن معاملاتی", cs, ds, 2.5))

        r.conditions.append(Condition("نبود خبر مهم", True, "چک نشده", 3.0))

        cb = (self.bias == direction) or (self.bias == Direction.NONE)
        r.conditions.append(Condition("بایاس بنیادی", cb, f"{self.bias.value}", 1.0))
        return r

    def analyze_mtf(self, df, df_htf, direction):
        r = LayerResult("MTF (H1)", self.LAYER_WEIGHTS['MTF'])
        if len(df) < 50:
            return r
        c = df['close']
        p = c.iloc[-1]
        e20 = Indicators.ema(c, 20).iloc[-1]
        e50 = Indicators.ema(c, 50).iloc[-1]

        hp = df_htf['close'].iloc[-1]
        h200 = Indicators.ema(df_htf['close'], 200).iloc[-1]
        cond1 = (p > e50 and hp > h200) if direction == Direction.BUY else (p < e50 and hp < h200)
        r.conditions.append(Condition("همسویی با HTF", cond1, "", 2))

        m, s, h = Indicators.macd(c)
        cond2 = (m.iloc[-1] > s.iloc[-1] and h.iloc[-1] > 0) if direction == Direction.BUY else (m.iloc[-1] < s.iloc[-1] and h.iloc[-1] < 0)
        r.conditions.append(Condition("MACD", cond2, f"{m.iloc[-1]:.2f}", 1.5))

        rsi = Indicators.rsi(c).iloc[-1]
        cond3 = (40 < rsi < 70) if direction == Direction.BUY else (30 < rsi < 60)
        r.conditions.append(Condition("RSI", cond3, f"{rsi:.1f}", 1))

        cond4, d4 = self.candle_pattern(df, direction)
        r.conditions.append(Condition("الگوی کندلی", cond4, d4, 1.5))

        cond5, d5 = self.check_bos(df, direction)
        r.conditions.append(Condition("BoS", cond5, d5, 1.5))
        return r

    def analyze_ltf(self, df, df_mtf, direction):
        r = LayerResult("LTF (M15)", self.LAYER_WEIGHTS['LTF'])
        if len(df) < 30:
            return r
        c = df['close']
        p = c.iloc[-1]
        e20 = Indicators.ema(c, 20).iloc[-1]
        m50 = Indicators.ema(df_mtf['close'], 50).iloc[-1]

        cond1 = (p > e20 and df_mtf['close'].iloc[-1] > m50) if direction == Direction.BUY else (p < e20 and df_mtf['close'].iloc[-1] < m50)
        r.conditions.append(Condition("همسویی با MTF", cond1, "", 1.5))

        last = df.iloc[-1]
        cond2 = (last['close'] > last['open']) if direction == Direction.BUY else (last['close'] < last['open'])
        r.conditions.append(Condition("کندل تأییدی", cond2, "", 2))

        av = df['volume'].rolling(20).mean().iloc[-1]
        cv = df['volume'].iloc[-1]
        r.conditions.append(Condition("حجم", cv > av * 1.2, "", 1))

        r.conditions.append(Condition("نبود واگرایی", not self.check_div(df, direction), "", 1))
        return r

    def analyze_risk(self, df_htf, df_ltf, direction, entry):
        r = LayerResult("ریسک", self.LAYER_WEIGHTS['RISK'])
        lb = df_ltf.tail(20)
        atr = Indicators.atr(df_ltf).iloc[-1]

        if direction == Direction.BUY:
            sl = min(lb['low'].min(), entry - 1.5 * atr)
        else:
            sl = max(lb['high'].max(), entry + 1.5 * atr)
        sd = abs(entry - sl)

        cond1 = 0.5 * atr < sd < 3 * atr
        r.conditions.append(Condition("SL منطقی", cond1, f"dist={sd:.2f}, ATR={atr:.2f}", 2))

        lbh = df_htf.tail(50)
        if direction == Direction.BUY:
            tp = max(lbh['high'].max(), entry * 1.005)
        else:
            tp = min(lbh['low'].min(), entry * 0.995)
        td = abs(tp - entry)
        rr = td / sd if sd > 0 else 0
        r.conditions.append(Condition("R:R ≥ 1:2", rr >= 2.0, f"1:{rr:.2f}", 2.5))

        risk_amt = ACCOUNT_BALANCE * (RISK_PERCENT / 100)
        size = round(risk_amt / (sd * 100), 2) if sd > 0 else 0
        r.conditions.append(Condition("حجم", size > 0, f"{size} lot", 1))

        r.sl_price = sl
        r.tp_price = tp
        r.rr_ratio = rr
        r.position_size = size
        return r

    def candle_pattern(self, df, direction):
        last = df.iloc[-1]
        body = abs(last['close'] - last['open'])
        rng = last['high'] - last['low']
        if rng == 0:
            return False, "بدون دامنه"
        uw = last['high'] - max(last['close'], last['open'])
        lw = min(last['close'], last['open']) - last['low']
        if direction == Direction.BUY:
            if lw > 2 * body and body < rng * 0.4:
                return True, "Pin Bar صعودی"
            if last['close'] > last['open'] and body > rng * 0.7:
                return True, "کندل صعودی"
        else:
            if uw > 2 * body and body < rng * 0.4:
                return True, "Pin Bar نزولی"
            if last['close'] < last['open'] and body > rng * 0.7:
                return True, "کندل نزولی"
        if len(df) >= 2:
            pv = df.iloc[-2]
            if direction == Direction.BUY:
                if last['close'] > last['open'] and pv['close'] < pv['open'] and last['close'] > pv['open'] and last['open'] < pv['close']:
                    return True, "Engulfing صعودی"
            else:
                if last['close'] < last['open'] and pv['close'] > pv['open'] and last['close'] < pv['open'] and last['open'] > pv['close']:
                    return True, "Engulfing نزولی"
        return False, "الگو نیست"

    def check_bos(self, df, direction):
        if len(df) < 20:
            return False, "داده کم"
        rec = df.tail(20)
        if direction == Direction.BUY:
            if rec['high'].iloc[-1] > rec['high'].iloc[:-1].max():
                return True, "BoS صعودی"
        else:
            if rec['low'].iloc[-1] < rec['low'].iloc[:-1].min():
                return True, "BoS نزولی"
        return False, "بدون شکست"

    def check_div(self, df, direction):
        if len(df) < 30:
            return False
        rsi = Indicators.rsi(df['close'])
        p = df['close'].tail(10)
        r = rsi.tail(10)
        if direction == Direction.BUY:
            return p.iloc[-1] > p.iloc[0] and r.iloc[-1] < r.iloc[0]
        return p.iloc[-1] < p.iloc[0] and r.iloc[-1] > r.iloc[0]

    def vetoes(self, htf, risk):
        v = []
        n = next((c for c in htf.conditions if "خبر" in c.name), None)
        if n and not n.passed:
            v.append("⛔ خبر مهم")
        rr = next((c for c in risk.conditions if "R:R" in c.name), None)
        if rr and not rr.passed:
            v.append("⛔ R:R کم")
        h = next((c for c in htf.conditions if "روند HTF" in c.name), None)
        if h and not h.passed:
            v.append("⛔ روند HTF مخالف")
        return v

    def analyze(self, df_htf, df_mtf, df_ltf, direction):
        entry = df_ltf['close'].iloc[-1]
        htf = self.analyze_htf(df_htf, direction)
        mtf = self.analyze_mtf(df_mtf, df_htf, direction)
        ltf = self.analyze_ltf(df_ltf, df_mtf, direction)
        risk = self.analyze_risk(df_htf, df_ltf, direction, entry)

        total = (
            htf.score * 0.35 + mtf.score * 0.30 +
            ltf.score * 0.20 + risk.score * 0.15
        )

        v = self.vetoes(htf, risk)
        if v:
            dec = Decision.VETOED
        elif total >= 85:
            dec = Decision.STRONG_ENTRY
        elif total >= 70:
            dec = Decision.CAUTIOUS_ENTRY
        elif total >= 55:
            dec = Decision.WAIT
        else:
            dec = Decision.NO_ENTRY

        return {
            'direction': direction,
            'entry': entry,
            'total': total,
            'decision': dec,
            'vetoes': v,
            'htf': htf,
            'mtf': mtf,
            'ltf': ltf,
            'risk': risk,
        }


# ========== اجرا ==========
print("📥 دریافت داده طلا...")
df_htf = fetch_data("4h", 500)
df_mtf = fetch_data("1h", 500)
df_ltf = fetch_data("15min", 500)

if df_htf is None or df_mtf is None or df_ltf is None:
    print("❌ خطا در دریافت داده")
    send_telegram("❌ خطا در دریافت داده از TwelveData")
    exit()

print(f"✅ HTF: {len(df_htf)} کندل")
print(f"✅ MTF: {len(df_mtf)} کندل")
print(f"✅ LTF: {len(df_ltf)} کندل")

bot = GoldBot()

print("\n🔍 تحلیل BUY...")
r_buy = bot.analyze(df_htf, df_mtf, df_ltf, Direction.BUY)

print("🔍 تحلیل SELL...")
r_sell = bot.analyze(df_htf, df_mtf, df_ltf, Direction.SELL)

best = r_buy if r_buy['total'] >= r_sell['total'] else r_sell

print("\n" + "=" * 55)
print(f"🥇 {SYMBOL}")
print(f"💰 قیمت: {best['entry']:.2f}")
print(f"🕐 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
print("=" * 55)
print(f"\n📊 BUY امتیاز: {r_buy['total']:.1f}%")
print(f"📊 SELL امتیاز: {r_sell['total']:.1f}%")
print(f"\n🏆 بهترین جهت: {best['direction'].value}")
print(f"🎯 امتیاز کل: {best['total']:.1f}%")
print(f"📌 تصمیم: {best['decision'].value}")

if best['vetoes']:
    print("\n⛔ وتوها:")
    for v in best['vetoes']:
        print(f"   {v}")

print("=" * 55)

# ========== فیلتر سیگنال ==========
ALLOWED_DECISIONS = [Decision.STRONG_ENTRY, Decision.CAUTIOUS_ENTRY]

should_send = (
    best['total'] >= MIN_SCORE and
    best['decision'] in ALLOWED_DECISIONS
)

if should_send:
    msg_lines = [
        f"<b>🥇 {SYMBOL}</b>",
        f"<b>جهت:</b> {best['direction'].value}",
        f"<b>قیمت:</b> {best['entry']:.2f}",
        f"<b>امتیاز:</b> {best['total']:.1f}%",
        f"<b>تصمیم:</b> {best['decision'].value}",
    ]

    if best['vetoes']:
        msg_lines.append("\n<b>⛔ وتوها:</b>")
        for v in best['vetoes']:
            msg_lines.append(v)

    if best['risk'].sl_price:
        msg_lines.extend([
            f"\n<b>💡 جزئیات:</b>",
            f"Entry: {best['entry']:.2f}",
            f"SL: {best['risk'].sl_price:.2f}",
            f"TP: {best['risk'].tp_price:.2f}",
            f"R:R: 1:{best['risk'].rr_ratio:.2f}",
            f"Size: {best['risk'].position_size} lot",
        ])

    msg_lines.append(f"\n📊 BUY: {r_buy['total']:.1f}% | SELL: {r_sell['total']:.1f}%")
    msg_lines.append(f"🕐 {datetime.now(timezone.utc).strftime('%H:%M UTC')}")

    send_telegram("\n".join(msg_lines))
    print("📤 پیام تلگرام ارسال شد (سیگنال قوی) ✅")
else:
    if best['total'] < MIN_SCORE:
        print(f"🔕 سیگنال ضعیف: امتیاز {best['total']:.1f}% < {MIN_SCORE}% - پیام ارسال نشد")
    else:
        print(f"🔕 تصمیم غیرمجاز: {best['decision'].value} - پیام ارسال نشد")

print("\n✅ تمام شد!")
