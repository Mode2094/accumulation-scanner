import ccxt
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime

# ========= INDICATORS =========

def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def compute_indicators(df):
    df["ema7"] = ema(df["close"], 7)
    df["ema14"] = ema(df["close"], 14)
    df["ema28"] = ema(df["close"], 28)

    # MACD
    ema12 = ema(df["close"], 12)
    ema26 = ema(df["close"], 26)
    df["macd"] = ema12 - ema26
    df["signal"] = ema(df["macd"], 9)
    df["hist"] = df["macd"] - df["signal"]

    # RSI - محسّن
    delta = df["close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    roll_up = pd.Series(gain).rolling(14).mean()
    roll_down = pd.Series(loss).rolling(14).mean()
    
    # تجنب القسمة على صفر
    rs = roll_up / (roll_down + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    # اتجاه RSI
    df["rsi_trend"] = df["rsi"].diff(3) > 0

    return df


# ========= FILTERS =========

def check_pump_filter(df, max_pump_percent=20):
    """فلتر لمنع العملات التي شهدت مضاربة قوية (Pump)"""
    if len(df) < 24:
        return True, "بيانات غير كافية للتحقق"
    
    # حساب نسبة التغير في آخر 24 ساعة
    price_24h_ago = df["close"].iloc[-24]
    current_price = df["close"].iloc[-1]
    price_change_percent = ((current_price - price_24h_ago) / price_24h_ago) * 100
    
    if price_change_percent > max_pump_percent:
        return False, f"ارتفاع كبير ({price_change_percent:.1f}%) - تجنب المضاربة"
    
    return True, "سعر مستقر"

def check_volatility_filter(df, max_volatility=0.05):
    """فلتر الاستقرار السعري"""
    if len(df) < 10:
        return True, "بيانات غير كافية"
    
    # حساب التقلب (الانحراف المعياري للتغيرات النسبية)
    price_changes = df["close"].pct_change().dropna()
    if len(price_changes) < 10:
        return True, "بيانات غير كافية"
    
    volatility = price_changes.rolling(10).std().iloc[-1]
    
    if pd.isna(volatility):
        return True, "تقلب طبيعي"
    
    if volatility > max_volatility:
        return False, f"تقلب عالي ({volatility:.3f}) - مخاطرة مرتفعة"
    
    return True, "تقلب مقبول"


# ========= SCORING (ACCUMULATION) - محسّن =========

def score_coin(df, ticker_data=None):
    if len(df) < 50:
        return 0, ["بيانات قليلة"], True

    last = df.iloc[-1]
    prev = df.iloc[-2]

    score = 0
    reasons = []
    passed_filters = True

    # تطبيق الفلترات الأمنية أولاً
    pump_ok, pump_msg = check_pump_filter(df)
    volatility_ok, volatility_msg = check_volatility_filter(df)
    
    if not pump_ok:
        reasons.append(f"❌ {pump_msg}")
        passed_filters = False
    else:
        reasons.append(f"✅ {pump_msg}")
    
    if not volatility_ok:
        reasons.append(f"❌ {volatility_msg}")
        passed_filters = False
    else:
        reasons.append(f"✅ {volatility_msg}")

    # إذا لم يجتاز الفلترات، نوقف التقييم
    if not passed_filters:
        return score, reasons, passed_filters

    # 1. مرشحات الاتجاه المحسنة
    if last["close"] > last["ema7"]:
        score += 1
        reasons.append("السعر أعلى من EMA7")

    if (last["ema7"] - df["ema7"].iloc[-4]) > 0:
        score += 1
        reasons.append("اتجاه EMA7 صاعد")

    if last["close"] > last["ema28"]:
        score += 1
        reasons.append("السعر فوق المتوسط العام EMA28")

    # 2. مرشح MACD محسّن
    if last["hist"] > 0 and prev["hist"] <= 0:
        score += 2
        reasons.append("تقاطع MACD Histogram إيجابي")
    elif last["hist"] > 0:
        score += 1
        reasons.append("MACD Histogram موجب")

    # 3. مرشح RSI محسّن
    if 40 <= last["rsi"] <= 70:
        score += 1
        reasons.append("RSI في منطقة تجميع صحية")
    
    if last["rsi_trend"] and last["rsi"] > 50:
        score += 1
        reasons.append("RSI صاعد وفوق 50")

    # 4. مرشح حجم التداول المحسّن
    avg_vol = df["volume"].iloc[-21:-1].mean()
    if avg_vol > 0 and last["volume"] > 1.5 * avg_vol:
        score += 2
        reasons.append("ارتفاع واضح في حجم التداول")
    elif avg_vol > 0 and last["volume"] > avg_vol:
        score += 1
        reasons.append("حجم تداول أعلى من المتوسط")

    # 5. مرشح السيولة (جديد)
    if ticker_data:
        quote_volume = ticker_data.get('quoteVolume', 0)
        if quote_volume > 1000000:  # مليون USDT
            score += 2
            reasons.append("سيولة عالية")
        elif quote_volume > 500000:
            score += 1
            reasons.append("سيولة جيدة")

    return score, reasons, passed_filters


# ========= TREND ANALYSIS =========

def market_trend(df):
    """تحليل الاتجاه العام للعملة"""
    if len(df) < 30:
        return "غير محدد"
    
    ema28_trend = df["ema28"].iloc[-1] > df["ema28"].iloc[-10]
    price_above_ema = df["close"].iloc[-1] > df["ema28"].iloc[-1]
    
    if ema28_trend and price_above_ema:
        return "صاعد قوي"
    elif price_above_ema:
        return "صاعد"
    elif not price_above_ema and not ema28_trend:
        return "هابط"
    else:
        return "متذبذب"


# ========= TRADE LEVELS (ENTRY / SL / TP / ETA) - محسّن =========

def compute_trade_levels(df, timeframe_hours=1):
    """
    يحسب:
    - entry_price: آخر إغلاق (حسابياً)
    - sl: أقرب قاع أخير أو % ثابت
    - tp1 / tp2: اعتماداً على نسبة المخاطرة (R:R = 1:2, 1:3)
    - resistance: أقرب مقاومة تقريبية (أعلى هاي آخر 30 شمعة)
    - eta_text: مدة تقريبية للوصول للمقاومة حسب سرعة الحركة
    """
    if len(df) < 30:
        return {
            "entry": float(df.iloc[-1]["close"]),
            "sl": None,
            "tp1": None,
            "tp2": None,
            "resistance": None,
            "eta_text": "غير متوفر (بيانات قليلة)",
            "risk_reward": None,
            "sl_type": None
        }

    last = df.iloc[-1]
    entry = float(last["close"])

    # تحسين حساب وقف الخسارة - نبحث عن أقرب دعم منطقي
    swing_low = float(df["low"].iloc[-10:].min())
    raw_risk = entry - swing_low

    # لو القاع تحت السعر بشكل منطقي (أقل من 8% مخاطرة)
    if raw_risk > 0 and raw_risk / entry <= 0.08:
        sl = swing_low
        sl_type = "دعم قريب"
    else:
        # مخاطرة افتراضية 3% مع تحسين
        sl = entry * (1 - 0.03)
        sl_type = "نسبة 3%"

    risk = max(entry - sl, 0.0)

    if risk == 0:
        tp1 = None
        tp2 = None
        risk_reward = None
    else:
        tp1 = entry + 2 * risk
        tp2 = entry + 3 * risk
        risk_reward = f"1:{2} و 1:{3}"

    # مقاومة تقريبية: أعلى high في آخر 30 شمعة مع تحسين
    recent_highs = df["high"].iloc[-30:]
    resistance = float(recent_highs.max())

    # حساب سرعة الحركة (volatility pace) محسّن
    recent_changes = df["close"].pct_change().iloc[-20:].dropna()
    if len(recent_changes) > 0:
        avg_move = float(recent_changes.abs().mean())
    else:
        avg_move = 0.01  # قيمة افتراضية

    if resistance <= entry or avg_move == 0:
        eta_text = "غير متوفر (لا توجد مقاومة قريبة أو الحركة ضعيفة)"
    else:
        distance_frac = (resistance - entry) / entry
        candles_needed = distance_frac / avg_move if avg_move > 0 else 50
        hours_needed = candles_needed * timeframe_hours

        if hours_needed < 1:
            eta_text = "أقل من ساعة"
        elif hours_needed < 6:
            eta_text = f"حوالي {round(hours_needed)} ساعة"
        elif hours_needed < 24:
            eta_text = f"حوالي {round(hours_needed)} ساعة"
        else:
            days = hours_needed / 24
            eta_text = f"حوالي {round(days, 1)} يوم"

    return {
        "entry": entry,
        "sl": sl,
        "sl_type": sl_type,
        "tp1": tp1,
        "tp2": tp2,
        "resistance": resistance,
        "eta_text": eta_text,
        "risk_reward": risk_reward
    }


# ========= FETCHING FROM BYBIT - محسّن =========

def fetch_data(exchange, symbol):
    raw = exchange.fetch_ohlcv(symbol, "1h", limit=200)
    df = pd.DataFrame(raw, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    return df


def init_exchange():
    print("🚀 تشغيل فلتر التجميع العربي المحسّن مع الفلترات الأمنية...\n")
    print("🛡️  الفلترات الأمنية المضافة:")
    print("   • فلتر المضاربة (Pump): استبعاد العملات التي ارتفعت >20% في 24h")
    print("   • فلتر التقلب: استبعاد العملات عالية التقلب")
    print("   • فلتر السيولة: حد أدنى 100K USDT")
    print("=" * 70 + "\n")
    
    print("🔄 جاري تحميل بيانات Bybit Spot ...")
    ex = ccxt.bybit({
        "enableRateLimit": True,
        "options": {"defaultType": "spot"}
    })
    ex.load_markets()
    print("🟢 تم تحميل الأسواق!\n")
    return ex


def get_usdt_pairs(exchange):
    markets = exchange.markets
    usdt = [s for s,m in markets.items() if m.get("type")=="spot" and s.endswith("/USDT")]

    tickers = exchange.fetch_tickers(usdt)
    
    # ترشيح حسب السيولة - محسّن
    liquid_pairs = []
    for s, t in tickers.items():
        quote_vol = float(t.get("quoteVolume") or 0)
        if quote_vol > 100000:  # حد أدنى للسيولة 100K USDT
            liquid_pairs.append((s, quote_vol))
    
    ranked = sorted(liquid_pairs, key=lambda x: x[1], reverse=True)
    
    return [s for s, _ in ranked[:80]]


# ========= JSON OUTPUT =========

def save_to_json(results, filename=None):
    """حفظ النتائج في ملف JSON للتكامل مع Flutter"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trading_signals_{timestamp}.json"
    
    # تحضير البيانات للتطبيق
    output_data = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_coins_analyzed": len(results),
            "successful_coins": len([r for r in results if r.get("passed_filters", True)]),
            "version": "2.0"
        },
        "signals": []
    }
    
    for result in results:
        if result.get("passed_filters", True):
            signal = {
                "symbol": result["symbol"],
                "score": result["score"],
                "max_score": result["max_score"],
                "current_price": result["price"],
                "rsi": result["rsi"],
                "trend": result["trend"],
                "liquidity": result["quote_volume"],
                "entry_price": result["entry"],
                "stop_loss": result["sl"],
                "take_profit_1": result["tp1"],
                "take_profit_2": result["tp2"],
                "resistance": result["resistance"],
                "eta": result["eta_text"],
                "risk_reward": result["risk_reward"],
                "reasons": result["reasons"],
                "timestamp": datetime.now().isoformat()
            }
            output_data["signals"].append(signal)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    return filename


# ========= MAIN - محسّن =========

def main():
    ex = init_exchange()
    symbols = get_usdt_pairs(ex)

    print(f"📌 سيتم فحص {len(symbols)} عملة بعد الترشيح الأولي للسيولة...\n")

    results = []
    tickers_cache = ex.fetch_tickers(symbols)
    
    filtered_count = 0

    for sym in symbols:
        print(f"➡️ فحص: {sym}")
        try:
            df = fetch_data(ex, sym)
            df = compute_indicators(df)
            score, reasons, passed_filters = score_coin(df, tickers_cache[sym])
            levels = compute_trade_levels(df, timeframe_hours=1)
            trend = market_trend(df)

            last = df.iloc[-1]

            if not passed_filters:
                filtered_count += 1

            results.append({
                "symbol": sym,
                "price": float(last["close"]),
                "rsi": float(last["rsi"]),
                "volume": float(last["volume"]),
                "quote_volume": float(tickers_cache[sym].get('quoteVolume', 0)),
                "score": score,
                "max_score": 11,
                "trend": trend,
                "reasons": reasons,
                "passed_filters": passed_filters,
                "entry": levels["entry"],
                "sl": levels["sl"],
                "sl_type": levels["sl_type"],
                "tp1": levels["tp1"],
                "tp2": levels["tp2"],
                "resistance": levels["resistance"],
                "eta_text": levels["eta_text"],
                "risk_reward": levels["risk_reward"]
            })

            time.sleep(0.1)

        except Exception as e:
            print(f"❌ خطأ مع {sym}: {e}")
            time.sleep(0.15)
            continue

    # فلترة النتائج التي اجتازت الفلترات فقط
    filtered_results = [r for r in results if r["passed_filters"]]
    
    # Top 7 من التي اجتازت الفلترات
    top = sorted(filtered_results, key=lambda x: x["score"], reverse=True)[:7]

    print("\n" + "="*70)
    print("           ⭐ أفضل 7 عملات آمنة للتجميع ⭐      ")
    print("="*70 + "\n")

    for i, r in enumerate(top, 1):
        print(f"{i}) {r['symbol']}")
        print(f"   💰 السعر الحالي: ${r['price']:.6f}")
        print(f"   🎯 درجة التجميع: {r['score']}/{r['max_score']}")
        print(f"   📊 RSI: {r['rsi']:.1f}")
        print(f"   📈 الاتجاه العام: {r['trend']}")
        print(f"   💎 السيولة: {r['quote_volume']:,.0f} USDT")
        
        # معلومات التداول المحسنة
        print(f"   🎯 نقاط التداول:")
        print(f"      • الدخول: {r['entry']:.6f}")
        if r['sl']:
            print(f"      • وقف الخسارة: {r['sl']:.6f} ({r['sl_type']})")
        if r['tp1']:
            print(f"      • الهدف 1: {r['tp1']:.6f}")
        if r['tp2']:
            print(f"      • الهدف 2: {r['tp2']:.6f}")
        if r['risk_reward']:
            print(f"      • نسبة المخاطرة: {r['risk_reward']}")
        
        if r["resistance"]:
            print(f"      • المقاومة القادمة: {r['resistance']:.6f}")
            print(f"      • المدة التقريبية: {r['eta_text']}")
        
        print(f"   🔍 أسباب الاختيار:")
        for reason in r["reasons"]:
            print(f"      • {reason}")
        print("-" * 60 + "\n")

    # حفظ النتائج في JSON
    json_filename = save_to_json(results)
    print(f"💾 تم حفظ النتائج في: {json_filename}")

    # إحصائيات نهائية
    avg_score = np.mean([r["score"] for r in filtered_results if r["score"] > 0])
    high_liquidity_count = len([r for r in filtered_results if r["quote_volume"] > 1000000])
    
    print(f"📈 إحصائيات عامة:")
    print(f"   • متوسط نقاط العملات: {avg_score:.1f}")
    print(f"   • عدد العملات المفحوصة: {len(results)}")
    print(f"   • العملات التي اجتازت الفلترات: {len(filtered_results)}")
    print(f"   • العملات المستبعدة: {filtered_count}")
    print(f"   • العملات عالية السيولة (>1M): {high_liquidity_count}")
    print(f"   • وقت التشغيل: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":

    main()



# ========= FLASK SERVER (لإظهار رابط signals) =========

from flask import Flask, send_file
app = Flask(__name__)

@app.route("/")
def home():
    return "Accumulation Scanner Running"

@app.route("/signals")
def signals():
    return send_file("latest.json", mimetype="application/json")
