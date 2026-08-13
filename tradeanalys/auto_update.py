#!/usr/bin/env python3
"""
XAU/USD Pattern Detector - Auto Update Script
===============================================
Jalankan script ini di server Anda untuk auto-regenerate HTML setiap 1 menit.

CARA PAKAI:
1. Simpan file ini di server (misal: /home/user/auto_update.py)
2. Install dependency: pip install pandas numpy matplotlib scipy
3. Jalankan manual: python3 auto_update.py
4. Atau setup cron job setiap 1 menit:
   crontab -e
   */1 * * * * /usr/bin/python3 /home/user/auto_update.py >> /home/user/pattern.log 2>&1

CATATAN:
- Script ini mengambil data 1-menit dari Yahoo Finance (max 7 hari)
- Output HTML di-save ke path yang Anda tentukan di bawah
- Pastikan folder output memiliki permission write
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from io import BytesIO
import base64
import warnings
warnings.filterwarnings('ignore')

# ============================================
# KONFIGURASI
# ============================================
OUTPUT_HTML_PATH = "index.html"  # Ganti sesuai path web server Anda
TICKER = "GC=F"  # XAU/USD (Gold Futures)
DATA_SOURCE = "yahoo_csv"  # Bisa "yahoo_csv" (dari file) atau generate dummy
CSV_DATA_PATH = "xauusd_1m.csv"  # Path ke data CSV 1-menit (jika pakai yahoo_csv)

# ============================================
# PATTERN DETECTOR CLASS
# ============================================

class PatternDetector:
    def __init__(self, data, ticker="XAU/USD"):
        self.ticker = ticker
        self.data = data.copy()
        self.patterns_found = []

    def detect_hammer(self, idx):
        row = self.data.iloc[idx]
        body = abs(row['Close'] - row['Open'])
        lower_shadow = min(row['Open'], row['Close']) - row['Low']
        upper_shadow = row['High'] - max(row['Open'], row['Close'])
        if body == 0: return False
        is_hammer = (lower_shadow >= 2.5 * body) and (upper_shadow <= 0.1 * body)
        if idx >= 3 and is_hammer:
            trend = self.data['Close'].iloc[idx-3:idx].pct_change().dropna().mean()
            return trend < 0
        return False

    def detect_shooting_star(self, idx):
        row = self.data.iloc[idx]
        body = abs(row['Close'] - row['Open'])
        upper_shadow = row['High'] - max(row['Open'], row['Close'])
        lower_shadow = min(row['Open'], row['Close']) - row['Low']
        if body == 0: return False
        is_shooting = (upper_shadow >= 2.5 * body) and (lower_shadow <= 0.1 * body)
        if idx >= 3 and is_shooting:
            trend = self.data['Close'].iloc[idx-3:idx].pct_change().dropna().mean()
            return trend > 0
        return False

    def detect_morning_star(self, idx):
        if idx < 2: return False
        c1, c2, c3 = self.data.iloc[idx-2], self.data.iloc[idx-1], self.data.iloc[idx]
        bearish = c1['Close'] < c1['Open']
        small = abs(c2['Close']-c2['Open'])/(c2['High']-c2['Low']) < 0.3 if (c2['High']-c2['Low'])>0 else False
        bullish = c3['Close'] > c3['Open']
        return bearish and small and bullish and c3['Close'] > (c1['Open']+c1['Close'])/2

    def detect_evening_star(self, idx):
        if idx < 2: return False
        c1, c2, c3 = self.data.iloc[idx-2], self.data.iloc[idx-1], self.data.iloc[idx]
        bullish = c1['Close'] > c1['Open']
        small = abs(c2['Close']-c2['Open'])/(c2['High']-c2['Low']) < 0.3 if (c2['High']-c2['Low'])>0 else False
        bearish = c3['Close'] < c3['Open']
        return bullish and small and bearish and c3['Close'] < (c1['Open']+c1['Close'])/2

    def find_pivots(self, window=5):
        highs = self.data['High'].values
        lows = self.data['Low'].values
        max_idx = argrelextrema(highs, np.greater, order=window)[0]
        min_idx = argrelextrema(lows, np.less, order=window)[0]
        return max_idx, min_idx

    def detect_double_top(self, tolerance=0.03):
        max_idx, min_idx = self.find_pivots(window=3)
        if len(max_idx) < 2: return None
        for i in range(len(max_idx)-1):
            p1, p2 = max_idx[i], max_idx[i+1]
            valleys = [v for v in min_idx if p1 < v < p2]
            if not valleys: continue
            price1 = float(self.data['High'].iloc[p1])
            price2 = float(self.data['High'].iloc[p2])
            if abs(price1-price2)/price1 < tolerance:
                v = valleys[len(valleys)//2]
                v_price = float(self.data['Low'].iloc[v])
                return {
                    'pattern': 'Double Top', 'type': 'Bearish Reversal',
                    'peak1': (self.data.index[p1], price1),
                    'peak2': (self.data.index[p2], price2),
                    'valley': (self.data.index[v], v_price),
                    'neckline': v_price,
                    'target': v_price - (price1 - v_price)
                }
        return None

    def detect_double_bottom(self, tolerance=0.03):
        max_idx, min_idx = self.find_pivots(window=3)
        if len(min_idx) < 2: return None
        for i in range(len(min_idx)-1):
            v1, v2 = min_idx[i], min_idx[i+1]
            peaks = [p for p in max_idx if v1 < p < v2]
            if not peaks: continue
            price1 = float(self.data['Low'].iloc[v1])
            price2 = float(self.data['Low'].iloc[v2])
            if abs(price1-price2)/price1 < tolerance:
                p = peaks[len(peaks)//2]
                p_price = float(self.data['High'].iloc[p])
                return {
                    'pattern': 'Double Bottom', 'type': 'Bullish Reversal',
                    'valley1': (self.data.index[v1], price1),
                    'valley2': (self.data.index[v2], price2),
                    'peak': (self.data.index[p], p_price),
                    'neckline': p_price,
                    'target': p_price + (p_price - price1)
                }
        return None

    def detect_head_and_shoulders(self, tolerance=0.04):
        max_idx, min_idx = self.find_pivots(window=3)
        if len(max_idx) < 3: return None
        for i in range(len(max_idx)-2):
            l, h, r = max_idx[i], max_idx[i+1], max_idx[i+2]
            lp = float(self.data['High'].iloc[l])
            hp = float(self.data['High'].iloc[h])
            rp = float(self.data['High'].iloc[r])
            if hp > lp and hp > rp and abs(lp-rp)/lp < tolerance:
                vals = [v for v in min_idx if l < v < r]
                if len(vals) >= 2:
                    neck = float(np.mean([self.data['Low'].iloc[v] for v in vals[:2]]))
                    return {
                        'pattern': 'Head & Shoulders', 'type': 'Bearish Reversal',
                        'left_shoulder': (self.data.index[l], lp),
                        'head': (self.data.index[h], hp),
                        'right_shoulder': (self.data.index[r], rp),
                        'neckline': neck,
                        'target': neck - (hp - neck)
                    }
        return None

    def detect_inverse_head_and_shoulders(self, tolerance=0.04):
        max_idx, min_idx = self.find_pivots(window=3)
        if len(min_idx) < 3: return None
        for i in range(len(min_idx)-2):
            l, h, r = min_idx[i], min_idx[i+1], min_idx[i+2]
            lp = float(self.data['Low'].iloc[l])
            hp = float(self.data['Low'].iloc[h])
            rp = float(self.data['Low'].iloc[r])
            if hp < lp and hp < rp and abs(lp-rp)/lp < tolerance:
                peaks = [p for p in max_idx if l < p < r]
                if len(peaks) >= 2:
                    neck = float(np.mean([self.data['High'].iloc[p] for p in peaks[:2]]))
                    return {
                        'pattern': 'Inverse Head & Shoulders', 'type': 'Bullish Reversal',
                        'left_shoulder': (self.data.index[l], lp),
                        'head': (self.data.index[h], hp),
                        'right_shoulder': (self.data.index[r], rp),
                        'neckline': neck,
                        'target': neck + (neck - hp)
                    }
        return None

    def scan_all_patterns(self):
        self.patterns_found = []
        for i in range(len(self.data)):
            if self.detect_hammer(i):
                self.patterns_found.append({'pattern':'Hammer','type':'Bullish Reversal',
                    'date':self.data.index[i],'price':float(self.data['Low'].iloc[i]),'index':i})
            if self.detect_shooting_star(i):
                self.patterns_found.append({'pattern':'Shooting Star','type':'Bearish Reversal',
                    'date':self.data.index[i],'price':float(self.data['High'].iloc[i]),'index':i})
            if self.detect_morning_star(i):
                self.patterns_found.append({'pattern':'Morning Star','type':'Bullish Reversal',
                    'date':self.data.index[i],'price':float(self.data['Close'].iloc[i]),'index':i})
            if self.detect_evening_star(i):
                self.patterns_found.append({'pattern':'Evening Star','type':'Bearish Reversal',
                    'date':self.data.index[i],'price':float(self.data['Close'].iloc[i]),'index':i})
        dt = self.detect_double_top()
        if dt: self.patterns_found.append(dt)
        db = self.detect_double_bottom()
        if db: self.patterns_found.append(db)
        hs = self.detect_head_and_shoulders()
        if hs: self.patterns_found.append(hs)
        ihs = self.detect_inverse_head_and_shoulders()
        if ihs: self.patterns_found.append(ihs)
        return self.patterns_found


# ============================================
# LOAD DATA
# ============================================

def load_data():
    """Load 1-minute data from CSV or create dummy data"""
    if DATA_SOURCE == "yahoo_csv" and os.path.exists(CSV_DATA_PATH):
        df = pd.read_csv(CSV_DATA_PATH)
        df.columns = [c.strip() for c in df.columns]
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
        return df
    else:
        # Generate dummy data for testing
        print("[!] Data file not found. Generating dummy data...")
        dates = pd.date_range(end=pd.Timestamp.now(), periods=500, freq='1min')
        base = 4450 + np.cumsum(np.random.normal(0, 0.5, 500))
        data = []
        for i, close in enumerate(base):
            high = close + abs(np.random.normal(1.5, 0.5))
            low = close - abs(np.random.normal(1.5, 0.5))
            open_p = close + np.random.normal(0, 0.3)
            high = max(high, open_p, close)
            low = min(low, open_p, close)
            data.append({
                'Date': dates[i], 'Open': round(open_p,2), 'High': round(high,2),
                'Low': round(low,2), 'Close': round(close,2), 'Volume': int(np.random.normal(100,30))
            })
        df = pd.DataFrame(data).set_index('Date')
        return df


# ============================================
# GENERATE CHART
# ============================================

def generate_chart(df, patterns):
    """Generate chart image as base64"""
    # Filter to last 500 candles + significant patterns
    df_chart = df.tail(500).copy()

    # Re-detect on subset to get correct indices
    det = PatternDetector(df_chart, ticker="XAU/USD 1M")
    det.patterns_found = []

    # Chart patterns
    dt = det.detect_double_top()
    if dt: det.patterns_found.append(dt)
    db = det.detect_double_bottom()
    if db: det.patterns_found.append(db)
    hs = det.detect_head_and_shoulders()
    if hs: det.patterns_found.append(hs)
    ihs = det.detect_inverse_head_and_shoulders()
    if ihs: det.patterns_found.append(ihs)

    # Recent candlestick patterns (last 30 candles)
    for i in range(max(0, len(df_chart)-30), len(df_chart)):
        if det.detect_hammer(i):
            det.patterns_found.append({'pattern':'Hammer','type':'Bullish Reversal',
                'date':df_chart.index[i],'price':float(df_chart['Low'].iloc[i]),'index':i})
        if det.detect_shooting_star(i):
            det.patterns_found.append({'pattern':'Shooting Star','type':'Bearish Reversal',
                'date':df_chart.index[i],'price':float(df_chart['High'].iloc[i]),'index':i})
        if det.detect_morning_star(i):
            det.patterns_found.append({'pattern':'Morning Star','type':'Bullish Reversal',
                'date':df_chart.index[i],'price':float(df_chart['Close'].iloc[i]),'index':i})
        if det.detect_evening_star(i):
            det.patterns_found.append({'pattern':'Evening Star','type':'Bearish Reversal',
                'date':df_chart.index[i],'price':float(df_chart['Close'].iloc[i]),'index':i})

    filtered = det.patterns_found

    fig, axes = plt.subplots(2, 1, figsize=(18, 14), gridspec_kw={'height_ratios': [3, 1]})
    ax1, ax2 = axes[0], axes[1]
    dates = df_chart.index

    for i, (idx, row) in enumerate(df_chart.iterrows()):
        color = '#26a69a' if row['Close'] >= row['Open'] else '#ef5350'
        height = abs(row['Close'] - row['Open'])
        bottom = min(row['Open'], row['Close'])
        rect = plt.Rectangle((i - 0.4, bottom), 0.8, height if height > 0 else 0.01,
                             facecolor=color, edgecolor=color, linewidth=0.8)
        ax1.add_patch(rect)
        ax1.plot([i, i], [row['Low'], row['High']], color=color, linewidth=0.6)

    ax1.set_xlim(-1, len(df_chart))
    ax1.set_ylabel('Harga (USD)', fontsize=13, fontweight='bold')
    ax1.set_title('XAU/USD (Gold) - 1-Minute Pattern Detection\nLast 500 Candles | Real-Time Analysis',
                  fontsize=16, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_facecolor('#fafafa')

    colors_vol = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(df_chart['Close'], df_chart['Open'])]
    ax2.bar(range(len(df_chart)), df_chart['Volume'], color=colors_vol, alpha=0.7, width=0.8)
    ax2.set_ylabel('Volume', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Waktu (1-Minute)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_facecolor('#fafafa')

    step = max(1, len(df_chart) // 12)
    ticks = range(0, len(df_chart), step)
    labels = [dates[i].strftime('%m/%d %H:%M') for i in ticks]
    ax1.set_xticks(ticks); ax1.set_xticklabels([])
    ax2.set_xticks(ticks); ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)

    for pat in filtered:
        if pat['pattern'] in ['Hammer', 'Shooting Star', 'Morning Star', 'Evening Star']:
            idx = pat['index']; price = pat['price']
            color = '#00c853' if 'Bullish' in pat['type'] else '#ff1744'
            ax1.annotate(pat['pattern'], xy=(idx, price),
                        xytext=(idx, price * 1.0008 if 'Bullish' in pat['type'] else price * 0.9992),
                        fontsize=7, color=color, fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.2),
                        ha='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85, edgecolor=color))
        elif pat['pattern'] == 'Double Top':
            p1, p2 = pat['peak1'], pat['peak2']
            idx1 = df_chart.index.get_loc(p1[0]); idx2 = df_chart.index.get_loc(p2[0])
            ax1.plot([idx1, idx2], [p1[1], p2[1]], 'r--', linewidth=2.5, alpha=0.8, label='Double Top')
            ax1.scatter([idx1, idx2], [p1[1], p2[1]], color='red', s=100, zorder=5, edgecolors='darkred', linewidths=2)
            ax1.axhline(y=pat['neckline'], color='red', linestyle=':', alpha=0.7, linewidth=1.5)
        elif pat['pattern'] == 'Double Bottom':
            v1, v2 = pat['valley1'], pat['valley2']
            idx1 = df_chart.index.get_loc(v1[0]); idx2 = df_chart.index.get_loc(v2[0])
            ax1.plot([idx1, idx2], [v1[1], v2[1]], 'g--', linewidth=2.5, alpha=0.8, label='Double Bottom')
            ax1.scatter([idx1, idx2], [v1[1], v2[1]], color='green', s=100, zorder=5, edgecolors='darkgreen', linewidths=2)
            ax1.axhline(y=pat['neckline'], color='green', linestyle=':', alpha=0.7, linewidth=1.5)
        elif pat['pattern'] == 'Head & Shoulders':
            ls, h, rs = pat['left_shoulder'], pat['head'], pat['right_shoulder']
            idx_ls = df_chart.index.get_loc(ls[0]); idx_h = df_chart.index.get_loc(h[0]); idx_rs = df_chart.index.get_loc(rs[0])
            ax1.plot([idx_ls, idx_h, idx_rs], [ls[1], h[1], rs[1]], 'r-', linewidth=2.5, alpha=0.85)
            ax1.scatter([idx_ls, idx_h, idx_rs], [ls[1], h[1], rs[1]], color='red', s=130, zorder=5, edgecolors='darkred', linewidths=2)
            ax1.axhline(y=pat['neckline'], color='red', linestyle=':', alpha=0.7, linewidth=1.5)
        elif pat['pattern'] == 'Inverse Head & Shoulders':
            ls, h, rs = pat['left_shoulder'], pat['head'], pat['right_shoulder']
            idx_ls = df_chart.index.get_loc(ls[0]); idx_h = df_chart.index.get_loc(h[0]); idx_rs = df_chart.index.get_loc(rs[0])
            ax1.plot([idx_ls, idx_h, idx_rs], [ls[1], h[1], rs[1]], 'g-', linewidth=2.5, alpha=0.85)
            ax1.scatter([idx_ls, idx_h, idx_rs], [ls[1], h[1], rs[1]], color='green', s=130, zorder=5, edgecolors='darkgreen', linewidths=2)
            ax1.axhline(y=pat['neckline'], color='green', linestyle=':', alpha=0.7, linewidth=1.5)

    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=9, framealpha=0.9, edgecolor='gray')
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    chart_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return chart_b64, filtered


# ============================================
# GENERATE HTML
# ============================================

def generate_html(df, chart_b64, filtered_patterns):
    current_price = float(df['Close'].iloc[-1])
    price_change = float(df['Close'].iloc[-1] - df['Close'].iloc[-2])
    pct_change = (price_change / float(df['Close'].iloc[-2])) * 100
    high_5d = float(df['High'].max())
    low_5d = float(df['Low'].min())
    last_update = df.index[-1].strftime('%Y-%m-%d %H:%M UTC')

    pattern_rows = ""
    for i, pat in enumerate(filtered_patterns, 1):
        icon = "📈" if "Bullish" in pat['type'] else "📉"
        badge_class = "bullish" if "Bullish" in pat['type'] else "bearish"
        if 'date' in pat:
            date_str = pat['date'].strftime('%m/%d %H:%M')
            price_str = f"{pat['price']:.2f}"
            extra = f"<span class='price-tag'>${price_str}</span>"
        else:
            date_str = "Multi-candle"
            extra = f"<span class='target-tag'>Neckline: {pat['neckline']:.2f} | Target: {pat['target']:.2f}</span>"
        pattern_rows += f"""
        <tr>
            <td class="num">{i}</td>
            <td class="pattern-name">{icon} {pat['pattern']}</td>
            <td><span class="badge {badge_class}">{pat['type']}</span></td>
            <td class="date">{date_str}</td>
            <td>{extra}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="refresh" content="60">
    <title>XAU/USD 1-Minute Pattern Detector - Live</title>
    <style>
        :root {{ --bg: #0a0e1a; --card: #111827; --text: #f8fafc; --muted: #64748b;
                --bullish: #10b981; --bearish: #ef4444; --accent: #f59e0b; --border: #1e293b; }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); line-height: 1.6; }}
        .container {{ max-width: 1440px; margin: 0 auto; padding: 20px; }}
        header {{ text-align: center; padding: 25px 20px; border-bottom: 1px solid var(--border); margin-bottom: 25px; }}
        header h1 {{ font-size: 2rem; margin-bottom: 6px; background: linear-gradient(90deg, #fbbf24, #f59e0b); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
        header p {{ color: var(--muted); font-size: 0.95rem; }}
        .live-badge {{ display: inline-flex; align-items: center; gap: 6px; background: rgba(16,185,129,0.12); color: var(--bullish); padding: 5px 14px; border-radius: 20px; font-size: 0.8rem; font-weight: 700; border: 1px solid rgba(16,185,129,0.25); margin-bottom: 10px; text-transform: uppercase; letter-spacing: 1px; }}
        .live-dot {{ width: 8px; height: 8px; background: var(--bullish); border-radius: 50%; animation: pulse 1.5s infinite; box-shadow: 0 0 8px var(--bullish); }}
        @keyframes pulse {{ 0%,100%{{opacity:1;transform:scale(1)}} 50%{{opacity:0.4;transform:scale(0.8)}} }}
        .refresh-bar {{ display: flex; align-items: center; justify-content: center; gap: 15px; padding: 10px; background: var(--card); border: 1px solid var(--border); border-radius: 10px; margin-bottom: 25px; font-size: 0.85rem; color: var(--muted); }}
        .refresh-bar .timer {{ font-family: 'Courier New', monospace; font-weight: 700; color: var(--accent); font-size: 1.1rem; background: rgba(245,158,11,0.1); padding: 4px 12px; border-radius: 6px; border: 1px solid rgba(245,158,11,0.2); }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 25px; }}
        .stat-card {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 18px; text-align: center; transition: all 0.25s ease; position: relative; overflow: hidden; }}
        .stat-card:hover {{ transform: translateY(-3px); border-color: #334155; }}
        .stat-label {{ color: var(--muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1.2px; }}
        .stat-value {{ font-size: 1.5rem; font-weight: 800; margin-top: 4px; }}
        .stat-change {{ font-size: 0.85rem; margin-top: 3px; font-weight: 600; }}
        .up {{ color: var(--bullish); }} .down {{ color: var(--bearish); }}
        .section {{ margin-bottom: 25px; }}
        .section-title {{ font-size: 1.15rem; font-weight: 700; margin-bottom: 12px; display: flex; align-items: center; gap: 10px; color: var(--text); }}
        .section-title::before {{ content: ''; width: 4px; height: 22px; background: linear-gradient(180deg, var(--accent), #fbbf24); border-radius: 2px; }}
        .tv-widget {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; height: 520px; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }}
        .chart-container {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 12px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }}
        .chart-container img {{ width: 100%; height: auto; border-radius: 8px; display: block; }}
        .pattern-table {{ width: 100%; border-collapse: separate; border-spacing: 0; background: var(--card); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }}
        .pattern-table th {{ background: linear-gradient(180deg, rgba(245,158,11,0.15), rgba(245,158,11,0.05)); color: var(--accent); padding: 14px 16px; text-align: left; font-weight: 700; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.8px; border-bottom: 1px solid var(--border); }}
        .pattern-table td {{ padding: 12px 16px; border-bottom: 1px solid var(--border); }}
        .pattern-table tr:hover {{ background: rgba(255,255,255,0.04); }}
        .pattern-table tr:last-child td {{ border-bottom: none; }}
        .num {{ color: var(--muted); font-weight: 700; width: 40px; font-size: 0.9rem; }}
        .pattern-name {{ font-weight: 700; font-size: 0.92rem; }}
        .date {{ color: var(--muted); font-size: 0.85rem; font-family: 'Courier New', monospace; }}
        .badge {{ display: inline-block; padding: 4px 10px; border-radius: 6px; font-size: 0.7rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.5px; }}
        .badge.bullish {{ background: rgba(16,185,129,0.12); color: var(--bullish); border: 1px solid rgba(16,185,129,0.25); }}
        .badge.bearish {{ background: rgba(239,68,68,0.12); color: var(--bearish); border: 1px solid rgba(239,68,68,0.25); }}
        .price-tag {{ color: var(--accent); font-weight: 700; font-size: 0.9rem; font-family: 'Courier New', monospace; }}
        .target-tag {{ color: var(--muted); font-size: 0.82rem; font-family: 'Courier New', monospace; }}
        .guide-box {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }}
        .guide-box p {{ margin-bottom: 10px; color: var(--muted); font-size: 0.9rem; line-height: 1.7; }}
        .guide-box strong {{ color: var(--accent); }}
        .footer {{ text-align: center; padding: 25px; color: var(--muted); border-top: 1px solid var(--border); margin-top: 15px; font-size: 0.8rem; }}
        @media (max-width: 768px) {{ header h1 {{ font-size: 1.4rem; }} .tv-widget {{ height: 380px; }} .stats-grid {{ grid-template-columns: repeat(2, 1fr); }} .refresh-bar {{ flex-direction: column; gap: 8px; }} }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="live-badge"><span class="live-dot"></span> Live 1-Minute</div>
            <h1>🥇 XAU/USD Pattern Detector</h1>
            <p>Real-Time 1-Minute Pattern Analysis for Gold | Auto-Refresh Every 60s</p>
        </header>
        <div class="refresh-bar">
            <span>⏱️ Next refresh in:</span>
            <span class="timer" id="countdown">60</span>
            <span class="last-update">🕐 Last update: {last_update} UTC</span>
        </div>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Harga Live</div>
                <div class="stat-value">${current_price:,.2f}</div>
                <div class="stat-change {'up' if price_change >= 0 else 'down'}">{'+' if price_change >= 0 else ''}{price_change:,.2f} ({'+' if pct_change >= 0 else ''}{pct_change:.3f}%)</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">High 5D</div>
                <div class="stat-value">${high_5d:,.2f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Low 5D</div>
                <div class="stat-value">${low_5d:,.2f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Pattern Aktif</div>
                <div class="stat-value">{len(filtered_patterns)}</div>
            </div>
        </div>
        <div class="section">
            <div class="section-title">📊 TradingView Chart - 1 Minute</div>
            <div class="tv-widget">
                <iframe id="tv-chart" src="https://www.tradingview.com/widgetembed/?frameElementId=tradingview_chart&symbol=OANDA%3AXAUUSD&interval=1&hidesidetoolbar=0&symboledit=1&saveimage=1&toolbarbg=f1f3f6&studies=%5B%5D&theme=dark&style=1&timezone=Asia%2FJakarta&studies_overrides=%7B%7D&overrides=%7B%7D&enabled_features=%5B%5D&disabled_features=%5B%5D&locale=id&utm_source=localhost&utm_medium=widget&utm_campaign=chart&utm_term=OANDA%3AXAUUSD" style="width:100%;height:100%;border:none;" allowtransparency="true" scrolling="no" allowfullscreen="true"></iframe>
            </div>
        </div>
        <div class="section">
            <div class="section-title">🔍 Pattern Detection Analysis (1-Minute)</div>
            <div class="chart-container">
                <img src="data:image/png;base64,{chart_b64}" alt="XAU/USD 1-Minute Pattern Analysis" loading="lazy">
            </div>
        </div>
        <div class="section">
            <div class="section-title">📋 Hasil Deteksi Pola (Real-Time)</div>
            <table class="pattern-table">
                <thead><tr><th>#</th><th>Pola</th><th>Sinyal</th><th>Waktu</th><th>Detail</th></tr></thead>
                <tbody>{pattern_rows}</tbody>
            </table>
        </div>
        <div class="section">
            <div class="section-title">⚡ Panduan Trading</div>
            <div class="guide-box">
                <p><strong>Double Top:</strong> Dua puncak di level sama + neckline break = entry SELL. Target = jarak peak ke neckline, diukur dari titik break.</p>
                <p><strong>Double Bottom:</strong> Dua lembah di level sama + neckline break = entry BUY. Target = jarak valley ke neckline, diukur dari titik break.</p>
                <p><strong>Head & Shoulders:</strong> Bahu kiri + kepala (lebih tinggi) + bahu kanan. Break neckline = SELL. Konfirmasi volume sangat penting.</p>
                <p><strong>Candlestick:</strong> Hammer (rejection bawah, bullish), Shooting Star (rejection atas, bearish), Morning/Evening Star (3-candle reversal pattern).</p>
            </div>
        </div>
        <div class="footer">
            <p>⚠️ Disclaimer: Analisis ini hanya untuk edukasi. Trading melibatkan risiko finansial. Selalu gunakan stop loss dan risk management.</p>
            <p>Generated by Pattern Detector v1.0 | Data: Yahoo Finance 1-Minute | Auto-Refresh: 60s | Last Update: {last_update} UTC</p>
        </div>
    </div>
    <script>
        let seconds = 60;
        const countdownEl = document.getElementById('countdown');
        function updateTimer() {{ seconds--; if (seconds <= 0) {{ seconds = 60; const iframe = document.getElementById('tv-chart'); iframe.src = iframe.src; }} countdownEl.textContent = seconds; }}
        setInterval(updateTimer, 1000);
    </script>
</body>
</html>"""

    with open(OUTPUT_HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[✓] HTML saved: {OUTPUT_HTML_PATH}")


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("XAU/USD Pattern Detector - Auto Update")
    print("=" * 60)
    print(f"[i] Output: {OUTPUT_HTML_PATH}")
    print(f"[i] Time: {pd.Timestamp.now()}")

    try:
        df = load_data()
        print(f"[✓] Data loaded: {len(df)} candles")

        chart_b64, filtered = generate_chart(df, None)
        print(f"[✓] Chart generated: {len(chart_b64)} chars")
        print(f"[✓] Patterns found: {len(filtered)}")

        generate_html(df, chart_b64, filtered)
        print("[✓] Done!")
    except Exception as e:
        print(f"[✗] Error: {e}")
        import traceback
        traceback.print_exc()
