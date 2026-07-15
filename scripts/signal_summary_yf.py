# IBKR-style Signal Summary + Backtest - yfinance Multi-Ticker
# Standalone script (Colab or local). Run tag is generated at execution time in HKT.
#
# Usage:
#   pip install yfinance pandas numpy tabulate
#   python scripts/signal_summary_yf.py                 # signals + backtest
#   python scripts/signal_summary_yf.py --no-backtest   # signals only
#   python scripts/signal_summary_yf.py --period 5y     # longer backtest window
#
# Notes on accuracy:
# - auto_adjust=True so SMA/RSI are computed on split/dividend-adjusted closes.
# - RSI uses Wilder's smoothing (SMMA), matching IBKR / TradingView / Investing.com.
# - Cross-source close validation: pass --verify to compare the last close
#   against Stooq and flag divergence > 0.5%.

import argparse
import warnings
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------- watchlist
CORE = ['INTC', 'MU', 'SNDK', 'MRVL', 'AMD', 'NVDA', 'QCOM', 'LITE', 'AAPL',
        'GOOG', '000660.KS', '7709.HK']
SPEC = ['SATS', '2209.HK']            # 2209.HK = YesAsia Holdings
LEV = ['SOXL', 'SOXS', 'SQQQ']
CRYPTO = ['BTC-USD', 'ETH-USD']
SECTOR_PROXY = ['SOXX']               # primary sector proxy; SOXL is fallback

ALL_TICKERS = CORE + SPEC + LEV + CRYPTO + SECTOR_PROXY

# Semis subset the sector momentum filter applies to
SEMIS = {'INTC', 'MU', 'SNDK', 'MRVL', 'AMD', 'NVDA', 'QCOM', 'LITE',
         '000660.KS', '7709.HK', 'SOXL', 'SOXS'}

SMA_PTS = {'Bullish': 2, 'Neutral': 0, 'Bearish': -2}
RSI_PTS = {'Strong Buy': 2, 'Buy': 1, 'Sell': -1, 'Strong Sell': -2, 'N/A': 0}
MOM_PTS = {'High': 1, 'Neutral': 0, 'Low': -1, 'N/A': 0}
TRADE_COST = 0.0005                   # 5 bps per position change (backtest)


def bucket_of(t):
    if t in CORE: return 'Core'
    if t in SPEC: return 'Speculative'
    if t in LEV: return 'Leveraged'
    if t in CRYPTO: return 'Crypto'
    return 'Sector'


# ---------------------------------------------------------------- indicators
def rsi_wilder(close, window=14):
    """Wilder's RSI (SMMA smoothing) - matches IBKR/TradingView values."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def sma_score_series(close):
    """Daily SMA points: +2 above SMA20/50/200, -2 below all, else 0."""
    s20 = close.rolling(20).mean()
    s50 = close.rolling(50).mean()
    s200 = close.rolling(200).mean()
    above = (close > s20).astype(int) + (close > s50).astype(int) + (close > s200).astype(int)
    score = pd.Series(0, index=close.index)
    score[above == 3] = 2
    score[above == 0] = -2
    return score.where(~s200.isna(), np.nan)


def rsi_pts_series(rsi):
    pts = pd.Series(np.nan, index=rsi.index)
    pts[rsi < 30] = 2
    pts[(rsi >= 30) & (rsi < 50)] = 1
    pts[(rsi >= 50) & (rsi <= 70)] = -1
    pts[rsi > 70] = -2
    return pts


def mom_pts_series(close):
    mom3m = close.pct_change(63) * 100
    pts = pd.Series(0.0, index=close.index)
    pts[mom3m > 20] = 1
    pts[mom3m <= -10] = -1
    return pts.where(~mom3m.isna(), np.nan)


def score_to_signal(score):
    if score >= 4: return 'Strong Buy'
    if score >= 2: return 'Buy'
    if score >= -1: return 'Neutral'
    if score >= -3: return 'Sell'
    return 'Strong Sell'


def daily_score(close, is_core, is_semi, sector_bullish):
    """Historical daily aggregate score, same rules as the live table."""
    score = sma_score_series(close) + rsi_pts_series(rsi_wilder(close))
    if is_core:
        score = score + mom_pts_series(close)
    if is_semi and sector_bullish is not None:
        sect = sector_bullish.reindex(close.index).ffill()
        score = score + sect.map({True: 1, False: -1})
    return score


# ---------------------------------------------------------------- backtest
def backtest(close, score, cost=TRADE_COST):
    """Long when score >= 2 (Buy/Strong Buy), flat otherwise.
    Signal on close(t) -> position held over t+1 (next close-to-close return).
    Returns summary metrics vs buy & hold, plus 20d forward-return stats."""
    df = pd.DataFrame({'close': close, 'score': score}).dropna()
    if len(df) < 60:
        return {}
    ret = df['close'].pct_change()
    pos = (df['score'] >= 2).astype(float)
    strat = pos.shift(1) * ret - pos.diff().abs().fillna(0) * cost
    strat = strat.fillna(0)

    equity = (1 + strat).cumprod()
    bh_equity = (1 + ret.fillna(0)).cumprod()
    n_years = len(df) / 252
    cagr = equity.iloc[-1] ** (1 / n_years) - 1 if n_years > 0 else np.nan
    bh_cagr = bh_equity.iloc[-1] ** (1 / n_years) - 1 if n_years > 0 else np.nan
    sharpe = (strat.mean() / strat.std() * np.sqrt(252)) if strat.std() > 0 else np.nan
    max_dd = (equity / equity.cummax() - 1).min()
    exposure = pos.shift(1).mean()

    # round-trip win rate
    entries = df.index[(pos == 1) & (pos.shift(1) != 1)]
    exits = df.index[(pos == 0) & (pos.shift(1) == 1)]
    wins = trades = 0
    for e in entries:
        x = exits[exits > e]
        end = x[0] if len(x) else df.index[-1]
        r = df.loc[end, 'close'] / df.loc[e, 'close'] - 1
        trades += 1
        wins += r > 0
    # 20d forward return after a Buy/Strong Buy day vs any day
    fwd20 = df['close'].pct_change(20).shift(-20)
    fwd_buy = fwd20[df['score'] >= 2].mean()
    fwd_all = fwd20.mean()

    return {
        'BT_CAGR_%': round(cagr * 100, 1),
        'BH_CAGR_%': round(bh_cagr * 100, 1),
        'BT_Sharpe': round(sharpe, 2),
        'BT_MaxDD_%': round(max_dd * 100, 1),
        'Exposure_%': round(exposure * 100, 0),
        'Trades': trades,
        'WinRate_%': round(wins / trades * 100, 0) if trades else np.nan,
        'Fwd20d_onBuy_%': round(fwd_buy * 100, 1) if not np.isnan(fwd_buy) else np.nan,
        'Fwd20d_all_%': round(fwd_all * 100, 1) if not np.isnan(fwd_all) else np.nan,
    }


# ---------------------------------------------------------------- email
# 与主应用同一套环境变量：EMAIL_SENDER / EMAIL_PASSWORD / EMAIL_RECEIVERS，
# SMTP 按发件人域名自动识别；不在下表的域名可用 EMAIL_SMTP_SERVER /
# EMAIL_SMTP_PORT / EMAIL_SMTP_SSL 显式指定。
SMTP_CONFIGS = {
    'qq.com': {'server': 'smtp.qq.com', 'port': 465, 'ssl': True},
    'foxmail.com': {'server': 'smtp.qq.com', 'port': 465, 'ssl': True},
    '163.com': {'server': 'smtp.163.com', 'port': 465, 'ssl': True},
    '126.com': {'server': 'smtp.126.com', 'port': 465, 'ssl': True},
    'gmail.com': {'server': 'smtp.gmail.com', 'port': 587, 'ssl': False},
    'outlook.com': {'server': 'smtp-mail.outlook.com', 'port': 587, 'ssl': False},
    'hotmail.com': {'server': 'smtp-mail.outlook.com', 'port': 587, 'ssl': False},
    'live.com': {'server': 'smtp-mail.outlook.com', 'port': 587, 'ssl': False},
    'aliyun.com': {'server': 'smtp.aliyun.com', 'port': 465, 'ssl': True},
    '139.com': {'server': 'smtp.139.com', 'port': 465, 'ssl': True},
}


def send_email_report(subject, body_text, attachments=()):
    """Send the report via SMTP (stdlib only). Returns True on success.
    Missing configuration is a graceful skip, never an error, so the
    analysis run stays green when email is not set up."""
    import html
    import os
    import smtplib
    from email.mime.application import MIMEApplication
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    sender = (os.getenv('EMAIL_SENDER') or '').strip()
    password = (os.getenv('EMAIL_PASSWORD') or '').strip()
    if not sender or not password:
        print('Email not configured (EMAIL_SENDER/EMAIL_PASSWORD empty); skipping email.')
        return False
    receivers = [r.strip() for r in (os.getenv('EMAIL_RECEIVERS') or '').split(',')
                 if r.strip()] or [sender]

    domain = sender.rsplit('@', 1)[-1].lower()
    auto = SMTP_CONFIGS.get(domain, {})
    host = (os.getenv('EMAIL_SMTP_SERVER') or '').strip() or auto.get('server')
    if not host:
        print(f'No SMTP config for sender domain "{domain}"; '
              'set EMAIL_SMTP_SERVER / EMAIL_SMTP_PORT. Skipping email.')
        return False
    port = int((os.getenv('EMAIL_SMTP_PORT') or '').strip() or auto.get('port') or 465)
    ssl_raw = (os.getenv('EMAIL_SMTP_SSL') or '').strip().lower()
    use_ssl = ssl_raw in ('1', 'true', 'yes') if ssl_raw else auto.get('ssl', port == 465)

    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = ', '.join(receivers)
    body = (f"<html><body><pre style='font-family:Menlo,Consolas,monospace;"
            f"font-size:12px'>{html.escape(body_text)}</pre></body></html>")
    msg.attach(MIMEText(body, 'html', 'utf-8'))
    for path in attachments:
        try:
            with open(path, 'rb') as f:
                part = MIMEApplication(f.read(), Name=os.path.basename(path))
            part['Content-Disposition'] = (
                f'attachment; filename="{os.path.basename(path)}"')
            msg.attach(part)
        except OSError:
            pass

    try:
        if use_ssl:
            server = smtplib.SMTP_SSL(host, port, timeout=30)
        else:
            server = smtplib.SMTP(host, port, timeout=30)
        with server:
            if not use_ssl:
                server.starttls()
            server.login(sender, password)
            server.sendmail(sender, receivers, msg.as_string())
        print(f'Email sent to {len(receivers)} receiver(s) via {host}:{port}.')
        return True
    except Exception as e:
        print(f'Email send failed: {e}')
        return False


# ---------------------------------------------------------------- pipeline
def build_summary(data, do_backtest=True):
    """data: yf.download(group_by='ticker') MultiIndex frame."""
    closes = {}
    for t in ALL_TICKERS:
        if isinstance(data.columns, pd.MultiIndex) and t in data.columns.get_level_values(0):
            s = data[t]['Close'].dropna()
            if len(s):
                closes[t] = s

    # sector filter series (SOXX primary, SOXL fallback)
    proxy_name = 'SOXX' if 'SOXX' in closes else ('SOXL' if 'SOXL' in closes else None)
    sector_bullish = None
    if proxy_name:
        pc = closes[proxy_name]
        sector_bullish = ((pc > pc.rolling(20).mean())
                          & (pc > pc.rolling(50).mean())
                          & (pc > pc.rolling(200).mean()))

    rows, bt_rows = [], []
    failed = [t for t in ALL_TICKERS if t not in closes]
    for t, close in closes.items():
        n = len(close)
        last = close.iloc[-1]
        s20 = close.rolling(20).mean().iloc[-1] if n >= 20 else np.nan
        s50 = close.rolling(50).mean().iloc[-1] if n >= 50 else np.nan
        s200 = close.rolling(200).mean().iloc[-1] if n >= 200 else np.nan
        rsi = rsi_wilder(close).iloc[-1] if n >= 15 else np.nan

        smas = [s for s in (s20, s50, s200) if not pd.isna(s)]
        above = [last > s for s in smas]
        sma_status = ('Bullish' if smas and all(above)
                      else 'Bearish' if smas and not any(above) else 'Neutral')

        rsi_sig = 'N/A'
        if not np.isnan(rsi):
            rsi_sig = ('Strong Buy' if rsi < 30 else 'Buy' if rsi < 50
                       else 'Sell' if rsi <= 70 else 'Strong Sell')

        mom3m = (last / close.iloc[-63] - 1) * 100 if n >= 63 else np.nan
        mom6m = (last / close.iloc[-126] - 1) * 100 if n >= 126 else np.nan
        if t in CORE and not np.isnan(mom3m):
            mom_class = 'High' if mom3m > 20 else 'Neutral' if mom3m > -10 else 'Low'
        else:
            mom_class = 'N/A'

        is_semi = t in SEMIS
        sector_now = 'N/A'
        if is_semi and sector_bullish is not None:
            sector_now = 'Bullish' if bool(sector_bullish.iloc[-1]) else 'Bearish'

        score = SMA_PTS[sma_status] + RSI_PTS[rsi_sig] + MOM_PTS[mom_class]
        score += {'Bullish': 1, 'Bearish': -1}.get(sector_now, 0)

        rows.append({
            'Symbol': t, 'Bucket': bucket_of(t),
            'LastBar': close.index[-1].strftime('%Y-%m-%d'),
            'Close': round(last, 2),
            'SMA20': round(s20, 2) if not pd.isna(s20) else np.nan,
            'SMA50': round(s50, 2) if not pd.isna(s50) else np.nan,
            'SMA200': round(s200, 2) if not pd.isna(s200) else np.nan,
            'SMA_Status': sma_status,
            'RSI_14': round(rsi, 1) if not np.isnan(rsi) else np.nan,
            'RSI_Signal': rsi_sig,
            'Mom3m_%': round(mom3m, 1) if not np.isnan(mom3m) else np.nan,
            'Mom6m_%': round(mom6m, 1) if not np.isnan(mom6m) else np.nan,
            'Mom_Class': mom_class,
            'Sector_Filter': sector_now,
            'Score': score,
            'Overall': score_to_signal(score),
        })

        if do_backtest:
            hist_score = daily_score(close, t in CORE, is_semi, sector_bullish)
            metrics = backtest(close, hist_score)
            if metrics:
                bt_rows.append({'Symbol': t, **metrics})

    df = pd.DataFrame(rows)
    bt = pd.DataFrame(bt_rows) if bt_rows else pd.DataFrame()
    if not bt.empty:
        df = df.merge(bt[['Symbol', 'BT_CAGR_%', 'BH_CAGR_%', 'BT_Sharpe',
                          'BT_MaxDD_%', 'WinRate_%', 'Fwd20d_onBuy_%']],
                      on='Symbol', how='left')
    return df, bt, proxy_name, failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--period', default='2y', help='history window (2y/5y/max)')
    ap.add_argument('--no-backtest', action='store_true')
    ap.add_argument('--verify', action='store_true',
                    help='cross-check last close vs Stooq (needs pandas-datareader)')
    ap.add_argument('--csv', default='signals_summary.csv')
    ap.add_argument('--email', action='store_true',
                    help='email the report (EMAIL_SENDER/EMAIL_PASSWORD/'
                         'EMAIL_RECEIVERS env, same as main app)')
    args = ap.parse_args()

    report = []

    def emit(line=''):
        print(line)
        report.append(line)

    import yfinance as yf
    run_tag = datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y-%m-%d %H:%M HKT')
    emit(f"Run tag: {run_tag}")
    print(f"Fetching {len(ALL_TICKERS)} tickers ({args.period})...")
    data = yf.download(ALL_TICKERS, period=args.period, group_by='ticker',
                       auto_adjust=True, progress=False, threads=True)

    df, bt, proxy_name, failed = build_summary(data, do_backtest=not args.no_backtest)

    emit("\n=== SIGNAL TABLE ===")
    emit(df.to_markdown(index=False))
    if not bt.empty:
        emit("\n=== BACKTEST (score>=2 long / else flat, next-close exec, "
             f"{TRADE_COST*1e4:.0f}bp cost) ===")
        emit(bt.to_markdown(index=False))
    if proxy_name:
        row = df[df['Symbol'] == proxy_name]
        status = row.iloc[0]['SMA_Status'] if len(row) else 'n/a'
        emit(f"\nSector proxy {proxy_name}: {status}")
    if failed:
        emit(f"Failed/invalid tickers: {failed}")

    if args.verify:
        try:
            import pandas_datareader.data as web
            print("\n=== CROSS-SOURCE CHECK (yfinance vs Stooq) ===")
            for t in [x for x in ALL_TICKERS if '.' not in x and '-' not in x]:
                try:
                    alt = web.DataReader(t, 'stooq').sort_index()['Close'].iloc[-1]
                    yfc = df.loc[df['Symbol'] == t, 'Close'].iloc[0]
                    dev = abs(alt / yfc - 1)
                    flag = ' <-- CHECK' if dev > 0.005 else ''
                    print(f"{t}: yf={yfc:.2f} stooq={alt:.2f} dev={dev:.2%}{flag}")
                except Exception:
                    print(f"{t}: stooq unavailable")
        except ImportError:
            print("pandas-datareader not installed; skip --verify")

    df.to_csv(args.csv, index=False)
    print(f"\nSaved {args.csv} | Run tag: {run_tag}")

    if args.email:
        send_email_report(f"Daily Signal Summary — {run_tag}",
                          '\n'.join(report), attachments=[args.csv])


if __name__ == '__main__':
    main()
