import time
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="All-in-One Delta V2", page_icon="📈", layout="wide")
BASE = "https://api.india.delta.exchange"

@st.cache_data(ttl=300)
def get_candles(symbol, resolution, start_ts, end_ts):
    step = {"5m":300,"15m":900,"30m":1800,"1h":3600,"4h":14400}[resolution]
    out, cur = [], int(start_ts)
    while cur < int(end_ts):
        e = min(cur + step*1990, int(end_ts))
        r = requests.get(f"{BASE}/v2/history/candles",
                         params={"resolution":resolution,"symbol":symbol,
                                 "start":cur,"end":e},
                         headers={"Accept":"application/json"}, timeout=20)
        r.raise_for_status()
        data = r.json().get("result", [])
        if not data: break
        out += data
        last = int(data[-1]["time"])
        cur = max(cur + step, last + step)
        time.sleep(.05)
    if not out: return pd.DataFrame()
    d = pd.DataFrame(out).drop_duplicates("time").sort_values("time")
    d["time"] = pd.to_datetime(d["time"], unit="s", utc=True)
    for c in ["open","high","low","close","volume"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.reset_index(drop=True)

def atr(d,n):
    pc=d.close.shift(1)
    tr=pd.concat([d.high-d.low,(d.high-pc).abs(),(d.low-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False).mean()

def macd(s):
    a=s.ewm(span=12,adjust=False).mean()
    b=s.ewm(span=26,adjust=False).mean()
    m=a-b; sig=m.ewm(span=9,adjust=False).mean()
    return m,sig,m-sig

def supertrend(d,n=20,f=2):
    a=atr(d,n); mid=(d.high+d.low)/2
    up=mid+f*a; lo=mid-f*a
    fu,fl=up.copy(),lo.copy()
    trend=pd.Series(1,index=d.index,dtype=int)
    for i in range(1,len(d)):
        fu.iloc[i]=up.iloc[i] if up.iloc[i]<fu.iloc[i-1] or d.close.iloc[i-1]>fu.iloc[i-1] else fu.iloc[i-1]
        fl.iloc[i]=lo.iloc[i] if lo.iloc[i]>fl.iloc[i-1] or d.close.iloc[i-1]<fl.iloc[i-1] else fl.iloc[i-1]
        if trend.iloc[i-1]==-1 and d.close.iloc[i]>fu.iloc[i-1]: trend.iloc[i]=1
        elif trend.iloc[i-1]==1 and d.close.iloc[i]<fl.iloc[i-1]: trend.iloc[i]=-1
        else: trend.iloc[i]=trend.iloc[i-1]
    return pd.Series(np.where(trend==1,fl,fu),index=d.index),trend

def features(d,cfg):
    x=d.copy()
    x["ema20"]=x.close.ewm(span=20,adjust=False).mean()
    x["ema50"]=x.close.ewm(span=50,adjust=False).mean()
    x["atr"]=atr(x,cfg["atr_len"])
    x["macd"],x["macd_sig"],x["macd_hist"]=macd(x.close)
    x["st_line"],x["st_dir"]=supertrend(x,cfg["atr_len"],cfg["st_factor"])
    x["vol_ma"]=x.volume.rolling(20).mean()
    x["vol_ratio"]=x.volume/x.vol_ma.replace(0,np.nan)
    x["swing_hi"]=x.high.rolling(cfg["lookback"]).max().shift(1)
    x["swing_lo"]=x.low.rolling(cfg["lookback"]).min().shift(1)
    x["sweep_low"]=(x.low<x.swing_lo)&(x.close>x.swing_lo)
    x["sweep_high"]=(x.high>x.swing_hi)&(x.close<x.swing_hi)
    x["displacement"]=(x.close-x.open).abs()>x.atr*cfg["disp_atr"]
    return x

def run_backtest(d,symbol,cfg):
    x=features(d,cfg); equity=cfg["capital"]; pos=None; last_i=-99999
    trades=[]; curve=[]
    day=None; day_start=equity
    start=max(60,cfg["lookback"]+2)
    for i in range(start,len(x)):
        r=x.iloc[i]; ts=r.time
        if day!=ts.date(): day=ts.date(); day_start=equity
        if pos:
            side=pos["side"]
            hit_sl=(r.low<=pos["sl"]) if side==1 else (r.high>=pos["sl"])
            hit_tp=(r.high>=pos["tp"]) if side==1 else (r.low<=pos["tp"])
            exit_px=None; reason=None
            if hit_sl: exit_px,reason=pos["sl"],"SL"
            elif hit_tp: exit_px,reason=pos["tp"],"TP"
            move=(r.close-pos["entry"])*side
            if exit_px is None and move>=pos["risk_unit"]*cfg["be_r"]:
                pos["sl"]=max(pos["sl"],pos["entry"]) if side==1 else min(pos["sl"],pos["entry"])
            if exit_px is None and move>=pos["risk_unit"]*cfg["trail_start_r"]:
                tr=r.close-side*r.atr*cfg["trail_atr"]
                pos["sl"]=max(pos["sl"],tr) if side==1 else min(pos["sl"],tr)
            if exit_px is not None:
                gross=(exit_px-pos["entry"])*pos["qty"]*side
                fee=(abs(pos["entry"]*pos["qty"])+abs(exit_px*pos["qty"]))*cfg["fee"]
                slip=abs(exit_px*pos["qty"])*cfg["slip"]
                net=gross-fee-slip; equity+=net
                trades.append([symbol,pos["time"],ts,"LONG" if side==1 else "SHORT",
                               pos["entry"],exit_px,pos["qty"],net,net/max(pos["risk_cash"],1e-9),reason])
                pos=None
        if (equity-day_start)/max(day_start,1)<=-cfg["daily_loss"]:
            curve.append([ts,equity]); continue
        if pos is None and i-last_i>=cfg["cooldown"]:
            L=S=0
            L += int(r.st_dir==1); S += int(r.st_dir==-1)
            L += int(r.close>r.ema20>r.ema50); S += int(r.close<r.ema20<r.ema50)
            L += int(r.macd_hist>0); S += int(r.macd_hist<0)
            if r.vol_ratio>=cfg["vol_mult"]: L+=1; S+=1
            L += int(r.sweep_low); S += int(r.sweep_high)
            L += int(r.displacement and r.close>r.open)
            S += int(r.displacement and r.close<r.open)
            side=1 if L>=cfg["min_score"] and L>S else (-1 if S>=cfg["min_score"] and S>L else 0)
            if side:
                entry=float(r.close); risk_unit=max(float(r.atr)*cfg["sl_atr"],entry*0.0005)
                sl=entry-side*risk_unit; tp=entry+side*risk_unit*cfg["rr"]
                risk_cash=equity*cfg["risk_pct"]
                qty=max(cfg["min_qty"],risk_cash/risk_unit*cfg["multiplier"])
                pos={"side":side,"entry":entry,"sl":sl,"tp":tp,"qty":qty,
                     "time":ts,"risk_cash":risk_cash,"risk_unit":risk_unit}
                last_i=i
        curve.append([ts,equity])
    if pos:
        r=x.iloc[-1]; exit_px=float(r.close)
        gross=(exit_px-pos["entry"])*pos["qty"]*pos["side"]
        fee=(abs(pos["entry"]*pos["qty"])+abs(exit_px*pos["qty"]))*cfg["fee"]
        slip=abs(exit_px*pos["qty"])*cfg["slip"]; net=gross-fee-slip; equity+=net
        trades.append([symbol,pos["time"],r.time,"LONG" if pos["side"]==1 else "SHORT",
                       pos["entry"],exit_px,pos["qty"],net,net/max(pos["risk_cash"],1e-9),"EOD"])
    cols=["symbol","entry_time","exit_time","side","entry","exit","qty","pnl","R","reason"]
    return pd.DataFrame(trades,columns=cols),pd.DataFrame(curve,columns=["time","equity"])

st.title("📈 All-in-One Delta — V2")
st.caption("Independent multi-asset research/backtester. It does not claim to reproduce proprietary Pushkar Raj Thakur/MirrorPip rules.")

with st.sidebar:
    st.header("V2 controls")
    symbols=st.multiselect("Delta symbols",
        ["BTCUSD","ETHUSD","SOLUSD","XRPUSD","XAUTUSD","PAXGUSD","SLVONUSD","BNBUSD","DOGEUSD","HYPEUSD","AVAXUSD","DOTUSD"],
        ["BTCUSD","ETHUSD","XAUTUSD"])
    resolution=st.selectbox("Timeframe",["5m","15m","30m","1h","4h"],1)
    months=st.slider("Backtest months",1,24,12)
    capital=st.number_input("Initial portfolio ₹",1000.0,10000000.0,10000.0,500.0)
    risk_pct=st.number_input("Risk/trade %",0.1,2.0,0.5,0.1)/100
    daily_loss=st.number_input("Daily loss stop %",0.5,10.0,3.0,0.5)/100
    fee=st.number_input("Fee per side %",0.0,0.5,0.05,0.01)/100
    slip=st.number_input("Slippage %",0.0,0.5,0.02,0.01)/100
    st.subheader("Signal engine")
    atr_len=st.number_input("Supertrend ATR length",5,100,20)
    st_factor=st.number_input("Supertrend factor",0.5,6.0,2.0,0.1)
    min_score=st.slider("Minimum confluence score",2,6,4)
    vol_mult=st.number_input("Volume threshold ×",0.5,5.0,1.0,0.1)
    lookback=st.number_input("Liquidity lookback",5,100,20)
    sl_atr=st.number_input("SL ATR ×",0.5,5.0,1.5,0.1)
    rr=st.number_input("Target R:R",1.0,5.0,2.0,0.25)
    disp_atr=st.number_input("Displacement body/ATR",0.2,3.0,0.8,0.1)
    be_r=st.number_input("Breakeven after R",0.5,3.0,1.0,0.25)
    trail_start_r=st.number_input("Trail starts after R",0.5,5.0,1.5,0.25)
    trail_atr=st.number_input("Trail ATR ×",0.5,5.0,1.0,0.1)
    cooldown=st.number_input("Cooldown bars",0,100,4)
    multiplier=st.number_input("Contract multiplier",0.000001,1000.0,1.0,0.000001)
    min_qty=st.number_input("Minimum qty",0.000001,1000.0,0.001,0.001)

if st.button("🚀 RUN V2 BACKTEST",type="primary",use_container_width=True):
    if not symbols: st.error("Select at least one symbol."); st.stop()
    cfg={"capital":capital,"risk_pct":risk_pct,"daily_loss":daily_loss,"fee":fee,"slip":slip,
         "atr_len":int(atr_len),"st_factor":st_factor,"min_score":min_score,
         "vol_mult":vol_mult,"lookback":int(lookback),"sl_atr":sl_atr,"rr":rr,
         "disp_atr":disp_atr,"be_r":be_r,"trail_start_r":trail_start_r,
         "trail_atr":trail_atr,"cooldown":int(cooldown),"multiplier":multiplier,"min_qty":min_qty}
    end=datetime.now(timezone.utc); start=end-timedelta(days=30*months)
    all_trades=[]; all_curves=[]; warnings=[]
    bar=st.progress(0)
    for n,s in enumerate(symbols,1):
        try:
            d=get_candles(s,resolution,int(start.timestamp()),int(end.timestamp()))
            if len(d)<100: warnings.append(f"{s}: only {len(d)} candles")
            else:
                t,c=run_backtest(d,s,cfg); all_trades.append(t); all_curves.append(c.assign(symbol=s))
        except Exception as e: warnings.append(f"{s}: {e}")
        bar.progress(n/len(symbols))
    if warnings:
        with st.expander("Data warnings"):
            for w in warnings: st.write("• "+w)
    if not all_trades: st.error("No results."); st.stop()
    trades=pd.concat(all_trades,ignore_index=True).sort_values("exit_time")
    eq=pd.DataFrame({"time":trades.exit_time,"equity":capital+trades.pnl.cumsum()}) if len(trades) else pd.DataFrame()
    final=float(eq.equity.iloc[-1]) if len(eq) else capital
    pnl=final-capital; win=(trades.pnl>0).mean()*100 if len(trades) else 0
    gains=trades.loc[trades.pnl>0,"pnl"].sum(); losses=trades.loc[trades.pnl<0,"pnl"].sum()
    pf=gains/abs(losses) if losses else np.inf
    dd=((eq.equity/eq.equity.cummax())-1).min()*100 if len(eq) else 0
    c1,c2,c3,c4,c5,c6=st.columns(6)
    c1.metric("Final",f"₹{final:,.2f}"); c2.metric("Net P&L",f"₹{pnl:,.2f}")
    c3.metric("Win rate",f"{win:.2f}%"); c4.metric("Profit factor",f"{pf:.2f}")
    c5.metric("Max DD",f"{dd:.2f}%"); c6.metric("Trades",len(trades))
    chart_eq = eq.set_index("time")[["equity"]]
    st.subheader("V2 Portfolio Equity Curve")
    st.line_chart(chart_eq, use_container_width=True, height=420)
    st.subheader("Per-symbol results")
    stats=trades.groupby("symbol").agg(Trades=("pnl","size"),PnL=("pnl","sum"),
        WinRate=("pnl",lambda s:(s>0).mean()*100),AvgR=("R","mean")).reset_index()
    st.dataframe(stats,use_container_width=True)
    st.subheader("Recent trades")
    st.dataframe(trades.tail(50),use_container_width=True)
    st.download_button("⬇️ Download trades CSV",trades.to_csv(index=False).encode(),
                       "all_in_one_delta_v2_trades.csv","text/csv")
st.divider()
st.caption("V2 uses Delta Exchange India's public historical candle endpoint. No API key and no live-order functions are included.")
