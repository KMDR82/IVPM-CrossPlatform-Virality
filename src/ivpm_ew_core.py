# -*- coding: utf-8 -*-
"""IVPM-EW core module.
Extracted verbatim from Cell 2 of notebooks/ivpm_ew_kaggle_v2.ipynb,
which is the canonical, executable source of all results in the paper.
Requires: numpy, pandas.
"""
import numpy as np
import pandas as pd

T = 72  # months per synthetic series

# === Cell 2: core module — generator (v2), detectors, classifiers ===

def _ar1_noise(n, rng, rho=0.5, sigma=0.15):
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = rho * e[t-1] + rng.normal(0, sigma)
    return e

def generate_series(typology, rng, censor=True, burst_prob=0.5):
    """typology in {'null','endogenous','exogenous'}; log10 scale.
    v2: a fraction of endogenous series is burst-initiated (Simge archetype)."""
    base = rng.uniform(1.0, 3.0)
    onset = int(rng.integers(18, T-24))
    lead_yt, lead_tr = int(rng.integers(1,4)), int(rng.integers(1,3))
    L = np.full(T, base, float)
    if typology == "endogenous":
        amp = rng.uniform(1.0, 2.0); k = rng.uniform(0.6, 1.4)
        ramp = amp/(1+np.exp(-k*(np.arange(T)-onset-6)))
        ds = onset + int(rng.integers(8,14)); d = rng.uniform(0.06,0.12)
        plateau = rng.uniform(0.35,0.6); decay = np.ones(T)
        for t in range(ds, T):
            decay[t] = plateau + (1-plateau)*np.exp(-d*(t-ds))
        L = base + ramp*decay
    elif typology == "exogenous":
        amp = rng.uniform(1.5,2.5); d = rng.uniform(0.25,0.5)
        for t in range(onset, T):
            L[t] = base + amp*np.exp(-d*(t-onset))
    sy = lead_yt if typology=="endogenous" else 0
    st = lead_tr if typology=="endogenous" else 0
    L_yt = np.concatenate([L[sy:], np.full(sy, L[-1])])
    L_tr = np.concatenate([L[st:], np.full(st, L[-1])])
    yt = L_yt + _ar1_noise(T, rng)
    tr0 = L_tr + _ar1_noise(T, rng)
    spu = L + _ar1_noise(T, rng)
    if typology=="endogenous" and rng.random() < burst_prob:
        # burst-initiated endogenous: 30-70% of eventual amplitude within 2 months
        for arr in (yt, tr0, spu):
            b = arr[:onset].mean(); amp2 = arr[onset:].max()-b
            frac = rng.uniform(0.3, 0.7)
            for t in range(onset, min(onset+2, T)):
                arr[t] = max(arr[t], b + frac*amp2 + rng.normal(0, 0.05))
    if censor:
        c = base + rng.uniform(0.5, 0.9)
        sp = np.where(spu >= c, spu, 0.0)
    else:
        sp = spu
    lin = 10**tr0
    tr = np.log10(np.maximum(1, np.round(100*lin/lin.max())))
    return dict(typology=typology, onset=onset, sp=sp, yt=yt, tr=tr,
                sp_uncensored=spu)

# ---------- causal flags ----------
def causal_flags(x, thr, burn_in=12, censored=False, persist=2):
    x = np.asarray(x, float); raw = np.zeros(len(x), bool)
    for t in range(burn_in, len(x)):
        past = x[:t]
        if censored:
            past = past[past > 0]
            if len(past) < 6 or x[t] == 0: continue
        sd = past.std() or 1e-6
        if x[t] > past.mean() + thr*sd: raw[t] = True
    out = np.zeros_like(raw)
    for t in range(persist-1, len(raw)):
        if raw[max(0,t-persist+1):t+1].sum() >= persist: out[t] = True
    return out

def cusum_flags(x, k=0.5, h=6.0, burn_in=12, censored=False):
    x = np.asarray(x, float); Cst = 0.0; out = np.zeros(len(x), bool)
    for t in range(burn_in, len(x)):
        past = x[:t]
        if censored:
            past = past[past > 0]
            if len(past) < 6 or x[t] == 0: continue
        sd = past.std() or 1e-6
        Cst = max(0.0, Cst + (x[t]-past.mean())/sd - k)
        if Cst > h: out[t] = True
    return out

def composite_flags(s, thr, burn_in=12, persist=2):
    """Causal CP-PSI: mean of the three expanding-window z-scores (single series)."""
    n = len(s["yt"]); comp = np.zeros(n)
    for t in range(burn_in, n):
        zs = []
        for x, cens in [(s["yt"],False),(s["tr"],False),(s["sp"],True)]:
            past = x[:t]
            if cens:
                past = past[past>0]
                if len(past)<6 or x[t]==0: continue
            sd = past.std() or 1e-6
            zs.append((x[t]-past.mean())/sd)
        comp[t] = np.mean(zs) if zs else 0.0
    raw = comp > thr
    out = np.zeros_like(raw)
    for t in range(persist-1, n):
        if raw[max(0,t-persist+1):t+1].sum() >= persist: out[t] = True
    return out

def first_confirmed(allf, confirm, window=2):
    for t in range(allf.shape[1]):
        lo = max(0, t-window+1)
        if allf[:, lo:t+1].any(axis=1).sum() >= confirm: return t
    return None

def detect_pe(s, thr=1.75, confirm=2, window=2, persist=2):
    allf = np.vstack([causal_flags(s["yt"],thr,persist=persist),
                      causal_flags(s["tr"],thr,persist=persist),
                      causal_flags(s["sp"],thr,censored=True,persist=persist)])
    return first_confirmed(allf, confirm, window)

def detect_cusum(s, k=0.5, h=6.0, confirm=2, window=2):
    allf = np.vstack([cusum_flags(s["yt"],k,h), cusum_flags(s["tr"],k,h),
                      cusum_flags(s["sp"],k,h,censored=True)])
    return first_confirmed(allf, confirm, window)

def detect_composite(s, thr=1.75, persist=2):
    f = composite_flags(s, thr, persist=persist)
    return int(f.argmax()) if f.any() else None

# ---------- regime classification ----------
def rise_ratio(yt, ign, h=12):
    b = yt[max(0,ign-6):ign].mean()
    peak = yt[ign:min(ign+h+1,len(yt))].max()
    return (yt[ign]-b)/max(peak-b, 1e-6)

def causal_features(s, ign, h):
    yt, tr, sp = s["yt"], s["tr"], s["sp"]
    b = yt[max(0,ign-6):ign].mean()
    w = yt[ign:min(ign+h+1,len(yt))]
    def jz(x, c=False):
        p = x[:ign]; p = p[p>0] if c else p
        return abs(x[ign]-p[-1])/(p.std() or 1e-6) if len(p)>5 else 0.0
    trb = tr[max(0,ign-6):ign].mean()
    tramp = max(tr[ign:min(ign+h+1,len(tr))].max()-trb, 1e-6)
    return [rise_ratio(yt,ign,h), (tr[ign]-trb)/tramp,
            (w.max()-yt[ign])/max(w.max()-b,1e-6),
            np.argmax(w)/max(h,1),
            (w[-1]-w.max())/max(w.max()-b,1e-6),
            jz(yt), jz(tr), jz(sp,True),
            yt[max(0,ign-12):ign].std()]
print("core module loaded")