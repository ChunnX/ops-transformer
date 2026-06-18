#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fag_viz.py —— FAG 分核/枚举可视化 (import fa_core)

  --mode split     §13.6.1 六种枚举在 (S1o×S2o) 网格上按核上色 -> fag_split_viz.png
  --mode l2window  线性scan / valid-space swizzle / macro swizzle 的 L2 活跃 K/V 窗口 -> fag_l2_window.png
  --mode both      两张都出 (默认)
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = SCRIPT_DIR / "fig"
MPL_CACHE_DIR = Path(os.environ.get("MPLCONFIGDIR", Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib-cache"))
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPL_CACHE_DIR)

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib import font_manager, gridspec
from fa_core import valid_linear, valid_macro, assign_linear, assign_stride, simulate_window

def _setup_cjk_font():
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.otf",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    for font_path in candidates:
        if Path(font_path).exists():
            font_manager.fontManager.addfont(font_path)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
            return

    keywords = ("NotoSansCJK", "Noto Sans CJK", "SourceHan", "PingFang", "STHeiti", "Heiti", "Songti")
    for font_path in font_manager.findSystemFonts():
        if any(key in font_path for key in keywords):
            font_manager.fontManager.addfont(font_path)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
            return

_setup_cjk_font()
plt.rcParams["axes.unicode_minus"] = False

AIC, CBN, MS = 4, 2, 4
CORE_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
INVALID, EMPTY = "#E8E6E0", "#FBFBFA"
def cc(c): return CORE_COLORS[c % len(CORE_COLORS)]

# =================================================================== split 网格图
def _cell(ax,x,y,color,ec="white",lw=1.0,txt=None,tcol="white",hatch=None):
    ax.add_patch(Rectangle((x,y),1,1,facecolor=color,edgecolor=ec,lw=lw,hatch=hatch))
    if txt is not None: ax.text(x+0.5,y+0.5,txt,ha="center",va="center",fontsize=7.5,color=tcol)
def _setup(ax,S2o,S1o,title,sub):
    ax.set_xlim(-0.2,S2o+0.2); ax.set_ylim(-0.2,S1o+0.9); ax.invert_yaxis()
    ax.set_aspect("equal"); ax.axis("off")
    ax.text(S2o/2,-0.55,title,ha="center",fontsize=12,fontweight="bold")
    ax.text(S2o/2,S1o+0.45,sub,ha="center",fontsize=8.5,color="#555")
    ax.text(S2o/2,-0.05,"S2o (key 块) →",ha="center",va="bottom",fontsize=7.5,color="#888")
    ax.text(-0.35,S1o/2,"← S1o (query 块)",ha="center",va="center",fontsize=7.5,color="#888",rotation=90)

def _p_linear(ax,N=8):
    valid=[(s1,s2) for s2 in range(N) for s1 in range(N) if s2<=s1]
    per=len(valid)/AIC; own={v:min(AIC-1,int(i//per)) for i,v in enumerate(valid)}
    w=0
    for s1 in range(N):
        for s2 in range(N):
            if (s1,s2) in own: _cell(ax,s2,s1,cc(own[(s1,s2)]),txt=f"c{own[(s1,s2)]}")
            else: _cell(ax,s2,s1,INVALID,hatch="///",tcol="#aaa",txt="skip"); w+=1
    _setup(ax,N,N,"① 普通线性 scan (causal)",f"owner=order//chunk·每核连续一段\n顺序跳无效(空扫 {w} 块)")
def _p_dense(ax,N=8):
    for o in range(N*N):
        s1,s2=o%N,o//N; c=(o//CBN)%AIC; _cell(ax,s2,s1,cc(c),txt=f"c{c}")
    _setup(ax,N,N,"② 非 TND dense swizzle (full)",f"owner=(order//{CBN})%{AIC}·跨核stride\n每核{CBN}连续块再轮转→滑动窗口(保L2)")
def _p_macro(ax,N=8,ms=4):
    o=0
    for m1 in range(N//ms):
        for m2 in range(m1+1):
            dg=(m1==m2)
            for l2 in range(ms):
                for l1 in range(ms):
                    s1,s2=m1*ms+l1,m2*ms+l2
                    if dg and s2>s1: continue
                    c=(o//CBN)%AIC; _cell(ax,s2,s1,cc(c),txt=f"c{c}"); o+=1
    for s1 in range(N):
        for s2 in range(N):
            if s2>s1 and (s2//ms!=s1//ms): _cell(ax,s2,s1,EMPTY,ec="#eee")
    for k in range(0,N+1,ms):
        ax.plot([0,N],[k,k],color="#333",lw=1.4,zorder=5); ax.plot([k,k],[0,N],color="#333",lw=1.4,zorder=5)
    _setup(ax,N,N,"③ 非 TND causal macro swizzle",f"宏块({ms}×{ms})聚类+三角内交错\n一宏块K/V一批核吃完再换→强L2;对角走真三角")
def _p_tnd_dense(ax):
    bs=[("b0",3,2),("b1",3,5),("b2",3,3)]; mx=max(b[2] for b in bs); x0=0; real=0; o=0; gap=0.7
    for nm,s1,s2 in bs:
        ax.add_patch(Rectangle((x0,0),mx,s1,fill=False,ls="--",ec="#bbb",lw=1.0))
        for c2 in range(mx):
            for c1 in range(s1):
                if c2<s2: c=(o//CBN)%AIC; _cell(ax,x0+c2,c1,cc(c),txt=f"c{c}"); o+=1; real+=1
                else: _cell(ax,x0+c2,c1,INVALID,hatch="xx",ec="#ddd",tcol="#bbb")
        ax.text(x0+mx/2,-0.45,nm,ha="center",fontsize=9,color="#444",fontweight="bold"); x0+=mx+gap
    tot=len(bs)*mx*bs[0][1]; W=x0-gap
    ax.set_xlim(-0.3,W+0.3); ax.set_ylim(-1.1,3+1.4); ax.invert_yaxis(); ax.set_aspect("equal"); ax.axis("off")
    ax.text(W/2,-0.95,"④ TND dense swizzle (varlen)",ha="center",fontsize=12,fontweight="bold")
    ax.text(W/2,3+0.55,f"按 batch prefix 紧排, 实块={real} vs pad-to-max={tot}",ha="center",fontsize=8.5,color="#555")
    ax.text(W/2,3+1.05,f"虚线/×纹=按最大S padding 的空算({(1-real/tot)*100:.0f}%, 被 TND 省掉)",ha="center",fontsize=8.5,color="#555")
def _p_tnd_causal(ax,N=8):
    o=0
    for s1 in range(N):
        for s2 in range(N):
            h=None if s2<=s1 else "..."; c=(o//CBN)%AIC; _cell(ax,s2,s1,cc(c),hatch=h,txt=f"c{c}"); o+=1
    _setup(ax,N,N,"⑤ TND causal swizzle (成对 N2G 镜像)","实色=group0正三角; 点纹=group1镜像填上半区\n两三角拼满矩形→利用率 50%→~100%")
def _p_tnd_band(ax,N=10,pre=1,nxt=2):
    o=0; ib=0
    for s1 in range(N):
        for s2 in range(N):
            if -nxt<=(s1-s2)<=pre: c=(o//CBN)%AIC; _cell(ax,s2,s1,cc(c),txt=f"c{c}"); o+=1; ib+=1
            else: _cell(ax,s2,s1,EMPTY,ec="#eee")
    _setup(ax,N,N,"⑥ TND band swizzle",f"只枚举带状envelope(宽≈{pre+nxt+1}块)\n带外空白跳过(跳过 {(1-ib/(N*N))*100:.0f}%, 省 1−w/S)")

def fig_split():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig,ax=plt.subplots(2,3,figsize=(17,11.5))
    _p_linear(ax[0,0]); _p_dense(ax[0,1]); _p_macro(ax[0,2])
    _p_tnd_dense(ax[1,0]); _p_tnd_causal(ax[1,1]); _p_tnd_band(ax[1,2])
    h=[Patch(facecolor=cc(c),edgecolor="white",label=f"AIC core {c}") for c in range(AIC)]
    h+=[Patch(facecolor=INVALID,hatch="///",edgecolor="#ccc",label="无效/跳过(空扫)"),
        Patch(facecolor=EMPTY,edgecolor="#ddd",label="未枚举(空白)")]
    fig.legend(handles=h,loc="lower center",ncol=6,fontsize=9.5,frameon=False,bbox_to_anchor=(0.5,0.005))
    fig.suptitle(f"FAG 任务枚举/分核方式可视化 (§13.6.1) — aicNum={AIC}, cbn={CBN}, 块号=处理它的核",fontsize=14,fontweight="bold",y=0.99)
    fig.text(0.5,0.955,"主线: 线性scan owner=order//chunk(连续,causal空扫多) ┃ swizzle owner=(order//cbn)%aic(cbn连续块+跨核stride,保L2又配平)",ha="center",fontsize=10,color="#444")
    plt.tight_layout(rect=[0,0.04,1,0.95])
    out=FIG_DIR / "fag_split_viz.png"; fig.savefig(out,dpi=140,bbox_inches="tight",facecolor="white")
    print("saved:",out)

# =================================================================== L2 窗口图
METHOD_COLORS={"线性 scan":"#E45756","valid-space swizzle":"#4C78A8","macro swizzle":"#54A24B"}
def _traj(ax,sim,title,note):
    ticks=np.arange(sim["T"]); ax.fill_between(ticks,sim["lo"],sim["hi"],color="#999",alpha=0.13,zorder=0,label="活跃 K/V 列跨度")
    for c in range(AIC):
        xs=[p[0] for p in sim["per_core"][c]]; ys=[p[1] for p in sim["per_core"][c]]
        ax.plot(xs,ys,"-o",color=cc(c),ms=3,lw=1.1,alpha=0.9,label=f"core {c}")
    avg=np.nanmean(sim["span"]); mx=np.nanmax(sim["span"])
    ax.set_title(f"{title}\n{note}\n平均活跃列跨度 ≈ {avg:.1f} (峰值 {mx:.0f})",fontsize=10.5)
    ax.set_xlabel("执行 tick(进度)→",fontsize=9); ax.set_ylabel("当拍读的 K/V 列 s2o",fontsize=9)
    N=16; ax.set_ylim(-0.5,N-0.5); ax.set_xlim(-1,sim["T"]); ax.grid(True,alpha=0.25,lw=0.5); return avg

def fig_l2window(N=16):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    lin=simulate_window(assign_linear(valid_linear(N),AIC),AIC)
    swz=simulate_window(assign_stride(valid_linear(N),AIC,CBN),AIC)
    mac=simulate_window(assign_stride(valid_macro(N,MS),AIC,CBN),AIC)
    fig=plt.figure(figsize=(16,10)); gs=gridspec.GridSpec(2,3,height_ratios=[1.25,1.0],hspace=0.5,wspace=0.28)
    a0=fig.add_subplot(gs[0,0]); a1=fig.add_subplot(gs[0,1]); a2=fig.add_subplot(gs[0,2])
    al=_traj(a0,lin,"① 线性 scan","owner=pos//chunk·各核散在三角不同列")
    as_=_traj(a1,swz,"② valid-space swizzle",f"owner=(pos//{CBN})%{AIC}·各核挤同列+整体滑动")
    am=_traj(a2,mac,"③ macro swizzle",f"宏块({MS}×{MS})内 stride·各核挤在一个宏块")
    a0.legend(fontsize=7.5,loc="upper left",ncol=2,framealpha=0.9)
    ab=fig.add_subplot(gs[1,:])
    for sim,name in [(lin,"线性 scan"),(swz,"valid-space swizzle"),(mac,"macro swizzle")]:
        ab.plot(np.arange(sim["T"]),sim["span"],lw=2,color=METHOD_COLORS[name],label=f"{name} (均值 {np.nanmean(sim['span']):.1f} 列)")
    ab.axhspan(0,4,color="#54A24B",alpha=0.06)
    ab.text(lin["T"]*0.995,2,"小跨度 = 活跃 K/V 列少 → 易驻 L2",ha="right",va="center",fontsize=9,color="#3a7a2c")
    ab.set_title("活跃 K/V 工作集(同时要驻留 L2 的列数)随执行推进 —— 越低越好",fontsize=11.5,fontweight="bold")
    ab.set_xlabel("执行 tick(进度)→",fontsize=9.5); ab.set_ylabel("活跃 K/V 列跨度\n(各核当拍 s2o 的 max−min+1)",fontsize=9.5)
    ab.set_ylim(0,N); ab.grid(True,alpha=0.3,lw=0.5); ab.legend(fontsize=10,loc="upper center",ncol=3,framealpha=0.95)
    fig.suptitle(f"swizzle 如何把活跃 K/V 压成滑动窗口 (causal, S1o=S2o={N} 块, aicNum={AIC}, cbn={CBN})",fontsize=14,fontweight="bold",y=0.985)
    fig.text(0.5,0.945,f"读法: 上排每线=一个核当拍读的 K/V 列, 阴影=各核列跨度(活跃工作集)。线性散开(≈{al:.0f}列), swizzle 挤成细带滑动(≈{as_:.0f}列), macro 挤宏块内(≈{am:.0f}列,且Q向也局部)。",ha="center",fontsize=9.5,color="#444")
    out=FIG_DIR / "fag_l2_window.png"; fig.savefig(out,dpi=140,bbox_inches="tight",facecolor="white")
    print(f"saved: {out}  (avg span 线性={al:.2f} swizzle={as_:.2f} macro={am:.2f})")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--mode",choices=["split","l2window","both"],default="both"); a=ap.parse_args()
    if a.mode in ("split","both"): fig_split()
    if a.mode in ("l2window","both"): fig_l2window()

if __name__=="__main__": main()
