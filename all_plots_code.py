import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np

df = pd.read_csv('radke_metrics_v4.csv')

# ── shared helpers ────────────────────────────────────────────────────────────
teams = sorted(df['Team'].unique())
palette = {
    'Team A':'#378ADD','Team B':'#D85A30','Team C':'#1D9E75',
    'Team D':'#BA7517','Team E':'#533AB7','Team F':'#993556',
    'Team G':'#3B6D11','Team H':'#185FA5','Team I':'#A32D2D',
    'Team J':'#0F6E56','Team K':'#639922','Team L':'#854F0B',
}
BASE='#2c2c2c'; OUTLINE='#555555'; HL_RING='#d63b2f'; ANNO_BG='#f5f5f5'

def dot_size(v, mn, mx, s_min=20, s_max=180):
    if pd.isna(v): return 20
    return s_min + (s_max-s_min)*(v-mn)/(mx-mn)

def quad_outliers(sub, xc, yc, xmed, ymed, n=1):
    sub = sub.dropna(subset=[xc,yc]).copy()
    corners = {'HH':(1,1),'HL':(1,-1),'LH':(-1,1),'LL':(-1,-1)}
    masks = {
        'HH':(sub[xc]>=xmed)&(sub[yc]>=ymed),
        'HL':(sub[xc]>=xmed)&(sub[yc]< ymed),
        'LH':(sub[xc]< xmed)&(sub[yc]>=ymed),
        'LL':(sub[xc]< xmed)&(sub[yc]< ymed),
    }
    out={}
    for q,mask in masks.items():
        qdf=sub[mask].copy()
        if qdf.empty: continue
        cx,cy=corners[q]
        qdf['_s']=cx*(qdf[xc]-xmed)+cy*(qdf[yc]-ymed)
        out[q]=qdf.nlargest(n,'_s').iloc[0]
    return out

def annotate_outliers(ax, sub, xc, yc, xmed, ymed, size_col, s_mn, s_mx):
    for q, row in quad_outliers(sub,xc,yc,xmed,ymed).items():
        px,py=row[xc],row[yc]
        s=dot_size(row[size_col],s_mn,s_mx)
        ax.scatter(px,py,s=s+100,color='none',edgecolors=HL_RING,linewidths=2.0,zorder=5)
        ax.scatter(px,py,s=s,color='#c0392b',alpha=0.92,linewidths=0.4,
                   edgecolors='white',zorder=6)
        lbl=f"{row['Team']} #{int(row['Player'])}"
        xrng=sub[xc].max()-sub[xc].min(); yrng=sub[yc].max()-sub[yc].min()
        ox=xrng*0.05 if px<xmed else -xrng*0.05
        oy=yrng*0.06 if py<ymed else -yrng*0.06
        ax.annotate(lbl,xy=(px,py),xytext=(px+ox,py+oy),color='#1a1a1a',fontsize=7.5,
            arrowprops=dict(arrowstyle='-',color='#888',lw=0.8),
            bbox=dict(boxstyle='round,pad=0.25',facecolor=ANNO_BG,
                      edgecolor='#ccc',linewidth=0.6,alpha=0.92),zorder=7)

def style_ax(ax, title, xl, yl):
    ax.set_title(title,color='#1a1a1a',fontsize=10,pad=8)
    ax.set_xlabel(xl,color='#555',fontsize=9.5)
    ax.set_ylabel(yl,color='#555',fontsize=9.5)
    ax.tick_params(colors='#888',labelsize=8.5)
    for sp in ax.spines.values(): sp.set_edgecolor('#ccc')
    ax.grid(True,color='#e8e8e8',linewidth=0.5,zorder=0)
    ax.set_facecolor('white')

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — offensive 2x2 bubble (PAA, NPPM, PrMA, TOA vs each other, PrSOA size)
# ═══════════════════════════════════════════════════════════════════════════════
med_paa=df['PAA'].median(); med_nppm=df['NPPM'].median()
med_prma=df['PrMA'].median(); med_toa=df['TOA'].median()
soa_min,soa_max=df['PrSOA'].min(),df['PrSOA'].max()

offensive_plots=[
    ('PAA','PrMA','PrSOA',soa_min,soa_max,
     'PAA — pass lane quality (γ)','PrMA — pressure when carrying',
     med_paa,med_prma,'Pass lane quality vs pressure when carrying','Dot size = PrSOA'),
    ('PAA','NPPM','PrSOA',soa_min,soa_max,
     'PAA — pass lane quality (γ)','NPPM — normalised passing +/−',
     med_paa,med_nppm,'Pass lane quality vs passing effectiveness','Dot size = PrSOA'),
    ('PrMA','NPPM','PrSOA',soa_min,soa_max,
     'PrMA — pressure when carrying','NPPM — normalised passing +/−',
     med_prma,med_nppm,'Pressure resilience','Dot size = PrSOA'),
    ('TOA','NPPM','PrSOA',soa_min,soa_max,
     'TOA — turnovers per 60 ES','NPPM — normalised passing +/−',
     med_toa,med_nppm,'Turnover rate vs passing effectiveness','Dot size = PrSOA'),
]
off_quad_text=[
    {'HH':'High quality\nunder pressure','HL':'Open passing,\nlow pressure',
     'LH':'Tight lanes,\nhigh pressure','LL':'Low quality,\nlow pressure'},
    {'HH':'Open lanes &\neffective passer','HL':'Open lanes,\npoor efficiency',
     'LH':'Tight lanes,\neffective passer','LL':'Tight lanes,\npoor efficiency'},
    {'HH':'Pressure-resilient\npasser','HL':'Heavy pressure,\nstruggling',
     'LH':'Low pressure,\neffective','LL':'Low pressure,\npoor output'},
    {'HH':'High TO,\nstill effective','HL':'High TO &\npoor passing',
     'LH':'Efficient,\nlow TO','LL':'Low TO,\npoor efficiency'},
]

fig,axes=plt.subplots(2,2,figsize=(18,15)); fig.patch.set_facecolor('white')
fig.suptitle('Offensive passing profiles — dot size = PrSOA\nCircled = quadrant outliers',
             color='#1a1a1a',fontsize=13,y=0.998)

for pidx,(ax,(xc,yc,sc,s_mn,s_mx,xl,yl,xmed,ymed,title,slabel)) in enumerate(
        zip(axes.flat,offensive_plots)):
    sub=df.dropna(subset=[xc,yc])
    for team in teams:
        t=sub[sub['Team']==team]; has_soa=t['PrSOA'].notna()
        tf=t[has_soa]
        if len(tf):
            ax.scatter(tf[xc],tf[yc],s=[dot_size(v,s_mn,s_mx) for v in tf['PrSOA']],
                       color=BASE,alpha=0.55,linewidths=0.4,edgecolors=OUTLINE,zorder=3)
        tn=t[~has_soa]
        if len(tn):
            ax.scatter(tn[xc],tn[yc],s=22,facecolors='none',
                       edgecolors=OUTLINE,linewidths=0.8,alpha=0.4,zorder=2)
    ax.axvline(xmed,color='#aaa',linewidth=0.9,linestyle='--',alpha=0.55,zorder=1)
    ax.axhline(ymed,color='#aaa',linewidth=0.9,linestyle='--',alpha=0.55,zorder=1)
    for q,(xf,yf,ha,va) in [('HH',(0.97,0.97,'right','top')),('HL',(0.97,0.03,'right','bottom')),
                              ('LH',(0.03,0.97,'left','top')), ('LL',(0.03,0.03,'left','bottom'))]:
        ax.text(xf,yf,off_quad_text[pidx][q],transform=ax.transAxes,
                color='#1a1a1a',fontsize=7.5,ha=ha,va=va,fontstyle='italic')
    annotate_outliers(ax,sub,xc,yc,xmed,ymed,sc,s_mn,s_mx)
    style_ax(ax,title,xl,yl)
    ax.text(0.5,-0.08,slabel,transform=ax.transAxes,color='#999',
            fontsize=7.5,ha='center',va='top',fontstyle='italic')

plt.tight_layout(rect=[0,0.04,1,0.985],h_pad=3.5,w_pad=3.0)
plt.savefig('radke_2x2_highlighted.png',dpi=180,bbox_inches='tight',facecolor='white')
plt.close(); print("Plot 1 saved")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — defensive 2x2 (BTT, PrSOA, Coverage, single colour + outliers)
# ═══════════════════════════════════════════════════════════════════════════════
med_btt=df['BTT'].median(); med_prsoa=df['PrSOA'].median()
med_coverage=df['Coverage_pct'].median()
btt_min,btt_max=df['BTT'].min(),df['BTT'].max()
soa_min2,soa_max2=df['PrSOA'].min(),df['PrSOA'].max()
cov_min,cov_max=df['Coverage_pct'].min(),df['Coverage_pct'].max()

def_plots=[
    ('BTT','PAA','PrSOA',soa_min2,soa_max2,
     'BTT — beaten by opponent passes','PAA — pass lane quality (γ)',
     med_btt,med_paa,'Defensive exposure (BTT) vs pass lane quality','Dot size = PrSOA'),
    ('BTT','NPPM','PrSOA',soa_min2,soa_max2,
     'BTT — beaten by opponent passes','NPPM — normalised passing +/−',
     med_btt,med_nppm,'Defensive exposure (BTT) vs passing effectiveness','Dot size = PrSOA'),
    ('PrSOA','BTT','Coverage_pct',cov_min,cov_max,
     'PrSOA — pressure at opponent shots','BTT — beaten by opponent passes',
     med_prsoa,med_btt,'Defensive pressure at shots vs being beaten by passes',
     'Dot size = Coverage %'),
    ('Coverage_pct','NPPM','BTT',btt_min,btt_max,
     'Coverage % — defensive presence','NPPM — normalised passing +/−',
     med_coverage,med_nppm,'Defensive coverage vs passing effectiveness','Dot size = BTT'),
]
def_quad_text=[
    {'HH':'Beaten often,\nstill open lanes','HL':'Beaten &\ntight lanes',
     'LH':'Well-defended,\nopen lanes','LL':'Low exposure,\ntight lanes'},
    {'HH':'Beaten often\nbut effective','HL':'Beaten &\nstruggling',
     'LH':'Rarely beaten,\neffective','LL':'Low exposure,\npoor output'},
    {'HH':'Aggressive at shots\nbut beaten in open play','HL':'High pressure,\nrarely beaten',
     'LH':'Low pressure,\ngets beaten','LL':'Low pressure,\nrarely beaten'},
    {'HH':'High coverage\n& effective','HL':'High coverage,\npoor output',
     'LH':'Low coverage,\neffective','LL':'Low coverage &\npoor output'},
]

fig,axes=plt.subplots(2,2,figsize=(18,15)); fig.patch.set_facecolor('white')
fig.suptitle('Defensive metrics — who is blocking lanes and causing chaos?\nCircled = quadrant outliers',
             color='#1a1a1a',fontsize=13,y=0.998)

for pidx,(ax,(xc,yc,sc,s_mn,s_mx,xl,yl,xmed,ymed,title,slabel)) in enumerate(
        zip(axes.flat,def_plots)):
    sub=df.dropna(subset=[xc,yc])
    ax.scatter(sub[xc],sub[yc],s=[dot_size(v,s_mn,s_mx) for v in sub[sc]],
               color=BASE,alpha=0.55,linewidths=0.4,edgecolors=OUTLINE,zorder=3)
    ax.axvline(xmed,color='#aaa',linewidth=0.9,linestyle='--',alpha=0.55,zorder=1)
    ax.axhline(ymed,color='#aaa',linewidth=0.9,linestyle='--',alpha=0.55,zorder=1)
    for q,(xf,yf,ha,va) in [('HH',(0.97,0.97,'right','top')),('HL',(0.97,0.03,'right','bottom')),
                              ('LH',(0.03,0.97,'left','top')), ('LL',(0.03,0.03,'left','bottom'))]:
        ax.text(xf,yf,def_quad_text[pidx][q],transform=ax.transAxes,
                color='#1a1a1a',fontsize=7.5,ha=ha,va=va,fontstyle='italic')
    annotate_outliers(ax,sub,xc,yc,xmed,ymed,sc,s_mn,s_mx)
    style_ax(ax,title,xl,yl)
    ax.text(0.5,-0.08,slabel,transform=ax.transAxes,color='#999',
            fontsize=7.5,ha='center',va='top',fontstyle='italic')

plt.tight_layout(rect=[0,0.04,1,0.985],h_pad=4.0,w_pad=3.5)
plt.savefig('radke_defensive_2x2.png',dpi=180,bbox_inches='tight',facecolor='white')
plt.close(); print("Plot 2 saved")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — standalone PrSOA vs BTT (dot size = Coverage %)
# ═══════════════════════════════════════════════════════════════════════════════
fig,ax=plt.subplots(figsize=(10,8)); fig.patch.set_facecolor('white'); ax.set_facecolor('white')
sub=df.dropna(subset=['PrSOA','BTT'])
ax.scatter(sub['PrSOA'],sub['BTT'],
           s=[dot_size(v,cov_min,cov_max) for v in sub['Coverage_pct']],
           color=BASE,alpha=0.55,linewidths=0.4,edgecolors=OUTLINE,zorder=3)
ax.axvline(med_prsoa,color='#aaa',linewidth=1.0,linestyle='--',alpha=0.7,zorder=1)
ax.axhline(med_btt,  color='#aaa',linewidth=1.0,linestyle='--',alpha=0.7,zorder=1)
prsoa_btt_quads={
    'HH':'Aggressive at shots\nbut beaten in open play',
    'HL':'High pressure at shots\n& rarely beaten — ideal',
    'LH':'Low pressure at shots\nbut gets bypassed',
    'LL':'Low pressure\n& rarely beaten',
}
for q,(xf,yf,ha,va) in [('HH',(0.97,0.97,'right','top')),('HL',(0.97,0.03,'right','bottom')),
                          ('LH',(0.03,0.97,'left','top')), ('LL',(0.03,0.03,'left','bottom'))]:
    ax.text(xf,yf,prsoa_btt_quads[q],transform=ax.transAxes,
            color='#1a1a1a',fontsize=8.5,ha=ha,va=va,fontstyle='italic')
annotate_outliers(ax,sub,'PrSOA','BTT',med_prsoa,med_btt,'Coverage_pct',cov_min,cov_max)
for v,lbl in [(cov_min,f'Low ({cov_min:.0f}%)'),(95.5,'Mid (95.5%)'),(cov_max,f'Full ({cov_max:.0f}%)')]:
    ax.scatter([],[],s=dot_size(v,cov_min,cov_max),color=BASE,alpha=0.6,
               edgecolors=OUTLINE,linewidths=0.4,label=lbl)
ax.scatter([],[],s=dot_size(95,cov_min,cov_max)+100,color='none',
           edgecolors=HL_RING,linewidths=2.2,label='Quadrant outlier')
ax.legend(title='Dot size = Coverage %',title_fontsize=8.5,fontsize=8.5,
          frameon=True,framealpha=0.9,edgecolor='#ddd',labelcolor='#444',loc='upper left')
style_ax(ax,'Defensive pressure at shots (PrSOA) vs beaten by passes (BTT)\n'
            'Dot size = Coverage % — circled = quadrant outliers',
         'PrSOA — average pressure applied at opponent shots',
         'BTT — beaten by opponent passes (total)')
plt.tight_layout()
plt.savefig('radke_plot3_prsoa_vs_btt.png',dpi=180,bbox_inches='tight',facecolor='white')
plt.close(); print("Plot 3 saved")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 4 — team defensive summary (PrSOA vs BTT, colour=NPPM, size=PAA)
# ═══════════════════════════════════════════════════════════════════════════════
team_avgs=df.groupby('Team').agg(
    PAA=('PAA','mean'),NPPM=('NPPM','mean'),PrMA=('PrMA','mean'),
    PrSOA=('PrSOA','mean'),BTT=('BTT','mean'),Coverage=('Coverage_pct','mean')).reset_index()
paa_min2,paa_max2=team_avgs['PAA'].min(),team_avgs['PAA'].max()
med_t_prsoa=team_avgs['PrSOA'].median(); med_t_btt=team_avgs['BTT'].median()

def tdot(v,mn,mx,s_min=120,s_max=600):
    return s_min+(s_max-s_min)*(v-mn)/(mx-mn)

fig,ax=plt.subplots(figsize=(11,9)); fig.patch.set_facecolor('white'); ax.set_facecolor('white')
norm_nppm=plt.Normalize(team_avgs['NPPM'].min(),team_avgs['NPPM'].max())
cmap=plt.cm.RdBu
for _,row in team_avgs.iterrows():
    ax.scatter(row['PrSOA'],row['BTT'],s=tdot(row['PAA'],paa_min2,paa_max2),
               color=cmap(norm_nppm(row['NPPM'])),alpha=0.82,linewidths=1.0,
               edgecolors='#555',zorder=3)
    ax.annotate(row['Team'],xy=(row['PrSOA'],row['BTT']),
                xytext=(row['PrSOA']+0.008,row['BTT']+0.18),
                color='#1a1a1a',fontsize=8.5,
                bbox=dict(boxstyle='round,pad=0.25',facecolor='white',
                          edgecolor='#ccc',linewidth=0.5,alpha=0.9),zorder=5)
ax.axvline(med_t_prsoa,color='#aaa',linewidth=1.0,linestyle='--',alpha=0.65,zorder=1)
ax.axhline(med_t_btt,  color='#aaa',linewidth=1.0,linestyle='--',alpha=0.65,zorder=1)
tql={
    (0.97,0.97,'right','top'):   'Aggressive at shots\nbut beaten in open play',
    (0.97,0.03,'right','bottom'):'Strong defensive profile\nhigh pressure, low BTT',
    (0.03,0.97,'left','top'):    'Low shot pressure\nbut gets bypassed',
    (0.03,0.03,'left','bottom'): 'Low pressure\n& rarely beaten',
}
for (xf,yf,ha,va),txt in tql.items():
    ax.text(xf,yf,txt,transform=ax.transAxes,color='#1a1a1a',
            fontsize=8,ha=ha,va=va,fontstyle='italic')
sm=plt.cm.ScalarMappable(cmap=cmap,norm=norm_nppm); sm.set_array([])
cbar=fig.colorbar(sm,ax=ax,shrink=0.6,pad=0.02)
cbar.set_label('Avg NPPM (blue = positive, red = negative)',fontsize=8.5,color='#555')
cbar.ax.tick_params(labelsize=8,colors='#888'); cbar.outline.set_edgecolor('#ddd')
for v,lbl in [(paa_min2,f'Low PAA ({paa_min2:.2f})'),
              ((paa_min2+paa_max2)/2,f'Mid PAA ({(paa_min2+paa_max2)/2:.2f})'),
              (paa_max2,f'High PAA ({paa_max2:.2f})')]:
    ax.scatter([],[],s=tdot(v,paa_min2,paa_max2),color='#888',alpha=0.7,
               edgecolors='#555',linewidths=0.8,label=lbl)
ax.legend(title='Dot size = avg PAA',title_fontsize=8.5,fontsize=8.5,
          frameon=True,framealpha=0.9,edgecolor='#ddd',labelcolor='#444',loc='upper left')
style_ax(ax,'Team-level defensive profile — PrSOA vs BTT\nDot size = avg PAA  |  Dot colour = avg NPPM',
         'Avg PrSOA — pressure applied at opponent shots','Avg BTT — beaten by opponent passes')
plt.tight_layout()
plt.savefig('radke_team_defensive_summary.png',dpi=180,bbox_inches='tight',facecolor='white')
plt.close(); print("Plot 4 saved")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 5 — Radke Fig 3 remake: PrMA vs TOA (a) and PAA vs TOA (b)
# ═══════════════════════════════════════════════════════════════════════════════
df['group']=df['NPPM'].apply(lambda x:'Positive NPPM' if x>=0 else 'Negative NPPM')
COL_POS='#1a7abf'; COL_NEG='#2e8b57'

highlights_fig3={
    'prma':{'HH':('Team A',6),'HL':('Team F',29),'LH':('Team I',27),'LL':('Team H',13)},
    'paa': {'HH':('Team F',29),'HL':('Team A',6),'LH':('Team C',44),'LL':('Team H',16)},
}

def draw_fig3(ax, yc, ymed, ylabel, plot_key, panel_label):
    ax.set_facecolor('#f0f0f0')
    sub=df.dropna(subset=['TOA',yc])
    for grp,col,mk in [('Positive NPPM',COL_POS,'o'),('Negative NPPM',COL_NEG,'s')]:
        g=sub[sub['group']==grp]
        ax.scatter(g['TOA'],g[yc],c=col,marker=mk,s=38,alpha=0.75,
                   linewidths=0.4,edgecolors='white',zorder=3,label=grp)
    ax.axhline(ymed,   color='#cc3333',linewidth=1.2,linestyle='-',alpha=0.8,zorder=2)
    ax.axvline(med_toa,color='#3399cc',linewidth=1.2,linestyle='-',alpha=0.8,zorder=2)
    for q,(team,player) in highlights_fig3[plot_key].items():
        row=df[(df['Team']==team)&(df['Player'].astype(str)==str(player))].iloc[0]
        if pd.isna(row['TOA']) or pd.isna(row[yc]): continue
        px,py=row['TOA'],row[yc]
        col=COL_POS if row['group']=='Positive NPPM' else COL_NEG
        mk='o' if row['group']=='Positive NPPM' else 's'
        ax.scatter(px,py,s=120,c=col,marker=mk,linewidths=1.8,edgecolors='white',zorder=5)
        ax.scatter(px,py,s=220,color='none',edgecolors=HL_RING,linewidths=1.6,zorder=6)
        lbl=f"{team} #{player}"
        xrng=sub['TOA'].max()-sub['TOA'].min(); yrng=sub[yc].max()-sub[yc].min()
        ox=-xrng*0.04 if px>med_toa else xrng*0.03
        oy= yrng*0.04 if py>ymed    else -yrng*0.04
        ax.annotate(lbl,xy=(px,py),xytext=(px+ox,py+oy),color='#1a1a1a',fontsize=8,
            arrowprops=dict(arrowstyle='-',color='#777',lw=0.7),
            bbox=dict(boxstyle='round,pad=0.22',facecolor='white',
                      edgecolor='#ccc',linewidth=0.5,alpha=0.95),zorder=7)
    ax.set_ylabel(ylabel,color='#1a1a1a',fontsize=10)
    ax.set_xlabel('TOA',color='#1a1a1a',fontsize=10)
    ax.tick_params(colors='#555',labelsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor('#ccc')
    ax.grid(True,color='white',linewidth=0.7,zorder=0)
    pos_h=mlines.Line2D([],[],color=COL_POS,marker='o',linestyle='None',markersize=6,label='Positive NPPM')
    neg_h=mlines.Line2D([],[],color=COL_NEG,marker='s',linestyle='None',markersize=6,label='Negative NPPM')
    pm_h =mlines.Line2D([],[],color='#cc3333',linewidth=1.2,label=f'{ylabel.split("—")[0].strip()} Median')
    tm_h =mlines.Line2D([],[],color='#3399cc',linewidth=1.2,label='TOA Median')
    ax.legend(handles=[pm_h,tm_h,pos_h,neg_h],fontsize=7.5,frameon=True,
              framealpha=0.85,edgecolor='#ccc',loc='upper right')
    ax.text(0.012,0.015,f'({panel_label})',transform=ax.transAxes,fontsize=10,color='#333',va='bottom')

fig,(ax1,ax2)=plt.subplots(2,1,figsize=(8,11)); fig.patch.set_facecolor('white')
draw_fig3(ax1,'PrMA',med_prma,'PrMA — pressure when carrying','prma','a')
draw_fig3(ax2,'PAA', med_paa, 'PAA — pass lane quality (γ)',  'paa', 'b')
fig.suptitle('PrMA and PAA vs TOA (n=233)',color='#1a1a1a',fontsize=11,y=1.01)
plt.tight_layout(h_pad=3.0)
plt.savefig('radke_fig3_remake.png',dpi=180,bbox_inches='tight',facecolor='white')
plt.close(); print("Plot 5 saved")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 6 — OVT vs BTT zero-sum passing game (dot colour = NPPM)
# ═══════════════════════════════════════════════════════════════════════════════
med_ovt = df['OVT'].median()
med_btt2 = df['BTT'].median()
nppm_min = df['NPPM'].min(); nppm_max = df['NPPM'].max()

def quad_outliers_ovt(sub, xc, yc, xmed, ymed):
    sub = sub.dropna(subset=[xc, yc]).copy()
    corners = {'HH':(1,1),'HL':(1,-1),'LH':(-1,1),'LL':(-1,-1)}
    masks = {
        'HH':(sub[xc]>=xmed)&(sub[yc]>=ymed),
        'HL':(sub[xc]>=xmed)&(sub[yc]< ymed),
        'LH':(sub[xc]< xmed)&(sub[yc]>=ymed),
        'LL':(sub[xc]< xmed)&(sub[yc]< ymed),
    }
    out = {}
    for q, mask in masks.items():
        qdf = sub[mask].copy()
        if qdf.empty: continue
        cx, cy = corners[q]
        qdf['_s'] = cx*(qdf[xc]-xmed) + cy*(qdf[yc]-ymed)
        out[q] = qdf.nlargest(1,'_s').iloc[0]
    return out

fig, ax = plt.subplots(figsize=(10, 8)); fig.patch.set_facecolor('white'); ax.set_facecolor('white')
norm_ovt = plt.Normalize(nppm_min, nppm_max); cmap_ovt = plt.cm.RdBu
sc = ax.scatter(df['OVT'], df['BTT'], c=df['NPPM'], cmap=cmap_ovt, norm=norm_ovt,
                s=45, alpha=0.72, linewidths=0.4, edgecolors='#aaa', zorder=3)
max_val = max(df['OVT'].max(), df['BTT'].max()) * 1.05
ax.plot([0, max_val], [0, max_val], color='#aaa', linewidth=1.0,
        linestyle='--', alpha=0.6, zorder=1)
ax.axvline(med_ovt,  color='#888', linewidth=0.9, linestyle=':', alpha=0.55, zorder=1)
ax.axhline(med_btt2, color='#888', linewidth=0.9, linestyle=':', alpha=0.55, zorder=1)
ovt_ql = {
    (0.97,0.97,'right','top'):   'High OVT, high BTT\ntwo-way active',
    (0.97,0.03,'right','bottom'):'High OVT, low BTT\nelite passing profile',
    (0.03,0.97,'left','top'):    'Low OVT, high BTT\nliability profile',
    (0.03,0.03,'left','bottom'): 'Low OVT, low BTT\nlow involvement',
}
for (xf,yf,ha,va),txt in ovt_ql.items():
    ax.text(xf,yf,txt,transform=ax.transAxes,color='#1a1a1a',
            fontsize=8.5,ha=ha,va=va,fontstyle='italic')
for q, row in quad_outliers_ovt(df,'OVT','BTT',med_ovt,med_btt2).items():
    px,py = row['OVT'],row['BTT']
    ax.scatter(px,py,s=110,c=[row['NPPM']],cmap=cmap_ovt,norm=norm_ovt,
               linewidths=0.8,edgecolors='#333',zorder=5)
    lbl = f"{row['Team']} #{int(row['Player'])}"
    xrng=df['OVT'].max()-df['OVT'].min(); yrng=df['BTT'].max()-df['BTT'].min()
    ox=-xrng*0.05 if px>med_ovt else xrng*0.03
    oy= yrng*0.05 if py>med_btt2 else -yrng*0.04
    ax.annotate(lbl,xy=(px,py),xytext=(px+ox,py+oy),color='#1a1a1a',fontsize=8,
        arrowprops=dict(arrowstyle='-',color='#777',lw=0.7),
        bbox=dict(boxstyle='round,pad=0.25',facecolor='white',
                  edgecolor='#ccc',linewidth=0.5,alpha=0.95),zorder=7)
cbar=fig.colorbar(sc,ax=ax,shrink=0.6,pad=0.02)
cbar.set_label('NPPM — normalised passing +/−\n(blue = positive, red = negative)',
               fontsize=8.5,color='#555')
cbar.ax.tick_params(labelsize=8,colors='#888'); cbar.outline.set_edgecolor('#ddd')
diag=mlines.Line2D([],[],color='#aaa',linewidth=1.0,linestyle='--',label='PPM = 0  (OVT = BTT)')
ax.legend(handles=[diag],fontsize=8.5,frameon=True,framealpha=0.9,
          edgecolor='#ddd',loc='upper left')
style_ax(ax,'Passing zero-sum game — OVT vs BTT\n'
            'Above diagonal = net positive passer  |  dot colour = NPPM',
         'OVT — opponents overtaken by passes (total)',
         'BTT — beaten by opponent passes (total)')
ax.set_xlim(-0.5,df['OVT'].max()*1.06); ax.set_ylim(-0.5,df['BTT'].max()*1.06)
plt.tight_layout()
plt.savefig('radke_ovt_vs_btt.png',dpi=180,bbox_inches='tight',facecolor='white')
plt.close(); print("Plot 6 saved")
