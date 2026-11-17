#!/usr/bin/env python3
"""Analisis reproducible de triage Pasto Salud. Python 3.10+.
Instalar: pip install pandas numpy scipy matplotlib
Ejecutar: python analisis_triage.py [--csv archivo.csv] [--salida resultados]
Si no se indica --csv descarga el CSV completo del portal oficial.
"""
import argparse, hashlib, io, json, re, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

URL='https://www.datos.gov.co/api/views/vt5n-eu2r/rows.csv?accessType=DOWNLOAD'
ALPHA=.05

def esc(x):
    return (str(x).replace('\\','\\textbackslash{}').replace('&','\\&').replace('%','\\%').replace('_','\\_').replace('#','\\#').replace('$','\\$'))
def num(x): return f'{x:.4f}' if np.isfinite(x) else 'NA'
def fmtp(x): return f'{x:.3e}' if x < .0001 else f'{x:.4f}'
def date_time(df, d, t):
    # Las columnas de hora pueden contener una fecha ficticia 1900-01-01.
    ds=pd.to_datetime(df[d],errors='coerce',dayfirst=False)
    raw=df[t].astype('string').str.strip()
    hh=raw.str.extract(r'(\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?)\s*$',flags=re.I)[0]
    parsed=pd.to_datetime(hh,errors='coerce',format='mixed')
    seconds=parsed.dt.hour*3600+parsed.dt.minute*60+parsed.dt.second
    return ds.dt.normalize()+pd.to_timedelta(seconds,unit='s')
def ci_mean(a):
    a=np.asarray(a,dtype=float); n=len(a); m=a.mean(); se=a.std(ddof=1)/np.sqrt(n)
    delta=stats.t.ppf(.975,n-1)*se
    return m-delta,m+delta

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--csv'); ap.add_argument('--salida',default='resultados'); args=ap.parse_args()
    out=Path(args.salida); out.mkdir(parents=True,exist_ok=True)
    if args.csv: data=Path(args.csv).read_bytes(); origin=str(Path(args.csv).resolve())
    else:
        req=urllib.request.Request(URL,headers={'User-Agent':'Mozilla/5.0 investigacion-academica'})
        with urllib.request.urlopen(req,timeout=120) as r: data=r.read()
        origin=URL
    (out/'fuente_original.csv').write_bytes(data)
    sha=hashlib.sha256(data).hexdigest()
    df=pd.read_csv(io.BytesIO(data),low_memory=False)
    df.columns=[c.strip().lower() for c in df.columns]
    required=['fecha_ing','hora_ingre','fecha_atencion','hora_atencion','triage','ips']
    missing=set(required)-set(df.columns)
    if missing: raise ValueError(f'Columnas faltantes: {missing}; presentes: {list(df.columns)}')
    ingreso=date_time(df,'fecha_ing','hora_ingre'); atencion=date_time(df,'fecha_atencion','hora_atencion')
    d=df.copy(); d['ingreso_dt']=ingreso; d['atencion_dt']=atencion
    d['minutos']=(atencion-ingreso).dt.total_seconds()/60
    d['triage']=d['triage'].astype('string').str.strip().str.upper()
    d['ips']=d['ips'].astype('string').str.strip()
    missing_times=int(d['minutos'].isna().sum()); negative=int((d['minutos']<0).sum())
    d=d.loc[d['minutos'].notna() & (d['minutos']>=0) & d['triage'].notna() & (d['triage']!='') & d['ingreso_dt'].notna()].copy()
    # Segunda variable cuantitativa: llegadas por hora por IPS; evitar mezcla de sedes.
    d['hora_bloque']=d['ingreso_dt'].dt.floor('h')
    d['llegadas_hora_ips']=d.groupby(['ips','hora_bloque'],dropna=False)['minutos'].transform('size')
    if len(d)<200: raise ValueError(f'Solo {len(d)} registros validos; se requieren al menos 200')
    if d['minutos'].nunique()<2: raise ValueError('Tiempo sin variabilidad; no se puede inferir')
    d.to_csv(out/'datos_limpios_powerbi.csv',index=False,encoding='utf-8-sig')
    by=d.groupby('triage')['minutos'].agg(['count','median','mean']).sort_values('count',ascending=False)
    groups=list(by.index[:2]);
    if len(groups)<2 or by.loc[groups,'count'].min()<2: raise ValueError('No hay dos grupos de triage con suficientes datos')
    x=d.loc[d.triage==groups[0],'minutos'].to_numpy(); y=d.loc[d.triage==groups[1],'minutos'].to_numpy()
    # Shapiro sobre muestra aleatoria fija de maximo 5000, por limite de precision del p-valor.
    sample=d['minutos'].sample(n=min(5000,len(d)),random_state=20261003).to_numpy()
    w,p_sh=stats.shapiro(sample)
    # Dos contrastes predefinidos: Welch de medias y Mann-Whitney de distribuciones.
    welch=stats.ttest_ind(x,y,equal_var=False)
    sx=np.var(x,ddof=1)/len(x); sy=np.var(y,ddof=1)/len(y)
    dfw=(sx+sy)**2/(sx*sx/(len(x)-1)+sy*sy/(len(y)-1)) if sx+sy>0 else np.nan
    diff=x.mean()-y.mean(); half=stats.t.ppf(.975,dfw)*np.sqrt(sx+sy)
    mw=stats.mannwhitneyu(x,y,alternative='two-sided',method='asymptotic')
    # Los IC reportados corresponden a medias; no se asigna IC a U.
    m=d['minutos'].to_numpy(); arrivals=d['llegadas_hora_ips'].to_numpy()
    desc=d[['minutos','llegadas_hora_ips']].agg(['mean','median','std','min','max','skew']).T
    desc.to_csv(out/'descriptivos.csv')
    by.to_csv(out/'grupos_triage.csv')
    plt.figure(); plt.hist(m,bins=60); plt.xlabel('Minutos entre ingreso y atencion'); plt.ylabel('Frecuencia'); plt.tight_layout(); plt.savefig(out/'histograma.png',dpi=160); plt.close()
    plt.figure(); stats.probplot(sample,dist='norm',plot=plt); plt.tight_layout(); plt.savefig(out/'qq.png',dpi=160); plt.close()
    plt.figure(); plt.boxplot([x,y],tick_labels=groups,showfliers=False); plt.xlabel('Triage'); plt.ylabel('Minutos entre ingreso y atencion'); plt.tight_layout(); plt.savefig(out/'cajas.png',dpi=160); plt.close()
    ci_all=ci_mean(m); ci_x=ci_mean(x); ci_y=ci_mean(y)
    details={'origen':origin,'fecha_ejecucion':str(pd.Timestamp.now().date()),'sha256':sha,'registros_originales':len(df),'registros_validos':len(d),'sin_tiempos':missing_times,'tiempos_negativos':negative,'periodo_inicio':str(d.ingreso_dt.min()),'periodo_fin':str(d.ingreso_dt.max()),'grupos':groups,'n_grupos':[len(x),len(y)],'shapiro_n':len(sample),'shapiro_w':float(w),'shapiro_p':float(p_sh),'welch_t':float(welch.statistic),'welch_gl':float(dfw),'welch_p':float(welch.pvalue),'mann_u':float(mw.statistic),'mann_p':float(mw.pvalue),'ic_media_general':ci_all,'ic_medias':[ci_x,ci_y],'ic_diferencia':(diff-half,diff+half)}
    (out/'resultados.json').write_text(json.dumps(details,ensure_ascii=False,indent=2,default=float),encoding='utf-8')
    lines=[r'\section{Resultados generados por el análisis reproducible}',
           'Fuente descargada o suministrada: '+esc(origin)+r'\\',
           'Fecha de ejecución: '+esc(details['fecha_ejecucion'])+r'\\',
           'SHA-256 del CSV original: '+esc(sha)+r'\\',
           f'Registros originales: {len(df)}; registros válidos: {len(d)}; sin tiempo calculable: {missing_times}; tiempos negativos: {negative}.',
           'Período observado en los registros válidos: '+esc(details['periodo_inicio'])+' a '+esc(details['periodo_fin'])+'.',
           r'\subsection{Avance 1: exploración}',
           r'\begin{center}\begin{tabular}{lrrrrrr}\toprule Variable & Media & Mediana & DE & Mín. & Máx. & Asimetría\\\midrule']
    for col,label in [('minutos','Minutos'),('llegadas_hora_ips','Llegadas/hora/IPS')]:
        row=desc.loc[col]; lines.append(label+' & '+' & '.join(num(row[k]) for k in ['mean','median','std','min','max','skew'])+r'\\')
    lines += [r'\bottomrule\end{tabular}\end{center}',
              f'Shapiro--Wilk sobre muestra aleatoria de {len(sample)} observaciones: $W={num(w)}$, $p={fmtp(p_sh)}$. '+('Se rechaza la normalidad en esta muestra.' if p_sh<ALPHA else 'No se rechaza la normalidad en esta muestra.'),
              r'\begin{figure}[ht]\centering\includegraphics[width=.76\linewidth]{resultados/histograma.png}\caption{Distribución del tiempo entre ingreso y atención. Interpretar su forma junto con la tabla descriptiva.}\end{figure}',
              r'\begin{figure}[ht]\centering\includegraphics[width=.76\linewidth]{resultados/qq.png}\caption{Gráfico Q--Q de la muestra empleada en Shapiro--Wilk.}\end{figure}',
              r'\begin{figure}[ht]\centering\includegraphics[width=.76\linewidth]{resultados/cajas.png}\caption{Distribución del tiempo en los dos grupos de triage más frecuentes; valores extremos ocultos solo en la visualización.}\end{figure}',
              r'\subsection{Avance 2: inferencia}',
              f'Se compararon los dos niveles de triage más frecuentes: {esc(groups[0])} ($n={len(x)}$) y {esc(groups[1])} ($n={len(y)}$). La elección automática es por frecuencia y debe interpretarse clínicamente con cautela.',
              r'\begin{center}\begin{tabular}{lrrrr}\toprule Grupo & n & Media & Mediana & IC 95\% media\\\midrule',
              f'{esc(groups[0])} & {len(x)} & {num(x.mean())} & {num(np.median(x))} & [{num(ci_x[0])}, {num(ci_x[1])}]'+r'\\',
              f'{esc(groups[1])} & {len(y)} & {num(y.mean())} & {num(np.median(y))} & [{num(ci_y[0])}, {num(ci_y[1])}]'+r'\\',
              r'\bottomrule\end{tabular}\end{center}',
              f'IC 95\\% de la media global: [{num(ci_all[0])}, {num(ci_all[1])}] minutos.',
              r'\begin{center}\begin{tabular}{lrrrrl}\toprule Prueba & Estadístico & gl & $p$ & IC 95\% diferencia de medias & Decisión ($\alpha=0.05$)\\\midrule',
              f'Welch & {num(welch.statistic)} & {num(dfw)} & {fmtp(welch.pvalue)} & [{num(diff-half)}, {num(diff+half)}] & '+('Rechazar $H_0$' if welch.pvalue<ALPHA else 'No rechazar $H_0$')+r'\\',
              f'Mann--Whitney & {num(mw.statistic)} & No aplica & {fmtp(mw.pvalue)} & No aplica & '+('Rechazar $H_0$' if mw.pvalue<ALPHA else 'No rechazar $H_0$')+r'\\',
              r'\bottomrule\end{tabular}\end{center}',
              'Welch contrasta igualdad de medias; Mann--Whitney contrasta igualdad de distribuciones. Son dos pruebas sobre los mismos grupos, por lo que no constituyen evidencia independiente. El intervalo de diferencia corresponde únicamente a Welch.',
              r'\subsection{Discusión parcial}',
              f'En los registros analizados, el promedio del intervalo entre ingreso y atención fue {num(m.mean())} minutos, con IC del 95\\% [{num(ci_all[0])}, {num(ci_all[1])}]. '+('La diferencia de medias entre los dos grupos analizados fue estadísticamente significativa según Welch.' if welch.pvalue<ALPHA else 'No se observó evidencia suficiente de diferencia de medias entre los dos grupos analizados según Welch.')+' La magnitud y el sentido de la diferencia deben leerse en la tabla, sin inferir causalidad.',
              ('La prueba de Mann--Whitney detectó diferencias entre las distribuciones.' if mw.pvalue<ALPHA else 'La prueba de Mann--Whitney no detectó evidencia suficiente de diferencias entre las distribuciones.')+' El conjunto no registra dotación de personal ni resultados clínicos; por tanto, no permite atribuir diferencias a recursos o calidad asistencial. El período y los posibles registros incompletos restringen la interpretación.']
    (out/'resultados.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Listo:',out/'resultados.tex','; SHA256:',sha,'; registros validos:',len(d))
if __name__=='__main__': main()
