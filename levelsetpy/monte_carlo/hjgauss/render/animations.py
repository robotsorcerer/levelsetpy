#!/usr/bin/env python3
"""Render matched solver/reference/error GIFs from native saved value fields."""
import argparse, hashlib, io, json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.interpolate import RegularGridInterpolator

CASES = [
    ('murmuration', 'murmuration_full_192_800_L6_c0.2_fd', 'murmuration_reference_192_800_L6_c0.2_fd', 'roysoc_murmuration', 'Starlings validation', 3, ('Relative position x₁', 'Relative position x₂'), 0),
    ('double_integrator', 'double_integrator_full_128_800_L6_c0.2', 'double_integrator_reference_128_800_L6_c0.2', 'roysoc_double_integrator', 'Double Integrator', 2, ('Position', 'Velocity'), None),
    ('dubins_relative', 'dubins_relative_full_224_400_L6_c0.2', 'dubins_relative_reference_224_400_L6_c0.2', 'roysoc_dubins', 'Dubins', 2, ('Relative position x₁', 'Relative position x₂'), 0),
    ('rockets', 'rockets_full_160_400_L6_c0.2', 'rockets_reference_160_400_L6_c0.2', 'roysoc_rockets', 'Rockets', 3, ('Relative position x₁', 'Relative position x₂'), 0),
]

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(root, run):
    path=root/(run+'.npz'); rec=json.loads(path.with_suffix('.json').read_text())
    h=digest(path)
    if rec.get('archive_sha256') and rec['archive_sha256'] != h: raise ValueError('Archive digest mismatch: '+run)
    with np.load(path) as f: data={k:f[k] for k in f.files}
    return rec,data,h

def interpolate(data, field, x, y):
    axes=[data['axis_0'],data['axis_1']]
    q=np.c_[x.ravel(),y.ravel()]
    if 'axis_2' in data:
        axes.append(data['axis_2']); q=np.c_[q,np.zeros(len(q))]
        period=len(axes[-1])*(axes[-1][1]-axes[-1][0]);q[:,-1]=(q[:,-1]-axes[-1][0])%period+axes[-1][0]
        axes[-1]=np.r_[axes[-1],axes[-1][0]+period];field=np.concatenate([field,field[:,:,:1]],axis=-1)
    return RegularGridInterpolator(tuple(axes),field,bounds_error=True)(q).reshape(x.shape)

def render(root,out,case):
    name,fr,rr,stem,title,extent,labels,heading=case
    frec,fd,fh=load(root,fr); rrec,rd,rh=load(root,rr)
    for key in ('case','delta','horizon','bounds','shape','coefficient_base'):
        if frec.get(key)!=rrec.get(key): raise ValueError(f'{name}: unmatched {key}')
    times=sorted((float(k[2:]),k) for k in fd if k.startswith('t_'))
    rt={float(k[2:]):k for k in rd if k.startswith('t_')}
    if [t for t,k in times] != sorted(rt): raise ValueError(f'{name}: reference times differ')
    axis=np.linspace(-extent,extent,129);X,Y=np.meshgrid(axis,axis,indexing='xy');frames=[];maxerr=0
    fields=[]
    for t,k in times:
        a=interpolate(rd,rd[rt[t]],X,Y); b=interpolate(fd,fd[k],X,Y); E=np.abs(a-b);maxerr=max(maxerr,float(E.max()));fields.append((t,a,b,E))
    for t,a,b,E in fields:
        fig,axs=plt.subplots(1,3,figsize=(12,4.5),dpi=90,layout='constrained')
        vmax=max(1.,float(np.max(np.abs(np.r_[a.ravel(),b.ravel()]))));im=None
        for ax,z,ttl in zip(axs,[a,b,E],['LevelsetPy reference','Full residual approximation',f'Absolute error (max {maxerr:.2e})']):
            error=ttl.startswith('Absolute')
            im=ax.contourf(X,Y,z,levels=np.linspace(0,maxerr,25) if error else np.linspace(-vmax,vmax,25),cmap='magma' if error else 'RdBu_r',extend='max' if error else 'both')
            if not ttl.startswith('Absolute') and z.min()<0<z.max(): ax.contour(X,Y,z,[0],colors='black',linewidths=1)
            ax.set(xlim=(-extent,extent),ylim=(-extent,extent),aspect='equal',xlabel=labels[0],title=ttl)
            ax.set_ylabel(labels[1] if ax is axs[0] else '')
        heading_text = title if 'validation' in title.lower() else f'{title} validation'
        fig.suptitle(f'{heading_text} · Elapsed horizon {t:g}',fontsize=14)
        cb=fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.78,label='Absolute error')
        cb.set_ticks(np.linspace(0,maxerr,5))
        cb.set_ticklabels([f'{v:.1e}' for v in np.linspace(0,maxerr,5)])
        if heading is not None: fig.text(.5,.005,f'Heading = {heading} rad · matched saved snapshots',ha='center',fontsize=8)
        with io.BytesIO() as bfr:
            fig.savefig(bfr,format='png');bfr.seek(0);frames.append(Image.open(bfr).convert('P',palette=Image.Palette.ADAPTIVE,colors=192).copy())
        plt.close(fig)
    gif=out/(stem+'.gif');frames[0].save(gif,save_all=True,append_images=frames[1:],duration=[400]*(len(frames)-1)+[1200],loop=0,optimize=False,disposal=2)
    frames[-1].convert('RGB').save(out/(stem+'_poster.png'))
    return dict(label=title,case=name,approximation_run=fr,reference_run=rr,approximation_sha256=fh,reference_sha256=rh,times=[t for t,k in times],max_absolute_difference=maxerr,error_scale_max=maxerr,gif_sha256=digest(gif))

def main():
    default_source=Path(__file__).resolve().parents[2]/'results/multidimensional'
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=default_source);p.add_argument('--output',type=Path,required=True,help='Output animation directory');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    records=[render(a.source,a.output,c) for c in CASES]
    (a.output/'roysoc_animations.json').write_text(json.dumps(dict(animations=records,columns=['LevelsetPy reference','Full residual approximation','Absolute value error'],temporal_interpolation=False),indent=2)+'\n')
    print('Rendered four matched comparison animations.')
if __name__=='__main__':main()
