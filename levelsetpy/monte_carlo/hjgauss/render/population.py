"""Render saved full-residual bird-model slices and the fresh query population."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.ndimage import label
from skimage.measure import euler_number
from hjgauss.render.figures import save,plane,native_identity
from hjgauss.validation.report import load,key,interpolate,population_evidence


def topology(v):
    mask=v<=0
    _,b0=label(mask,np.array([[0,1,0],[1,1,1],[0,1,0]]))
    chi=int(euler_number(mask,connectivity=1))
    return int(b0),int(b0-chi),chi


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('results/multidimensional'))
    p.add_argument('--figure-dir',type=Path,required=True)
    p.add_argument('--n',type=int,default=192);p.add_argument('--steps',type=int,default=800)
    p.add_argument('--method',choices=['etd2','fd'],default='etd2')
    a=p.parse_args()
    population=population_evidence(a.root,dict(n=a.n,steps=a.steps,method=a.method))
    if not population.get('passed'):raise RuntimeError('Population provenance has not passed: '+str(population))
    full=load(a.root,key('murmuration',a.n,steps=a.steps,method=a.method))
    ref=load(a.root,key('murmuration',a.n,'reference',steps=a.steps,method=a.method))
    times=np.linspace(0,2,11);planes=[];reference=[];counts=[];rcounts=[]
    for t in times:
        X,Y,V=plane(full,0,3,n=193,field=full['data'][f't_{t:g}'])
        _,_,R=plane(ref,0,3,n=193,field=ref['data'][f't_{t:g}'])
        planes.append(V);reference.append(R);counts.append(topology(V));rcounts.append(topology(R))
    counts=np.asarray(counts);rcounts=np.asarray(rcounts)
    centers=np.c_[1.15*np.cos(np.arange(7)*2*np.pi/7),1.15*np.sin(np.arange(7)*2*np.pi/7)]
    fig=plt.figure(figsize=(8,5.8),layout='constrained');grid=fig.add_gridspec(2,4,width_ratios=[1,1,1,.12])
    for j,i in enumerate([0,2,4,6,8,10]):
        ax=fig.add_subplot(grid[j//3,j%3]);V=planes[i]
        im=ax.contourf(X,Y,V,levels=np.linspace(-1,1,21),cmap='RdBu_r',extend='both')
        if np.min(V)<0<np.max(V):ax.contour(X,Y,V,levels=[0],colors='k',linewidths=1)
        for c in centers:ax.add_patch(Circle(c,.6,fill=False,edgecolor='.25',linestyle=':',linewidth=.8))
        ax.set_aspect('equal');ax.set_xlim(-3,3);ax.set_ylim(-3,3)
        ax.set_xticks([-3,0,3]);ax.set_yticks([-3,0,3]);ax.tick_params(labelbottom=j>=3,labelleft=j%3==0)
        ax.set_title(rf'$t={times[i]:.1f}\,\mathrm{{s}}$',fontsize=13,pad=7)
    cb=fig.colorbar(im,cax=fig.add_subplot(grid[:,-1]),ticks=[-1,-.5,0,.5,1])
    cb.ax.set_title('Value',fontsize=12,pad=20)
    fig.supxlabel(r'Rel. pos. $x_1$ (m)',fontsize=14)
    fig.supylabel(r'Rel. pos. $x_2$ (m)',fontsize=14)
    fig.suptitle('Attacked-agent model: fixed-horizon values',fontsize=15)
    provenance=dict(full=native_identity(full),reference=native_identity(ref),heading=0,
                    slice_grid=[193,193],window=[[-3,3],[-3,3]])
    save(fig,a.figure_dir/'murmur_brt_panels.jpg',provenance={**provenance,'displayed_times':[0,.4,.8,1.2,1.6,2.]})
    fig=plt.figure(figsize=(8,3),layout='constrained');grid=fig.add_gridspec(2,3,height_ratios=[.22,1])
    legend_ax=fig.add_subplot(grid[0,:]);legend_ax.axis('off');handles=None
    for j,title in enumerate([r'Components $\beta_0$',r'Holes $\beta_1$',r'Euler characteristic $\chi$']):
        ax=fig.add_subplot(grid[1,j]);h1,=ax.plot(times,counts[:,j],'-o',color='k',markersize=4,label='Full residual')
        h2,=ax.plot(times,rcounts[:,j],'--x',color='.45',markersize=5,label='Independent reference')
        ax.set_title(title,fontsize=13);ax.set_xticks([0,1,2]);ax.grid(alpha=.2)
        ax.set_yticks(np.unique(np.r_[counts[:,j],rcounts[:,j]]));handles=[h1,h2]
    legend_ax.legend(handles=handles,loc='center',ncol=2,frameon=False,fontsize=12)
    fig.supxlabel('Elapsed time (s)',fontsize=14)
    fig.supylabel('Count',fontsize=14)
    save(fig,a.figure_dir/'murmur_topology.jpg',provenance={**provenance,'times':times.tolist(),
         'fine_counts':counts.tolist(),'reference_counts':rcounts.tolist(),'connectivity':1})
    with np.load(a.root/'population_values_100000.npz') as f:states=f['states']
    index=np.random.default_rng(0).choice(len(states),2000,replace=False)
    fig,ax=plt.subplots(figsize=(6,5),layout='constrained')
    sc=ax.scatter(states[index,0],states[index,1],c=states[index,3],s=5,cmap='twilight',vmin=-np.pi,vmax=np.pi,alpha=.75)
    V=planes[-1]
    if np.min(V)<0<np.max(V):ax.contour(X,Y,V,levels=[0],colors='k',linewidths=1.2)
    ax.scatter(centers[:,0],centers[:,1],marker='x',c='k',s=45,linewidths=1.3)
    for c in centers:ax.add_patch(Circle(c,.6,fill=False,edgecolor='k',linestyle=':',linewidth=.9))
    ax.set_xlim(-3,3);ax.set_ylim(-3,3);ax.set_aspect('equal')
    ax.set_xlabel(r'Rel. pos. $x_1$ (m)');ax.set_ylabel(r'Rel. pos. $x_2$ (m)')
    cb=fig.colorbar(sc,ax=ax,fraction=.04,pad=.035,ticks=[-np.pi,0,np.pi])
    cb.ax.set_yticklabels([r'$-\pi$','0',r'$\pi$']);cb.set_label('Heading (rad)',fontsize=14)
    ax.set_title(f'{len(states):,} states; {len(index):,} displayed',fontsize=14,pad=12)
    save(fig,a.figure_dir/'murmur_phase_space.jpg',provenance={**provenance,
         'population_archive_sha256':population['archive_sha256'],'population_seed':population['seed'],
         'population_count':len(states),'display_seed':0,'display_count':len(index),'context_contour_time':2.})
    rec=dict(times=times.tolist(),fine_counts=counts.tolist(),reference_counts=rcounts.tolist(),
             count_order=['components','holes','Euler characteristic'],observation_window=[[-3,3],[-3,3]],
             heading=0,connectivity='4-neighbor foreground; Euler connectivity 1',
             full_run=full['key'],reference_run=ref['key'],sampling_grid=[193,193],
             full_archive_sha256=provenance['full']['archive_sha256'],
             reference_archive_sha256=provenance['reference']['archive_sha256'],
             displayed_population_count=len(index),full_population_count=len(states),display_seed=0,
             topology_matches_reference=bool(np.all(counts==rcounts)),
             interpretation='Topology of thresholded finite-window numerical slices; no invariant or safety certificate')
    (a.root/'topology_validation.json').write_text(json.dumps(rec,indent=2)+'\n')

if __name__=='__main__':main()
