"""Render same-viscosity comparisons from native full-residual run archives."""
from __future__ import annotations
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from hjgauss.validation.report import load,key,interpolate,assess

plt.rcParams.update({'font.family':'DejaVu Sans','font.weight':'bold',
    'axes.labelweight':'bold','axes.titleweight':'bold','figure.titleweight':'bold',
    'font.size':12,'axes.labelsize':14,'xtick.labelsize':11,'ytick.labelsize':11,
    'mathtext.fontset':'dejavusans','mathtext.default':'bf'})


def scientific(value, _position=None):
    """Compact scientific notation for plotted residuals and errors."""
    if value == 0:return '0'
    mantissa,exponent=f'{value:.1e}'.split('e')
    mantissa=mantissa.rstrip('0').rstrip('.')
    return f'{mantissa}e{int(exponent)}'


def native_identity(sol):
    rec=sol['record']
    archive=Path(sol['archive_path'])
    return dict(run=sol['key'],archive_sha256=rec.get('archive_sha256') or hashlib.sha256(archive.read_bytes()).hexdigest(),
                source_digests=rec['source_digests'],shape=rec['shape'],steps=rec['steps'],
                delta=rec['delta'],horizon=rec['horizon'],method=rec.get('method',rec['kind']))


def save(fig,path,placement_width=384.1122/72.27,provenance=None):
    """Save one raster, with font/placement provenance and live bounding boxes."""
    for text in fig.findobj(matplotlib.text.Text):text.set_fontweight('bold')
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer();width,height=fig.get_size_inches()
    roles=[]
    for text in fig.findobj(matplotlib.text.Text):
        if not text.get_visible() or not text.get_text():continue
        box=text.get_window_extent(renderer)
        # Tick-label objects that are not currently drawn may have a dummy box.
        if box.width<=0 or box.height<=0:continue
        prop=text.get_fontproperties()
        roles.append(dict(text=text.get_text(),source_pt=text.get_fontsize(),
                          effective_pt=text.get_fontsize()*placement_width/width,
                          font_file=matplotlib.font_manager.findfont(prop),
                          weight=prop.get_weight(),bbox_pixels=list(box.bounds)))
    metadata=dict(canvas_inches=[float(width),float(height)],dpi=300,bbox_dpi=fig.dpi,
                  placement_width_inches=placement_width,uniform_scaling=True,
                  font_family='DejaVu Sans',font_weight='bold',text_roles=roles,
                  display_unit_scales=dict(length_metres=1.,time_seconds=1.,velocity_metres_per_second=1.),
                  data_clipping='All data marks clipped by their axes; text is unclipped')
    if provenance is not None:metadata['data_provenance']=provenance
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=300,facecolor='white',pil_kwargs={'quality':98,'subsampling':0,'optimize':True})
    metadata.update(image_format='JPEG',jpeg_quality=98,chroma_subsampling='4:4:4',
                    figure_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (path.with_suffix('.json')).write_text(json.dumps(metadata,indent=2)+'\n')
    plt.close(fig)


def plane(sol,theta,extent,n=161,field=None):
    ax=np.linspace(-extent,extent,n);X,Y=np.meshgrid(ax,ax,indexing='xy')
    q=np.c_[X.ravel(),Y.ravel(),np.full(X.size,theta)]
    return X,Y,interpolate(sol,q,field).reshape(X.shape)


def game(full,reference,path,title,headings,extent):
    if any(full['record'][k]!=reference['record'][k] for k in ['case','delta','horizon','bounds','shape']):
        raise ValueError('Figure comparison settings differ')
    ncols=len(headings)
    fig=plt.figure(figsize=(8,6.2),layout='constrained')
    grid=fig.add_gridspec(3,ncols+1,width_ratios=[1]*ncols+[.12])
    data=[]
    for theta in headings:
        X,Y,R=plane(reference,theta,extent);_,_,V=plane(full,theta,extent)
        data.append((X,Y,R,V,np.abs(R-V)))
    value_max=max(1.,max(float(np.max(np.abs(a))) for row in data for a in row[2:4]))
    error_max=max(1e-6,max(float(np.max(row[4])) for row in data))
    axes=[]
    for row in range(3):
        for col,theta in enumerate(headings):
            ax=fig.add_subplot(grid[row,col]);axes.append(ax)
            X,Y,R,V,E=data[col];Z=[R,V,E][row]
            if row<2:
                im=ax.contourf(X,Y,Z,levels=np.linspace(-value_max,value_max,21),
                               cmap='RdBu_r',extend='both')
                if np.min(Z)<0<np.max(Z):ax.contour(X,Y,Z,levels=[0],colors='k',linewidths=1)
                if row==0:value_im=im
            else:
                error_im=ax.contourf(X,Y,Z,levels=np.linspace(0,error_max,17),
                                     cmap='magma',extend='neither')
            ax.set_aspect('equal');ax.set_xlim(-extent,extent);ax.set_ylim(-extent,extent)
            ax.tick_params(labelbottom=row==2,labelleft=col==0)
            ax.set_xticks([-extent,0,extent]);ax.set_yticks([-extent,0,extent])
            if row==0:ax.set_title(rf'$\theta={theta/np.pi:g}\pi$',fontsize=13,pad=8)
            if col==0:ax.set_ylabel(['Reference','Full residual','Abs. difference'][row],fontsize=13)
    fig.supxlabel(r'Rel. pos. $x_1$ (m)',fontsize=14)
    fig.supylabel(r'Rel. pos. $x_2$ (m)',fontsize=14)
    fig.suptitle(title,fontsize=15)
    cax=fig.add_subplot(grid[:2,-1]);cb=fig.colorbar(value_im,cax=cax,ticks=np.linspace(-value_max,value_max,5))
    # The title clears the extension tip. The error colorbar has no extension.
    cb.ax.set_title('Value',fontsize=12,pad=20)
    cax2=fig.add_subplot(grid[2,-1]);cb2=fig.colorbar(error_im,cax=cax2,
        ticks=np.linspace(0,error_max,4),format=FuncFormatter(scientific))
    cb2.ax.set_title('Abs. diff.',fontsize=11,pad=20)
    save(fig,path,provenance=dict(full=native_identity(full),reference=native_identity(reference),
                                 headings=list(map(float,headings)),observation_half_width=extent,slice_grid=[161,161]))


def double(full,path):
    fig,axs=plt.subplots(2,2,figsize=(7.2,5.8),layout='constrained')
    X,Y=np.meshgrid(*full['axes'],indexing='ij')
    for i,(ax,t) in enumerate(zip(axs.ravel(),[.2,.5,.8,1.])):
        V=full['data'][f't_{t:g}']
        im=ax.contourf(X,Y,V,levels=np.linspace(-1,1,21),cmap='RdBu_r',extend='both')
        if np.min(V)<0<np.max(V):ax.contour(X,Y,V,levels=[0],colors='k',linewidths=1)
        ax.set_xlim(-2,2);ax.set_ylim(-2,2);ax.set_aspect('equal')
        ax.set_title(rf'$t={t:.2f}\,\mathrm{{s}}$',fontsize=13,pad=7)
        ax.tick_params(labelbottom=i>=2)
    fig.supxlabel('Position (m)',fontsize=14)
    fig.supylabel('Velocity (m/s)',fontsize=14)
    cb=fig.colorbar(im,ax=axs.ravel().tolist(),fraction=.035,pad=.035,ticks=[-1,-.5,0,.5,1])
    cb.ax.set_title('Value',fontsize=12,pad=20)
    fig.suptitle('Double integrator: full residual',fontsize=15)
    save(fig,path,provenance=dict(full=native_identity(full),snapshot_times=[.2,.5,.8,1.]))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('results/multidimensional'))
    p.add_argument('--figure-dir',type=Path,required=True)
    p.add_argument('--cases',nargs='+',default=['rockets','dubins_relative','dubins_single','double_integrator'])
    p.add_argument('--n',type=int)
    p.add_argument('--selection',type=Path)
    p.add_argument('--diagnostic',action='store_true',help='Allow diagnostic rendering before all gates pass; use a separate output directory')
    args=p.parse_args()
    selection_path=args.selection or args.root/'selection.json'
    selection=json.loads(selection_path.read_text()) if selection_path.exists() else {}
    summary=assess(args.root,selection)
    for name in args.cases:
        if not args.diagnostic and not summary['cases'][name].get('passed'):
            raise RuntimeError(f'{name} has not passed all numerical gates')
        config=selection.get(name,dict(n=128,steps=800))
        n=args.n or config['n'];steps=config.get('steps',800);method=config.get('method','etd2')
        full=load(args.root,key(name,n,steps=steps,method=method))
        if name=='double_integrator':double(full,args.figure_dir/'double_integrator.jpg');continue
        reference=load(args.root,key(name,n,'reference',steps=steps,method=method))
        headings=[-np.pi/2,0,np.pi/2] if name!='dubins_single' else [-np.pi/2,0,np.pi/4,np.pi/2]
        titles={'rockets':'Rockets: fixed-horizon value comparison',
                'dubins_relative':'Relative Dubins game: fixed-horizon values',
                'dubins_single':'Single Dubins vehicle: pose-target values'}
        files={'rockets':'rockets_3d_slices.jpg','dubins_relative':'dubins_3d_comparison.jpg',
               'dubins_single':'dubins_single.jpg'}
        game(full,reference,args.figure_dir/files[name],titles[name],headings,3. if name=='rockets' else 2.)

if __name__=='__main__':main()
