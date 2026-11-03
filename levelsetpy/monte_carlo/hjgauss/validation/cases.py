"""Reproducible refinement loop for complete-residual multidimensional solves.

Each run is retained with its settings, equation residuals, timing and fields.
Acceptance is numerical and scoped to the specified equations and boxes.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from hjgauss.core.spectral import solve, variable_coefficient

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def manufactured(dimension, delta=.3):
    bounds=((-np.pi,np.pi),)*dimension
    speed=np.arange(1,dimension+1)*.15
    def exact(t,x):
        phase=[q-a*t for q,a in zip(x,speed)]
        s=[np.sin(q) for q in phase];cs=[np.cos(q) for q in phase]
        v=np.prod(np.broadcast_arrays(*s),axis=0)*np.exp(-.5*delta*dimension*t)
        grad=[]
        for i in range(dimension):
            g=cs[i]*np.exp(-.5*delta*dimension*t)
            for j in range(dimension):
                if i!=j:g=g*s[j]
            grad.append(g)
        return v,tuple(grad)
    def hamiltonian(t,x,p):
        _,pe=exact(t,x)
        return (sum(a*q for a,q in zip(speed,p))
                +.1*sum(q*q for q in p)-.1*sum(q*q for q in pe))
    return dict(initial=lambda x:exact(0,x)[0],hamiltonian=hamiltonian,
                exact=exact,bounds=bounds,delta=delta,horizon=.4,
                interior=((-np.pi,np.pi),)*dimension)


def diagnostics(sol, interior):
    coords=np.meshgrid(*sol['axes'],indexing='ij',sparse=True)
    mask=np.ones(sol['value'].shape,dtype=bool)
    for q,(a,b) in zip(coords,interior): mask &= (q>=a)&(q<=b)
    out={k:sol[k] for k in ['steps','dt','shape','bounds','delta','horizon','kind',
                           'min_transform','runtime']}
    out['value_range_interior']=[float(np.min(sol['value'][mask])),float(np.max(sol['value'][mask]))]
    out['defect_sample_times']=[r[0] for r in sol['defects']]
    for i,name in [(1,'original_defect'),(2,'transformed_defect'),(3,'chain_rule_gap')]:
        arrays=[r[i][mask] for r in sol['defects'] if r[i] is not None]
        if arrays:
            out[name+'_sup']=max(float(np.max(np.abs(a))) for a in arrays)
            out[name+'_rms']=max(float(np.sqrt(np.mean(a*a))) for a in arrays)
    return out


def compare(coarse,fine,interior):
    coords=np.meshgrid(*coarse['axes'],indexing='ij')
    mask=np.ones(coarse['value'].shape,dtype=bool)
    for q,(a,b) in zip(coords,interior):mask&=(q>=a)&(q<=b)
    pts=np.stack([q[mask] for q in coords],axis=-1)
    interp=RegularGridInterpolator(fine['axes'],fine['value'],bounds_error=True)
    err=coarse['value'][mask]-interp(pts)
    return dict(sup=float(np.max(np.abs(err))),rms=float(np.sqrt(np.mean(err*err))))


def store(out,name,sol,record):
    out.mkdir(parents=True,exist_ok=True)
    record['source_digests']={f:hashlib.sha256((PACKAGE_ROOT/f).read_bytes()).hexdigest()
                              for f in ['core/spectral.py','validation/cases.py']}
    record['numpy_version']=np.__version__
    np.savez_compressed(out/(name+'.npz'),value=sol['value'],
        **{f'axis_{i}':a for i,a in enumerate(sol['axes'])},
        **{f't_{t:g}':v for t,v in sol['snapshots'].items()})
    (out/(name+'.json')).write_text(json.dumps(record,indent=2)+'\n')


def game_case(name, length=6.):
    """Canonical elapsed-time Hamiltonians derived from physical controls.

    Target minimization uses H_elapsed=-min_u p.f; pursuit/evasion uses
    H_elapsed=-max_ue min_up p.f. These runs concern terminal reachability.
    """
    dim=2 if name=='double_integrator' else 3
    bounds=((-length,length),)*2+(((-np.pi,np.pi),) if dim==3 else ())
    radius=1.5 if name=='rockets' else .6 if name=='murmuration' else .5
    def initial(x):
        if name=='murmuration':
            angles=np.arange(7)*2*np.pi/7
            raw=np.minimum.reduce(np.broadcast_arrays(*[
                np.sqrt((x[0]-1.15*np.cos(a))**2+(x[1]-1.15*np.sin(a))**2+.08**2)
                -np.sqrt(radius**2+.08**2) for a in angles]))
        else:
            r2=x[0]**2+x[1]**2
            if name=='dubins_single':r2=r2+4*np.sin(x[2]/2)**2
            raw=np.sqrt(r2+.08**2)-np.sqrt(radius**2+.08**2)
        val=np.tanh(raw)
        # C-infinity completion to the constant far-field condition, without
        # changing the payoff in the declared physical observation region.
        cutoff=1.
        for q in x[:2]:
            s=np.clip((np.abs(q)-(length-1.5)),0,1)
            a=np.exp(-1/np.maximum(s,1e-15))
            b=np.exp(-1/np.maximum(1-s,1e-15))
            cutoff=cutoff*b/(a+b)
        return 1-(1-val)*cutoff
    def hamiltonian(t,x,p):
        if name=='double_integrator':return -x[1]*p[0]+np.abs(p[1])
        if name=='dubins_single':
            return -np.cos(x[2])*p[0]-np.sin(x[2])*p[1]+np.abs(p[2])
        if name=='rockets':
            return (-np.cos(x[2])*p[0]+(1-np.sin(x[2]))*p[1]
                    -np.abs(x[0]*p[0]-p[2])+np.abs(x[0]*p[1]+p[2]))
        return (p[0]*(1-np.cos(x[2]))-p[1]*np.sin(x[2])
                -np.abs(p[0]*x[1]-p[1]*x[0]-p[2])+np.abs(p[2]))
    horizon=.5 if name=='rockets' else 2. if name=='murmuration' else 1.
    extent=3. if name in {'rockets','murmuration'} else 2.
    return dict(initial=initial,hamiltonian=hamiltonian,bounds=bounds,
                delta=.18 if name=='murmuration' else .08,horizon=horizon,
                interior=((-extent,extent),)*2+(((-np.pi,np.pi),) if dim==3 else ()))


def run_game(out,name,n,steps,kind='full',length=6.,coefficient_base=.2,angle_n=None):
    case=game_case(name,length)
    shape=(n,n) if name=='double_integrator' else (n,n,angle_n or max(16,n//2))
    snapshots=np.linspace(0,case['horizon'],11)
    sol=solve(case['initial'],case['hamiltonian'],bounds=case['bounds'],shape=shape,
              delta=case['delta'],horizon=case['horizon'],steps=steps,kind=kind,
              coefficient=variable_coefficient(case['bounds'],base=coefficient_base),
              snapshots=snapshots)
    rec=diagnostics(sol,case['interior']);rec['case']=name
    rec['coefficient_base']=coefficient_base
    rec['interpretation']='terminal-horizon reachable set; no stopping/obstacle'
    key=f'{name}_{kind}_{n}_{steps}_L{length:g}_c{coefficient_base:g}'
    if angle_n: key+=f'_A{angle_n}'
    store(out,key,sol,rec)
    print(json.dumps(rec),flush=True)
    return sol,rec


def run_manufactured(out):
    rows=[]
    for d in (2,3):
        case=manufactured(d)
        for n,steps in [(16,40),(24,80),(32,160),(48,320)]:
            for kind in ['full','reference']:
                sol=solve(case['initial'],case['hamiltonian'],bounds=case['bounds'],
                          shape=(n,)*d,delta=case['delta'],horizon=.4,
                          steps=steps,kind=kind)
                rec=diagnostics(sol,case['interior'])
                coords=np.meshgrid(*sol['axes'],indexing='ij',sparse=True)
                truth=case['exact'](.4,coords)[0]
                rec['exact_sup_error']=float(np.max(np.abs(sol['value']-truth)))
                rec['case']=f'manufactured_{d}d'
                name=f'manufactured_{d}d_{kind}_{n}_{steps}'
                store(out,name,sol,rec);rows.append(rec)
                print(json.dumps(rec),flush=True)
    (out/'manufactured_summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=Path('results/multidimensional'))
    parser.add_argument('--suite',choices=['manufactured','games'],default='manufactured')
    parser.add_argument('--cases',nargs='+',default=['double_integrator','dubins_single','dubins_relative','rockets','murmuration'])
    parser.add_argument('--n',type=int,default=64)
    parser.add_argument('--steps',type=int,default=400)
    parser.add_argument('--kind',choices=['full','reference'],default='full')
    parser.add_argument('--length',type=float,default=6.)
    parser.add_argument('--coefficient-base',type=float,default=.2)
    parser.add_argument('--angle-n',type=int)
    args=parser.parse_args()
    if args.suite=='manufactured':run_manufactured(args.out)
    else:
        for name in args.cases:
            try:
                run_game(args.out,name,args.n,args.steps,args.kind,args.length,args.coefficient_base,args.angle_n)
            except (FloatingPointError,ValueError) as exc:
                record=dict(case=name,n=args.n,steps=args.steps,kind=args.kind,
                            length=args.length,error=str(exc),status='failed')
                args.out.mkdir(parents=True,exist_ok=True)
                key=f'{name}_{args.kind}_{args.n}_{args.steps}_L{args.length:g}_c{args.coefficient_base:g}'
                (args.out/(key+'_failure.json')).write_text(json.dumps(record,indent=2)+'\n')
                print(json.dumps(record),flush=True)

if __name__=='__main__': main()
