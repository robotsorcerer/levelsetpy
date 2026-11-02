"""Read native refinement records and apply the predeclared numerical gates."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from multidimensional_validation import game_case
from validation_loop import CASES


def key(name,n,kind='full',steps=800,length=6.,base=.2,angle=None,method='etd2'):
    k=f'{name}_{kind}_{n}_{steps}_L{length:g}_c{base:g}'
    return k+(f'_A{angle}' if angle else '')+('_fd' if method=='fd' else '')


def load(root,k,check_source=True):
    rec=json.loads((root/(k+'.json')).read_text())
    archive=root/(k+'.npz')
    if rec.get('archive_sha256') and hashlib.sha256(archive.read_bytes()).hexdigest()!=rec['archive_sha256']:
        raise ValueError(f'Archive digest mismatch: {k}')
    if check_source:
        for name,digest in rec['source_digests'].items():
            if hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()!=digest:
                raise ValueError(f'Stale source: {k}/{name}')
    with np.load(archive) as f:
        data={name:f[name] for name in f.files}
    axes=tuple(data[f'axis_{i}'] for i in range(data['value'].ndim))
    if data['value'].shape!=tuple(rec['shape']) or any(len(a)!=n or not np.all(np.diff(a)>0) for a,n in zip(axes,rec['shape'])):
        raise ValueError(f'Archive shape or axes mismatch: {k}')
    if not np.all(np.isfinite(data['value'])):raise ValueError(f'Nonfinite archive: {k}')
    return dict(record=rec,axes=axes,data=data,value=data['value'],key=k)


def interpolate(sol,queries,field=None):
    axes=list(sol['axes']);values=sol['value'] if field is None else field
    queries=np.asarray(queries).copy()
    if len(axes)==3:
        axis=axes[-1];period=axis.size*(axis[1]-axis[0])
        queries[:,-1]=(queries[:,-1]-axis[0])%period+axis[0]
        axes[-1]=np.r_[axis,axis[0]+period]
        values=np.concatenate([values,values[:,:,:1]],axis=-1)
    return RegularGridInterpolator(axes,values,bounds_error=True)(queries)


def sample_queries(name,n=65,angle_n=32):
    bounds=game_case(name)['interior']
    axes=[np.linspace(a,b,n) for a,b in bounds[:2]]
    if len(bounds)==3:axes.append(np.linspace(-np.pi,np.pi,angle_n,endpoint=False))
    return np.stack([v.ravel() for v in np.meshgrid(*axes,indexing='ij')],axis=-1)


def discrepancy(a,b,queries):
    av=interpolate(a,queries);bv=interpolate(b,queries);err=av-bv
    return dict(sup=float(np.max(np.abs(err))),rms=float(np.sqrt(np.mean(err**2))),
                sign_disagreement_fraction=float(np.mean((av<=0)!=(bv<=0))))


def population_evidence(root,config):
    """Verify the saved cohort, paired values, counts, seed and field bindings."""
    if isinstance(config,int):config=dict(n=config,steps=800)
    try:
        rec=json.loads((root/'population_100000.json').read_text());issues=[]
        if not rec.get('passed'):issues.append('Population producer did not pass')
        method=config.get('method','etd2')
        for label,kind in [('full','full'),('reference','reference')]:
            expected=key('murmuration',config['n'],kind,steps=config['steps'],method=method)
            path=root/(expected+'.npz')
            if rec.get(label+'_run')!=expected or not path.exists():
                issues.append('Population/native run mismatch: '+label)
            elif hashlib.sha256(path.read_bytes()).hexdigest()!=rec.get(label+'_archive_sha256'):
                issues.append('Native archive hash mismatch: '+label)
        archive=root/'population_values_100000.npz'
        if hashlib.sha256(archive.read_bytes()).hexdigest()!=rec.get('archive_sha256'):
            issues.append('Population archive hash mismatch')
        with np.load(archive) as f:
            states=f['states'];v=f['values'];r=f['reference_values'];seed=int(f['seed'])
        if states.shape!=(100000,4) or v.shape!=(100000,) or r.shape!=(100000,):
            raise ValueError('Population array shape mismatch')
        if not np.all(np.isfinite(states)) or not np.all(np.isfinite(v)) or not np.all(np.isfinite(r)):
            issues.append('Nonfinite population state or paired value')
        counts=dict(count=len(states),finite_count=int(np.sum(np.isfinite(v)&np.isfinite(r))),
                    positive_count=int(np.sum(v>0)),nonpositive_count=int(np.sum(v<=0)),
                    reference_positive_count=int(np.sum(r>0)),sign_disagreement_count=int(np.sum((v>0)!=(r>0))))
        if any(rec.get(k)!=n for k,n in counts.items()):issues.append('Population count metadata mismatch')
        rng=np.random.default_rng(seed)
        replay=np.column_stack([rng.uniform(-3,3,100000),rng.uniform(-3,3,100000),
                                rng.uniform(-1,1,100000),rng.uniform(-np.pi,np.pi,100000)])
        if seed!=rec.get('seed') or not np.array_equal(states,replay):issues.append('Population seed replay mismatch')
        if rec.get('sampled_absolute_discrepancy')!=float(np.max(np.abs(v-r))):
            issues.append('Population discrepancy metadata mismatch')
        rec['passed']=not issues
        if issues:rec['issues']=issues
        return rec
    except (OSError,ValueError,KeyError) as e:
        return dict(passed=False,missing=str(e))


def assess(root,selection=None):
    criteria=json.loads((root/'acceptance_criteria.json').read_text())
    results={}
    selection=selection or {}
    for name in CASES:
        config=selection.get(name,128)
        if isinstance(config,int):config=dict(n=config,steps=800,medium=96 if config==128 else config*3//4,medium_steps=800)
        n=config['n'];steps=config['steps'];method=config.get('method','etd2');q=sample_queries(name)
        row=dict(case=name,fine_n=n,query_count=len(q),gates={},missing=[])
        try:
            fine=load(root,key(name,n,steps=steps,method=method));r=fine['record']
        except (OSError,KeyError,ValueError) as e:
            row['missing'].append(str(e));results[name]=row;continue
        row['record']=r
        for metric,limit in [('original_defect_sup','game_original_defect_sampled_sup'),
                             ('original_defect_rms','game_original_defect_sampled_rms')]:
            row['gates'][metric]=dict(value=r[metric],limit=criteria[limit],passed=r[metric]<=criteria[limit])
        row['gates']['positivity']=dict(value=r['min_transform'],passed=r['min_transform']>0)
        comparisons=[('space',key(name,config['medium'],steps=config['medium_steps'],method=method),'game_space_refinement_sup'),
                     ('independent',key(name,n,'reference',steps=steps,method=method),'game_independent_reference_sup'),
                     ('time',key(name,n,steps=steps*2,method=method),'game_time_halving_sup'),
                     ('coefficient',key(name,n,steps=steps,base=.35,method=method),'game_coefficient_change_sup'),
                     ('domain',key(name,n*5//4,steps=steps*5//4,length=7.5,angle=n//2,method=method),'game_domain_extension_sup')]
        for label,k,threshold in comparisons:
            try:
                other=load(root,k);values=discrepancy(fine,other,q)
                row[label]=values
                row['gates'][label]=dict(value=values['sup'],limit=criteria[threshold],
                                          passed=values['sup']<=criteria[threshold])
            except (OSError,KeyError,ValueError) as e:row['missing'].append(str(e))
        try:
            baseline_key=key(name,n,'reference',steps=steps,method=method)
            baseline=load(root,baseline_key);extended=load(root,baseline_key+'_boundary')
            if extended['record'].get('baseline_archive_sha256')!=hashlib.sha256((root/(baseline_key+'.npz')).read_bytes()).hexdigest():
                raise ValueError('Stale isolated boundary record')
            values=discrepancy(baseline,extended,q);row['boundary']=values
            row['gates']['boundary']=dict(value=values['sup'],limit=criteria['game_domain_extension_sup'],
                                         passed=values['sup']<=criteria['game_domain_extension_sup'])
        except (OSError,KeyError,ValueError) as e:row['missing'].append(str(e))
        v=interpolate(fine,q)
        row['zero_sublevel_query_count']=int(np.sum(v<=0))
        # An empty reachable set is a scientific outcome, not automatically a
        # failed solver; disclose it instead of manufacturing a zero contour.
        row['has_zero_contour']=bool(np.min(v)<0<np.max(v))
        row['passed']=not row['missing'] and all(v['passed'] for v in row['gates'].values())
        results[name]=row
    manufactured=[]
    schemes=['etd2']+(['fd'] if any(isinstance(c,dict) and c.get('method')=='fd' for c in selection.values()) else [])
    for d in [2,3]:
      for scheme in schemes:
        for kind in ['full','reference']:
            try:
                rec=load(root,f'manufactured_{d}d_{kind}_48_320'+('_fd' if scheme=='fd' else ''))['record']
                manufactured.append(dict(dimension=d,kind=kind,scheme=scheme,error=rec['exact_sup_error'],
                    passed=rec['exact_sup_error']<=criteria['manufactured_exact_sup']))
            except (OSError,KeyError,ValueError) as e:
                manufactured.append(dict(dimension=d,kind=kind,scheme=scheme,passed=False,missing=str(e)))
    population=population_evidence(root,selection.get('murmuration',128))
    game_pass=all(r.get('passed',False) for r in results.values())
    summary=dict(criteria=criteria,cases=results,manufactured=manufactured,population=population,
                 game_checks_passed=game_pass,
                 passed=game_pass and all(r['passed'] for r in manufactured) and population.get('passed',False),
                 interpretation='Scoped numerical validation; sampled defects do not certify continuum error or safety')
    (root/'validation_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Multidimensional numerical validation','',summary['interpretation'],'',
           'Discrepancies are maxima on 4,225 common interpolated queries in 2D or 135,200 in 3D (65 × 65 × 32 headings). Field refinement can change both grid and time steps; fixed-grid time halving is reported separately. Equation defects are sampled at native collocation nodes at three interior times. The box check combines domain extension, a smaller time step, and a changed smooth coefficient wavelength; it is not an isolated boundary-error estimate.','',
           '| Case | Fine grid | Space | Independent | Time | Coefficient | Combined box | Sampled defect (sup) | Status |',
           '|---|---|---|---|---|---|---|---|---|']
    for name,r in results.items():
        vals=[f"{r.get(v,{}).get('sup',float('nan')):.3g}" for v in ['space','independent','time','coefficient','domain']]
        defect=r.get('record',{}).get('original_defect_sup',float('nan'))
        lines.append('| '+' | '.join([name,str(r['fine_n']),*vals,f'{defect:.3g}','pass' if r.get('passed') else 'pending or failed'])+' |')
    (root/'VALIDATION_REPORT.md').write_text('\n'.join(lines)+'\n')
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('results/multidimensional'))
    p.add_argument('--selection',type=Path)
    args=p.parse_args();selection=json.loads(args.selection.read_text()) if args.selection else None
    out=assess(args.root,selection)
    for name,r in out['cases'].items():
        print(name, {k:round(v['value'],6) for k,v in r['gates'].items()},
              'missing',len(r['missing']),'pass',r.get('passed',False))

if __name__=='__main__':main()
