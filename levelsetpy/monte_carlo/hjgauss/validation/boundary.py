"""Isolated box-extension checks in the original value equation.

Spatial spacing, heading count, time steps, physical Hamiltonian and the
baseline initial-condition completion are fixed. The reference equation has
no transformation coefficient, so the comparison isolates the numerical box.
"""
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import argparse,hashlib,json,os,uuid
import numpy as np
from hjgauss.core.spectral import solve as solve_nd
from hjgauss.validation.cases import game_case,diagnostics
from hjgauss.validation.report import load,key
from hjgauss.validation.run import _atomic_json,_digest


def run(task):
    out,name,config=task;n=config['n'];steps=config['steps'];method=config.get('method','etd2')
    baseline=key(name,n,'reference',steps=steps,method=method);name_key=baseline+'_boundary'
    here=Path(__file__).resolve().parents[1]
    sources=['core/spectral.py','validation/cases.py','validation/boundary.py']
    if method=='fd':sources.append('core/finite_diff.py')
    frozen={s:_digest(here/s) for s in sources}
    base=load(out,baseline);base_hash=_digest(out/(baseline+'.npz'));del base
    try:
        cache=load(out,name_key);rec=cache['record'];del cache
        if rec.get('source_digests')==frozen and rec.get('baseline_archive_sha256')==base_hash:
            return dict(key=name_key,status='cached')
    except (OSError,ValueError,KeyError):pass
    attempt_id='boundary_'+uuid.uuid4().hex
    attempt=out/'attempts'/attempt_id;attempt.mkdir(parents=True,exist_ok=False)
    try:
        # Keep the physical initial-condition function from the baseline box.
        case=game_case(name,6.)
        bounds=((-7.5,7.5),)*2+(((-np.pi,np.pi),) if len(case['bounds'])==3 else ())
        shape=(n*5//4,n*5//4)+((n//2,) if len(bounds)==3 else ())
        kwargs=dict(bounds=bounds,shape=shape,delta=case['delta'],horizon=case['horizon'],steps=steps)
        if method=='fd':
            from hjgauss.core.finite_diff import original_reference
            sol=original_reference(case['initial'],case['hamiltonian'],**kwargs)
        else:sol=solve_nd(case['initial'],case['hamiltonian'],kind='reference',**kwargs)
        rec=diagnostics(sol,case['interior'])
        rec.update(case=name,baseline_run=baseline,baseline_archive_sha256=base_hash,
                   initial_completion_half_width=6.,source_digests=frozen,
                   method='Isolated box extension of the original-equation reference',attempt_id=attempt_id)
        if any(_digest(here/s)!=d for s,d in frozen.items()):raise RuntimeError('Source drift during box check')
        np.savez_compressed(attempt/(name_key+'.npz'),value=sol['value'],
                            **{f'axis_{i}':a for i,a in enumerate(sol['axes'])})
        del sol
        rec['archive_sha256']=_digest(attempt/(name_key+'.npz'))
        _atomic_json(attempt/(name_key+'.json'),rec);checked=load(attempt,name_key);del checked
        _atomic_json(attempt/'record.json',dict(key=name_key,status='completed',record=rec))
        os.replace(attempt/(name_key+'.npz'),out/(name_key+'.npz'))
        os.replace(attempt/(name_key+'.json'),out/(name_key+'.json'))
        return dict(key=name_key,status='completed',runtime=rec['runtime'])
    except Exception as exc:
        rec=dict(key=name_key,status='failed',error=str(exc),attempt_id=attempt_id,source_digests=frozen)
        _atomic_json(attempt/'record.json',rec)
        for p in attempt.glob('*.npz'):p.unlink()
        return rec


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('results/multidimensional'))
    p.add_argument('--selection',type=Path);p.add_argument('--cases',nargs='+',required=True)
    p.add_argument('--workers',type=int,default=1);a=p.parse_args()
    selection=json.loads((a.selection or a.root/'selection.json').read_text());results=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(run,(a.root,name,selection[name])) for name in a.cases]
        for f in as_completed(futures):
            rec=f.result();results.append(rec);print(json.dumps(rec),flush=True)
            _atomic_json(a.root/('boundary_'+'_'.join(a.cases)+'_jobs.json'),results)
    if any(r['status']=='failed' for r in results):raise SystemExit(1)

if __name__=='__main__':main()
