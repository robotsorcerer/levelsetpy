"""Run reproducible multidimensional refinement and sensitivity batches.

No gate is relaxed after observing failures. Each stage writes native fields
and a durable failure record. Re-run a stage to resume matching cached runs.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import hashlib
import json
import traceback
import os
import platform
import sys
import uuid
from datetime import datetime, timezone

CASES=['double_integrator','dubins_single','dubins_relative','rockets','murmuration']
_WORKER_BASELINE = None


def _digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def _atomic_json(path,value):
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    os.replace(temporary,path)


def job(task):
    global _WORKER_BASELINE
    from hjgauss.validation.cases import run_game
    from hjgauss.validation.report import load
    out,name,n,steps,kind,length,base,angle,*scheme=task
    method=scheme[0] if scheme else 'etd2'
    if method not in {'etd2','fd'}:raise ValueError(method)
    if method=='fd':
        from hjgauss.validation.fd import run_game
    key=f'{name}_{kind}_{n}_{steps}_L{length:g}_c{base:g}'
    if angle: key+=f'_A{angle}'
    if method=='fd':key+='_fd'
    path=out/(key+'.json')
    here=Path(__file__).resolve().parents[1]
    sources=['core/spectral.py','validation/cases.py']
    if method=='fd':sources+=['core/finite_diff.py','validation/fd.py']
    digests={f:hashlib.sha256((here/f).read_bytes()).hexdigest() for f in sources}
    if _WORKER_BASELINE is None:_WORKER_BASELINE={}
    if method not in _WORKER_BASELINE:_WORKER_BASELINE[method]=digests
    if digests!=_WORKER_BASELINE[method]:
        attempt_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:12]
        attempt=out/'attempts'/attempt_id;attempt.mkdir(parents=True,exist_ok=False)
        rec=dict(key=key,status='failed',error='Numerical sources changed after worker initialization',
                 source_digests=digests,attempt_id=attempt_id)
        _atomic_json(attempt/'record.json',rec);_atomic_json(out/(key+'_failure.json'),rec)
        return rec
    if path.exists() and (out/(key+'.npz')).exists():
        try:
            native=load(out,key);rec=native['record']
            expected_shape=(n,n) if name=='double_integrator' else (n,n,angle or max(16,n//2))
            matches=(rec.get('source_digests')==digests and rec.get('case')==name
                     and rec.get('kind')==kind and rec.get('steps')==steps
                     and tuple(rec['shape'])==expected_shape
                     and rec.get('coefficient_base')==base
                     and rec.get('bounds',[[]])[0]==[-length,length])
            del native
            if matches:
                if not rec.get('archive_sha256'):
                    rec['archive_sha256']=_digest(out/(key+'.npz'))
                    _atomic_json(path,rec)
                return dict(key=key,status='cached',archive_sha256=rec['archive_sha256'])
        except Exception:
            # An unreadable or mismatched archive cannot be accepted as evidence.
            pass
    attempt_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:12]
    attempt=out/'attempts'/attempt_id
    attempt.mkdir(parents=True,exist_ok=False)
    try:
        sol,rec=run_game(attempt,name,n,steps,kind,length,base,angle)
        del sol
        final_digests={f:_digest(here/f) for f in digests}
        if final_digests!=digests or rec['source_digests']!=digests:
            raise RuntimeError('Numerical sources drifted during execution; results rejected')
        native=load(attempt,key);del native
        rec.update(archive_sha256=_digest(attempt/(key+'.npz')),attempt_id=attempt_id,
                   environment=dict(python=sys.version,executable=sys.executable,platform=platform.platform()),
                   source_freeze_verified=True)
        _atomic_json(attempt/'record.json',dict(key=key,status='completed',record=rec))
        # Publish the archive atomically, then its hash-bound metadata. A reader
        # during the short gap rejects the pair rather than accepting old data.
        os.replace(attempt/(key+'.npz'),out/(key+'.npz'))
        _atomic_json(path,rec)
        (attempt/(key+'.json')).unlink()
        return dict(key=key,status='completed',runtime=rec['runtime'],attempt_id=attempt_id)
    except Exception as exc:
        rec=dict(key=key,status='failed',error=str(exc),traceback=traceback.format_exc(),
                 source_digests=digests,attempt_id=attempt_id)
        out.mkdir(parents=True,exist_ok=True)
        _atomic_json(attempt/'record.json',rec)
        _atomic_json(out/(key+'_failure.json'),rec)
        for partial in attempt.glob('*.npz'):partial.unlink()
        return rec


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,default=Path('results/multidimensional'))
    p.add_argument('--stage',choices=['refinement','sensitivity'],required=True)
    p.add_argument('--cases',nargs='+',default=CASES)
    p.add_argument('--fine-n',type=int,default=128)
    p.add_argument('--workers',type=int,default=3)
    a=p.parse_args();tasks=[]
    for name in a.cases:
        if a.stage=='refinement':
            specs=[(64,800,'full',6.,.2,None),(96,800,'full',6.,.2,None),
                   (a.fine_n,800,'full',6.,.2,None),(a.fine_n,800,'reference',6.,.2,None)]
        else:
            specs=[(a.fine_n,1600,'full',6.,.2,None),
                   (a.fine_n,800,'full',6.,.35,None),
                   (a.fine_n*5//4,1000,'full',7.5,.2,a.fine_n//2)]
        tasks.extend((a.out,name,*spec) for spec in specs)
    tasks=list(dict.fromkeys(tasks));records=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(job,t) for t in tasks]
        for f in as_completed(futures):
            rec=f.result();records.append(rec);print(json.dumps(rec),flush=True)
            a.out.mkdir(parents=True,exist_ok=True)
            (a.out/(a.stage+'_jobs.json')).write_text(json.dumps(records,indent=2)+'\n')
    if any(rec['status']=='failed' for rec in records):raise SystemExit(1)

if __name__=='__main__':main()
