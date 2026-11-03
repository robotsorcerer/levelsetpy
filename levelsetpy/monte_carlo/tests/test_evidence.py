"""Regression checks for evidence binding and rejected-result propagation."""
import hashlib
import json
from pathlib import Path
import numpy as np
from hjgauss.validation import cases, run
from hjgauss.render import latex as paper_tex


def test_rejected_population_is_not_emitted_as_selected_run_numbers(tmp_path,monkeypatch):
    cases={name:dict(gates={},missing=['not yet computed']) for name in run.CASES}
    cases['rockets']['gates']={'defect':dict(value=.2,limit=.15,passed=False)}
    summary=dict(population=dict(passed=False,count=100000,positive_count=99828,
                                sign_disagreement_count=17),cases=cases,
                 analytic=[],passed=False)
    monkeypatch.setattr(paper_tex,'assess',lambda root,selection:summary)
    paper_tex.generate(tmp_path,tmp_path/'paper',{})
    out=tmp_path/'paper/sec_roysoc/figures'
    numbers=(out/'multidimensional_numbers.tex').read_text()
    assert r'\NDPopulationPositive}{\text{pending}}' in numbers
    assert '99{,}828' not in numbers
    table=(out/'multidimensional_table.tex').read_text()
    assert 'Status' in table
    assert next(line for line in table.splitlines() if line.startswith('Rockets')).endswith(r'Fail\\')


def fake_solver(counter):
    def solve(out,name,n,steps,kind,length,base,angle):
        counter.append(name);out.mkdir(parents=True,exist_ok=True)
        key=f'{name}_{kind}_{n}_{steps}_L{length:g}_c{base:g}'
        shape=(n,n);axis=np.linspace(-length,length,n,endpoint=False)
        np.savez_compressed(out/(key+'.npz'),value=np.full(shape,.125),axis_0=axis,axis_1=axis)
        here=Path(run.__file__).resolve().parents[1]
        rec=dict(case=name,kind=kind,steps=steps,shape=shape,bounds=[[-length,length]]*2,
                 coefficient_base=base,runtime=.01,
                 source_digests={f:hashlib.sha256((here/f).read_bytes()).hexdigest()
                   for f in ['core/spectral.py','validation/cases.py']})
        (out/(key+'.json')).write_text(json.dumps(rec))
        return None,rec
    return solve


def test_cache_rejects_tampered_archive_and_retains_attempt_receipts(tmp_path,monkeypatch):
    calls=[];monkeypatch.setattr(cases,'run_game',fake_solver(calls))
    monkeypatch.setattr(run,'_WORKER_BASELINE',None)
    task=(tmp_path,'double_integrator',8,20,'full',6.,.2,None)
    first=run.job(task)
    assert first['status']=='completed'
    assert run.job(task)['status']=='cached'
    assert len(calls)==1
    archive=tmp_path/(first['key']+'.npz')
    axis=np.linspace(-6,6,8,endpoint=False)
    np.savez_compressed(archive,value=np.full((8,8),.875),axis_0=axis,axis_1=axis)
    second=run.job(task)
    assert second['status']=='completed'
    assert len(calls)==2
    assert second['attempt_id']!=first['attempt_id']
    assert len(list((tmp_path/'attempts').glob('*/record.json')))==2
    rec=json.loads((tmp_path/(first['key']+'.json')).read_text())
    assert rec['archive_sha256']==hashlib.sha256(archive.read_bytes()).hexdigest()


def test_execution_source_drift_rejects_output_before_publication(tmp_path,monkeypatch):
    calls=[];monkeypatch.setattr(cases,'run_game',fake_solver(calls))
    monkeypatch.setattr(run,'_WORKER_BASELINE',None)
    real_digest=run._digest
    monkeypatch.setattr(run,'_digest',lambda p:'drift' if p.name=='spectral.py' else real_digest(p))
    result=run.job((tmp_path,'double_integrator',8,20,'full',6.,.2,None))
    assert result['status']=='failed'
    assert 'drift' in result['error']
    assert not (tmp_path/(result['key']+'.npz')).exists()
    assert not (tmp_path/(result['key']+'.json')).exists()
    assert list((tmp_path/'attempts').glob('*/record.json'))


def test_early_worker_source_drift_retains_failure_receipt(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'_WORKER_BASELINE',{'etd2':{'obsolete':'digest'}})
    result=run.job((tmp_path,'double_integrator',8,20,'full',6.,.2,None))
    assert result['status']=='failed'
    receipts=list((tmp_path/'attempts').glob('*/record.json'))
    assert len(receipts)==1
    assert json.loads(receipts[0].read_text())['attempt_id']==result['attempt_id']
