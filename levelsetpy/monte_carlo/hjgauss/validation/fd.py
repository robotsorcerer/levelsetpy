"""Native-run producer for the full FD4 source solve and FD6 original reference."""
import argparse
from pathlib import Path
import hashlib,json
import numpy as np
from hjgauss.core.finite_diff import solve,original_reference
from hjgauss.core.spectral import variable_coefficient
from hjgauss.validation.cases import analytic_case,game_case,diagnostics

PACKAGE_ROOT=Path(__file__).resolve().parents[1]
SOURCES=['core/spectral.py','validation/cases.py','core/finite_diff.py','validation/fd.py']


def run_analytic(out):
    here=PACKAGE_ROOT;out=Path(out);out.mkdir(parents=True,exist_ok=True)
    digests={f:hashlib.sha256((here/f).read_bytes()).hexdigest() for f in SOURCES}
    records=[]
    for d in (2,3):
        case=analytic_case(d)
        for n,steps in [(16,40),(24,80),(32,160),(48,320)]:
            for kind,method in [('full',solve),('reference',original_reference)]:
                sol=method(case['initial'],case['hamiltonian'],bounds=case['bounds'],shape=(n,)*d,
                           delta=case['delta'],horizon=case['horizon'],steps=steps)
                coords=np.meshgrid(*sol['axes'],indexing='ij',sparse=True)
                rec=diagnostics(sol,case['interior'])
                rec.update(case=f'analytic_{d}d',method=sol['method'],source_digests=digests,
                           exact_sup_error=float(np.max(np.abs(sol['value']-case['exact'](.4,coords)[0]))))
                if any(hashlib.sha256((here/f).read_bytes()).hexdigest()!=h for f,h in digests.items()):
                    raise RuntimeError('Source drift during analytic suite')
                name=f'analytic_{d}d_{kind}_{n}_{steps}_fd'
                np.savez_compressed(out/(name+'.npz'),value=sol['value'],
                                    **{f'axis_{i}':a for i,a in enumerate(sol['axes'])})
                (out/(name+'.json')).write_text(json.dumps(rec,indent=2)+'\n')
                records.append(rec);print(d,n,kind,rec['exact_sup_error'],flush=True)
    return records


def run_game(out,name,n,steps,kind='full',length=6.,coefficient_base=.2,angle_n=None):
    here=PACKAGE_ROOT
    digests={f:hashlib.sha256((here/f).read_bytes()).hexdigest() for f in SOURCES}
    case=game_case(name,length)
    shape=(n,n) if name=='double_integrator' else (n,n,angle_n or max(16,n//2))
    kwargs=dict(bounds=case['bounds'],shape=shape,delta=case['delta'],horizon=case['horizon'],
                steps=steps,snapshots=np.linspace(0,case['horizon'],11))
    if kind=='full':
        sol=solve(case['initial'],case['hamiltonian'],
                  coefficient=variable_coefficient(case['bounds'],base=coefficient_base),**kwargs)
    elif kind=='reference':sol=original_reference(case['initial'],case['hamiltonian'],**kwargs)
    else:raise ValueError(kind)
    if any(hashlib.sha256((here/f).read_bytes()).hexdigest()!=h for f,h in digests.items()):
        raise RuntimeError('Numerical source drift during FD run')
    rec=diagnostics(sol,case['interior'])
    rec.update(case=name,coefficient_base=coefficient_base,method=sol['method'],
               source_digests=digests,numpy_version=np.__version__,
               interpretation='Terminal-horizon viscous Cauchy equation; all analytic residual terms retained')
    name_key=f'{name}_{kind}_{n}_{steps}_L{length:g}_c{coefficient_base:g}'
    if angle_n:name_key+=f'_A{angle_n}'
    name_key+='_fd';out=Path(out);out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/(name_key+'.npz'),value=sol['value'],
        **{f'axis_{i}':a for i,a in enumerate(sol['axes'])},
        **{f't_{t:g}':v for t,v in sol['snapshots'].items()})
    (out/(name_key+'.json')).write_text(json.dumps(rec,indent=2)+'\n')
    print(json.dumps(rec),flush=True)
    return sol,rec


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('results/multidimensional'))
    args=parser.parse_args()
    run_analytic(args.out)


if __name__=='__main__':main()
