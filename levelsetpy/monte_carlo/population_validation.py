"""Fresh seeded 100,000-state evaluation of the altitude-invariant bird model.

This evaluates one reduced relative-state value field. It does not simulate
a coupled flock trajectory or certify the signs of approximate values.
"""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
from validation_report import load,key,interpolate


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('results/multidimensional'))
 p.add_argument('--n',type=int,default=192);p.add_argument('--steps',type=int,default=800)
 p.add_argument('--seed',type=int,default=20261101);p.add_argument('--cohort-only',action='store_true')
 p.add_argument('--method',choices=['etd2','fd'],default='etd2')
 a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True)
 rng=np.random.default_rng(a.seed)
 states=np.column_stack([rng.uniform(-3,3,100000),rng.uniform(-3,3,100000),
                         rng.uniform(-1,1,100000),rng.uniform(-np.pi,np.pi,100000)])
 cohort=a.root/'population_states_100000.npz'
 np.savez_compressed(cohort,states=states,seed=np.array(a.seed))
 if a.cohort_only:
  print('Saved',states.shape,'uniformly distributed query states; no PDE values yet.');return
 full=load(a.root,key('murmuration',a.n,steps=a.steps,method=a.method))
 reference=load(a.root,key('murmuration',a.n,'reference',steps=a.steps,method=a.method))
 queries=states[:,[0,1,3]]
 start=time.perf_counter();values=interpolate(full,queries);elapsed=time.perf_counter()-start
 rvalues=interpolate(reference,queries)
 finite=np.isfinite(values)&np.isfinite(rvalues)
 np.savez_compressed(a.root/'population_values_100000.npz',states=states,values=values,
                     reference_values=rvalues,seed=np.array(a.seed))
 rec=dict(seed=a.seed,count=len(states),finite_count=int(np.sum(finite)),
   state_components=['relative x','relative y','altitude','relative heading'],
   distribution='Independent uniform x,y in [-3,3], altitude in [-1,1], heading in [-pi,pi)',
   positive_count=int(np.sum(values>0)),nonpositive_count=int(np.sum(values<=0)),
   reference_positive_count=int(np.sum(rvalues>0)),
   sign_disagreement_count=int(np.sum((values>0)!=(rvalues>0))),
   sampled_absolute_discrepancy=float(np.max(np.abs(values-rvalues))),
   query_seconds=elapsed,full_run=full['key'],reference_run=reference['key'],
   full_archive_sha256=hashlib.sha256((a.root/(full['key']+'.npz')).read_bytes()).hexdigest(),
   reference_archive_sha256=hashlib.sha256((a.root/(reference['key']+'.npz')).read_bytes()).hexdigest(),
   altitude_reduction='Equal-bound vertical controls cancel; altitude-independent payoff implies p_z=0',
   interpretation='One relative-state field evaluated at 100000 four-component states; no joint 400000-dimensional solve, flock trajectory simulation, or safety certificate',
   passed=len(states)==100000 and bool(np.all(finite)))
 rec['archive_sha256']=hashlib.sha256((a.root/'population_values_100000.npz').read_bytes()).hexdigest()
 (a.root/'population_100000.json').write_text(json.dumps(rec,indent=2)+'\n')
 print(json.dumps(rec,indent=2))

if __name__=='__main__':main()
