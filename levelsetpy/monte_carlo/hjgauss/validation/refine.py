"""Next refinement round; thresholds remain fixed in acceptance_criteria.json."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import argparse,json
from hjgauss.validation.run import job, _atomic_json

SELECTION={
 'double_integrator':dict(n=128,steps=800,medium=96,medium_steps=800),
 'dubins_single':dict(n=128,steps=800,medium=96,medium_steps=800),
 'rockets':dict(n=160,steps=400,medium=128,medium_steps=800),
 'dubins_relative':dict(n=224,steps=400,medium=192,medium_steps=400),
 'murmuration':dict(n=192,steps=800,medium=128,medium_steps=800,method='fd'),
}

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['refinement','sensitivity'],required=True)
 p.add_argument('--workers',type=int,default=3);p.add_argument('--cases',nargs='+',default=['rockets','dubins_relative','murmuration'])
 a=p.parse_args();out=Path('results/multidimensional');tasks=[]
 _atomic_json(out/'selection.json',SELECTION)
 for name in a.cases:
  n=SELECTION[name]['n'];steps=SELECTION[name]['steps'];method=SELECTION[name].get('method','etd2')
  specs=[(n,steps,'full',6.,.2,None),(n,steps,'reference',6.,.2,None)] if a.stage=='refinement' else [
   (n,steps*2,'full',6.,.2,None),(n,steps,'full',6.,.35,None),
   (n*5//4,steps*5//4,'full',7.5,.2,n//2)]
  if a.stage=='refinement' and method=='fd':
   specs.insert(0,(SELECTION[name]['medium'],SELECTION[name]['medium_steps'],'full',6.,.2,None))
  tasks.extend((out,name,*s,method) for s in specs)
 tasks=list(dict.fromkeys(tasks));records=[]
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  fs=[pool.submit(job,t) for t in tasks]
  for f in as_completed(fs):
   rec=f.result();records.append(rec);print(json.dumps(rec),flush=True)
   _atomic_json(out/(a.stage+'_'+'_'.join(a.cases)+'_jobs.json'),records)
 if any(r['status']=='failed' for r in records):raise SystemExit(1)

if __name__=='__main__':main()
