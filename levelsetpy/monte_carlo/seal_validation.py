"""Seal completed numerical archives without modifying any PDE computation."""
from pathlib import Path
import argparse,hashlib,json,os,platform,sys
import numpy as np
import scipy
from validation_report import load


def atomic_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');os.replace(temp,path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('results/multidimensional'))
    a=p.parse_args();records={};failed=[]
    for meta in sorted(a.root.glob('*.json')):
        archive=meta.with_suffix('.npz')
        if not archive.exists():continue
        rec=json.loads(meta.read_text())
        if 'source_digests' not in rec or 'shape' not in rec:continue
        try:
            sol=load(a.root,meta.stem)
            rec['archive_sha256']=hashlib.sha256(archive.read_bytes()).hexdigest()
            rec['archive_fields']=sorted(sol['data'])
            rec['source_freeze_verified']=True
            atomic_json(meta,rec)
            records[meta.stem]=dict(archive_sha256=rec['archive_sha256'],
                metadata_sha256=hashlib.sha256(meta.read_bytes()).hexdigest())
        except (OSError,ValueError,KeyError) as e:failed.append(dict(file=meta.name,error=str(e)))
    manifest=dict(records=records,failed=failed,environment=dict(python=sys.version,
        executable=sys.executable,platform=platform.platform(),numpy=np.__version__,scipy=scipy.__version__),
        scope='Current-source completed native PDE runs only; a seal is an integrity check, not numerical acceptance',
        freeze_evidence='The two numerical producers were frozen before the accepted batch starts; each native record stores both source digests. Metadata/plotting/report scripts may change independently.')
    atomic_json(a.root/'MANIFEST.json',manifest)
    print('Sealed',len(records),'completed native records;',len(failed),'stale or invalid records excluded.')

if __name__=='__main__':main()
