#!/usr/bin/env python3
"""Recompute review tables and paired CIs from saved episode records."""
import hashlib
import json
from pathlib import Path
import platform
import sys
import numpy as np
import scipy
ROOT=Path(__file__).resolve().parents[1]

def interval(d):
    d=np.asarray(d)
    draws=np.random.default_rng(913).choice(d,(10000,len(d)),replace=True).mean(1)
    return dict(mean=float(d.mean()),ci95=np.quantile(draws,[.025,.975]).tolist(),n=len(d))

def main():
    folder=ROOT/'results'
    old=json.loads((folder/'compatible_comparison_30seeds.json').read_text())
    mip=json.loads((folder/'mip_comparison_30seeds.json').read_text())
    threshold=json.loads((folder/'mip_threshold_30seeds.json').read_text())
    repeated=json.loads((folder/'original_baselines_rerun.json').read_text())
    three=json.loads((folder/'mip_three_agents_30seeds.json').read_text())
    ccbs=json.loads((folder/'ccbs_30seeds/results.json').read_text())
    sha=hashlib.sha256((ROOT/'cache/compatible_brt.npz').read_bytes()).hexdigest()
    for d in [old,mip,threshold,repeated,three,ccbs]:
        assert d['table_sha256']==sha
    lookup={(r['seed'],r['policy']):r for r in old['records']}
    # Timing is machine/load dependent; count and feasibility diagnostics must reproduce.
    counts=['collisions','infeasible_agent_steps','unsatisfied_agent_steps','multi_threat_agent_steps','compatible_multi_threat_steps']
    for r in mip['records']+repeated['records']:
        if (r['seed'],r['policy']) in lookup:
            original=lookup[r['seed'],r['policy']]
            for k in counts:
                assert r[k]==original[k], (r['seed'],r['policy'],k,r[k],original[k])
    records=repeated['records']+mip['records']+threshold['records']
    policies=sorted(set(r['policy'] for r in records))
    summary={}
    ours={r['seed']:r for r in mip['records'] if r['policy']=='compatible'}
    for policy in policies:
        rows=sorted([r for r in records if r['policy']==policy],key=lambda r:r['seed'])
        assert [r['seed'] for r in rows]==list(range(30))
        summary[policy]=dict(contacts_mean=float(np.mean([r['collisions'] for r in rows])),
                             contact_free=sum(r['collisions']==0 for r in rows),
                             paired_difference=interval([ours[r['seed']]['collisions']-r['collisions'] for r in rows]))
    valid=[r for r in ccbs['records'] if r['solved']]
    cs=dict(solved=len(valid),attempted=len(ccbs['records']),
            failed_seeds=[r['seed'] for r in ccbs['records'] if not r['solved']],
            minimum_nominal_separation=min(r['nominal_min_separation'] for r in valid),
            mean_duration=float(np.mean([r['duration'] for r in valid])),arms={})
    for arm in ['unfiltered','chen_mip','simultaneous']:
        cs['arms'][arm]=dict(mean_contacts=float(np.mean([r['execution'][arm]['contacts'] for r in valid])),
                            contact_free=sum(r['execution'][arm]['contacts']==0 for r in valid),
                            mean_final_goal_error=float(np.mean([r['execution'][arm]['final_mean_goal_error'] for r in valid])),
                            paired_difference=interval([r['execution']['simultaneous']['contacts']-r['execution'][arm]['contacts'] for r in valid]))
    source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for base in ('src','experiments','tests') for p in (ROOT/base).glob('*.py')}
    output=dict(table_sha256=sha,original_count_diagnostics_reproduced=True,lifelong=summary,
                three_agents=three['paired_collision_differences'],ccbs=cs,
                environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform()),source_sha256=source_hashes)
    (folder/'external_baseline_summary.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k not in ('source_sha256',)},indent=2))
if __name__=='__main__':main()
