"""Persistent file-based interface for an in-conversation policy; no LLM API calls."""
import base64, hashlib, json, sys, time, urllib.request
from pathlib import Path
import numpy as np
from PIL import Image
import mujoco
ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs'/'smoke_001'
RUN.mkdir(exist_ok=True)
SIM='http://127.0.0.1:8763'
def call(path,payload=None):
    req=urllib.request.Request(SIM+path,data=None if payload is None else json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=60) as r:return json.load(r)
def observe(sid):
    o=call('/session/'+sid+'/observe')
    d=RUN/f"obs_{o['t']:04d}";d.mkdir(exist_ok=True)
    for k,v in o['images'].items():
        a=np.frombuffer(base64.b64decode(v),np.uint8).reshape(o['shapes'][k]);Image.fromarray(a).save(d/(k+'.png'))
    (d/'raw.json').write_text(json.dumps(o))
    m=mujoco.MjModel.from_xml_path(str(ROOT/'assets/x5/dual_x5_scene.xml'));data=mujoco.MjData(m)
    for prefix,offset in [('left_',0),('right_',7)]:
        for j in range(6):data.qpos[int(m.joint(prefix+f'joint{j+1}').qposadr)]=o['state'][offset+j]
        data.qpos[int(m.joint(prefix+'joint7').qposadr)]=-.01+.054*o['state'][offset+6]
    mujoco.mj_forward(m,data)
    summary={k:v for k,v in o.items() if k not in ('images','shapes')}
    summary['ee']={p:{'xyz':data.site_xpos[m.site(p+'_ee').id].tolist(),'rotation':data.site_xmat[m.site(p+'_ee').id].reshape(3,3).tolist()} for p in ('left','right')}
    summary['image_dir']=str(d)
    (RUN/'latest.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
cmd=sys.argv[1]
if cmd=='start':
    if (RUN/'session.txt').exists(): raise RuntimeError('Existing evidence: choose a fresh RUN directory before starting')
    payload={'layout':json.loads((ROOT/'layouts/put_bottles_into_dustbin_0.json').read_text()),'instruction':'Pick up the bottles and throw them into the dustbin, using handover when needed.','task':'put_bottles'}
    sid=call('/session',payload)['session_id'];(RUN/'session.txt').write_text(sid);observe(sid)
elif cmd=='observe':observe((RUN/'session.txt').read_text())
elif cmd=='hold':
    sid=(RUN/'session.txt').read_text();o=json.loads((RUN/'latest.json').read_text());n=int(sys.argv[2]);assert 1<=n<=5
    decision={'observation_step':o['t'],'mode':'direct','action':'hold measured pose','steps':n,'reason':sys.argv[3]}
    (RUN/f"decision_{o['t']:04d}.json").write_text(json.dumps(decision,indent=2))
    ack=call('/session/'+sid+'/act',{'joints':[o['state']]*n});(RUN/f"ack_{o['t']:04d}.json").write_text(json.dumps(ack));print(ack);observe(sid)
elif cmd=='follow_one':
    sid=(RUN/'session.txt').read_text();o=json.loads((RUN/'latest.json').read_text());assert o['t']==5
    a=np.array(json.loads((RUN/'proposal_deployed_0005.json').read_text())['actions']);assert a.shape==(50,14) and np.isfinite(a).all()
    decision={'observation_step':5,'mode':'hybrid_smoke','action':'follow','steps':1,'reason':'Candidate first step changes arm joints by less than 0.009 rad and EEF positions by less than 2 mm. Accept one step to validate the proposal-execution-feedback path; occluded views prevent task-performance conclusions.'}
    (RUN/'decision_0005.json').write_text(json.dumps(decision,indent=2))
    ack=call('/session/'+sid+'/act',{'joints':a[:1].tolist()});(RUN/'ack_0005.json').write_text(json.dumps(ack));print(ack);observe(sid)
