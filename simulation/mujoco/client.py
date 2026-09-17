"""Conversation policy tools: start / observe / infer / follow / eef / finish.
Each execution binds a unique request_id to a fresh observed step.
"""
import argparse,base64,json,os,time,uuid,urllib.request,urllib.error
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parent
SIM=os.environ.get('ASTRA_SIM','http://127.0.0.1:8763')
PI=os.environ.get('ASTRA_PI05','http://127.0.0.1:8642')
HTTP_TIMEOUT=float(os.environ.get('ASTRA_HTTP_TIMEOUT','90'))
CAM={'cam_base':'cam_high','left_cam_wrist':'cam_left_wrist','right_cam_wrist':'cam_right_wrist'}
DIRECT_STEP=float(os.environ.get('ASTRA_DIRECT_STEP','0.018'))
def http(url,payload=None):
    req=urllib.request.Request(url,data=None if payload is None else json.dumps(payload,allow_nan=False).encode(),headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(req,timeout=HTTP_TIMEOUT) as r:return json.load(r)
    except urllib.error.HTTPError as e:raise RuntimeError(f'HTTP {e.code}: {e.read().decode()}') from e

def observe(sid):
    obs=http(f'{SIM}/session/{sid}/observe');dest=ROOT/'runs'/sid/'observations'/f"{obs['t']:06d}";dest.mkdir(parents=True,exist_ok=True)
    for k,b in obs['images'].items():Image.fromarray(np.frombuffer(base64.b64decode(b),np.uint8).reshape(obs['shapes'][k])).save(dest/(k+'.png'))
    (dest/'observation.json').write_text(json.dumps(obs))
    summary={k:v for k,v in obs.items() if k not in ('images','shapes')};summary['image_dir']=str(dest)
    (ROOT/'runs'/sid/'latest.json').write_text(json.dumps(summary,indent=2));return obs,summary

def infer(sid,obs):
    payload={'images':{v:obs['images'][k] for k,v in CAM.items()},'shapes':{v:obs['shapes'][k] for k,v in CAM.items()},'state':obs['state'],'prompt':obs['instruction']}
    begin=time.monotonic();out=http(PI+'/infer',payload);a=np.asarray(out['actions'],float)
    if a.shape==(1,50,14):a=a[0]
    if a.shape!=(50,14) or not np.isfinite(a).all():raise ValueError(f'Invalid proposal shape/content: {a.shape}')
    # Pi0.5 occasionally emits tiny normalized gripper overshoots.  The
    # simulator protocol is explicitly [0,1]; clamp only the two gripper
    # columns and preserve all arm joint values for replay/audit.
    a[:, [6, 13]] = np.clip(a[:, [6, 13]], 0.0, 1.0)
    out.update(actions=a.tolist(),observation_id=obs['observation_id'],wall_ms=(time.monotonic()-begin)*1000,instruction=obs['instruction'])
    path=ROOT/'runs'/sid/f"proposal_{obs['t']:06d}.json";path.write_text(json.dumps(out));return path,out

def execute(sid,step,route,payload,reason):
    if route == 'act' and 'joints' in payload:
        actions = np.asarray(payload['joints'], dtype=float).copy()
        if actions.ndim != 2 or actions.shape[1] != 14 or not np.isfinite(actions).all():
            raise ValueError('Expected finite joint rows of width 14')
        clipped = int(np.count_nonzero((actions[:, [6,13]] < 0) | (actions[:, [6,13]] > 1)))
        actions[:, [6,13]] = np.clip(actions[:, [6,13]], 0, 1)
        payload = {**payload, 'joints': actions.tolist()}
        reason += f' (gripper values clipped to [0,1]: {clipped})'
    body={'request_id':uuid.uuid4().hex,'expected_step':step,**payload,'decision_summary':reason}
    return http(f'{SIM}/session/{sid}/{route}',body)

def direct_move(sid, side, target_xyz, gripper, reason, tolerance=0.012):
    """Move one arm to a visual-policy waypoint using fresh observations.
    The other arm holds its measured pose; each request is <=4 cm and one step.
    """
    target=np.asarray(target_xyz,float)
    if side not in ('left','right') or target.shape!=(3,): raise ValueError('bad direct target')
    # Keep the wrist orientation fixed throughout a Cartesian move.  Reusing
    # the latest measured quaternion as the next target lets small IK errors
    # accumulate and can eventually steer the solver into a different branch.
    initial_obs,_=observe(sid)
    fixed_quat={s:list(initial_obs['ee'][s]['quat_wxyz']) for s in ('left','right')}
    for _ in range(160):
        obs,_=observe(sid)
        ee=obs['ee']
        cur=np.asarray(ee[side]['xyz'],float)
        delta=target-cur;dist=float(np.linalg.norm(delta))
        if dist<=tolerance:
            for _ in range(8):
                if abs(float(ee[side]['gripper'])-float(gripper))<=0.05: break
                goals={s:{'xyz':ee[s]['xyz'],'quat_wxyz':ee[s]['quat_wxyz'],
                          'gripper':float(gripper if s==side else ee[s]['gripper'])}
                        for s in ('left','right')}
                for s in ('left','right'): goals[s]['quat_wxyz']=fixed_quat[s]
                execute(sid,obs['t'],'eef',{'goals':goals,'steps':3},reason+' (gripper settle)')
                obs,_=observe(sid); ee=obs['ee']
            return obs
        nxt=cur+delta*min(1.0,DIRECT_STEP/max(dist,1e-9))
        goals={}
        for s in ('left','right'):
            goals[s]={'xyz':(nxt.tolist() if s==side else ee[s]['xyz']),
                      'quat_wxyz':fixed_quat[s],
                      'gripper':float(gripper if s==side else ee[s]['gripper'])}
        execute(sid,obs['t'],'eef',{'goals':goals,'steps':1},reason)
    raise RuntimeError('direct_move did not converge')

if __name__=='__main__':
    raise SystemExit(
        'Legacy diagnostic CLI retired: use python3 eval_control.py --help. '
        'client.py remains an internal library for environment tests; its raw observations are not policy inputs.'
    )
