import copy,json,uuid,urllib.request,urllib.error
from pathlib import Path
from client import http,SIM,observe,execute
root=Path(__file__).resolve().parent
r=http(SIM+'/session',{'layout':json.loads((root/'layouts/put_bottles_into_dustbin_0.json').read_text()),'instruction':'Pick up the bottles and throw them into the dustbin, using handover when needed.','task':'put_bottles'})
sid=r['session_id'];print('session',sid,flush=True)
o,_=observe(sid)
goals=copy.deepcopy(o['ee'])
for side in goals:goals[side]['xyz'][2]+=.02;goals[side]['gripper']=1
body={'request_id':uuid.uuid4().hex,'expected_step':o['t'],'goals':goals,'steps':5,'decision_summary':'Integration test: lift both end effectors 2 cm and open grippers.'}
r=http(f'{SIM}/session/{sid}/eef',body);print('first',r,flush=True)
r2=http(f'{SIM}/session/{sid}/eef',body);assert r2==r
obs,_=observe(sid);assert obs['t']==5
print('idempotent replay PASS',flush=True)
try:http(f'{SIM}/session/{sid}/eef',{**body,'request_id':uuid.uuid4().hex})
except RuntimeError as e:assert '409' in str(e);print('stale rejection PASS',flush=True)
else:raise AssertionError('stale accepted')
print('eef_measured',obs['ee'],flush=True)
row=obs['state'];row[6]=row[13]=1
print('hold',execute(sid,obs['t'],'act',{'joints':[row]*20},'Hold observed configuration to verify numerical stability'),flush=True)
obs,_=observe(sid)
print('final',obs['t'],obs['termination'],flush=True)
print('finish',execute(sid,obs['t'],'finish',{},'End integration test and assemble control-rate videos'),flush=True)
