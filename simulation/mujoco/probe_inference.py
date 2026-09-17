import json,urllib.request,time
from pathlib import Path
p=Path(__file__).resolve().parent/'runs/smoke_001'
o=json.loads((p/'obs_0005/raw.json').read_text())
keys={'cam_base':'cam_high','left_cam_wrist':'cam_left_wrist','right_cam_wrist':'cam_right_wrist'}
payload={'images':{v:o['images'][k] for k,v in keys.items()},'shapes':{v:o['shapes'][k] for k,v in keys.items()},'state':o['state'],'prompt':o['instruction']}
req=urllib.request.Request('http://127.0.0.1:8642/infer',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
t=time.monotonic()
try:
    with urllib.request.urlopen(req,timeout=45) as r:out=json.load(r)
except urllib.error.HTTPError as e:
    out={'http_error':e.code,'body':e.read().decode()}
(p/'proposal_deployed_0005.json').write_text(json.dumps(out));print({'seconds':time.monotonic()-t,'keys':list(out),'error':out.get('body'),'shape':__import__('numpy').asarray(out.get('actions',[])).shape})
