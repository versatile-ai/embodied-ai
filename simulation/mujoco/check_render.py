import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'harness'))
from simsvc import Session
from PIL import Image
import numpy as np
r=Path(__file__).resolve().parent
s=Session(json.loads((r/'layouts/put_bottles_into_dustbin_0.json').read_text()),'Put bottles in bin')
out=r/'runs/render_fixed';out.mkdir(exist_ok=True)
for name,cid in zip(('cam_base','left_cam_wrist','right_cam_wrist'),s.cam_ids):
 s.renderer.update_scene(s.data,cid);Image.fromarray(s.renderer.render()).save(out/(name+'.png'))
print('nbody',s.model.nbody,'nq',s.model.nq,'finite',np.isfinite(s.data.qpos).all())
s.renderer.close()
