"""Undo upstream unconditional +90 degree X bake in copied object meshes.
The inverse restores layout-compatible upright bottles / horizontal basket floors.
Never runs on robot STL meshes. A marker prevents double conversion.
"""
from pathlib import Path
import json,numpy as np
root=Path(__file__).resolve().parent/'assets/meshes'
marker=root/'axis_fix.json'
if marker.exists():raise SystemExit('Already corrected')
report={}
for p in root.glob('*.obj'):
    lines=p.read_text().splitlines();out=[];verts=[]
    for line in lines:
        if line.startswith('v '):
            x,y,z=map(float,line.split()[1:4]);v=[x,z,-y];verts.append(v);out.append('v '+' '.join(f'{n:.9f}' for n in v))
        else:out.append(line)
    if not verts:continue
    p.write_text('\n'.join(out)+'\n');a=np.asarray(verts)
    box={'min':a.min(axis=0).tolist(),'max':a.max(axis=0).tolist()}
    p.with_suffix('.bbox.json').write_text(json.dumps(box));report[p.name]=box
marker.write_text(json.dumps({'transform':'x,y,z -> x,z,-y (inverse of upstream converter bake)','basis':'upstream usdz2obj.py and layout floor-height consistency; original USD stages unavailable in local snapshot','meshes':report},indent=2))
