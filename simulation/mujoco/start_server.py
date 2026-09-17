"""Start the isolated local service and retain its PID/log across chat turns."""
import json,os,subprocess,sys,time,urllib.request
from pathlib import Path
root=Path(__file__).resolve().parent
try:
 with urllib.request.urlopen('http://127.0.0.1:8763/health',timeout=2) as response:
  health=json.load(response)
 if not str(health.get('version','')).startswith('astra-isolated-v'):
  raise RuntimeError('8763 occupied by another service')
 print('Already running: http://127.0.0.1:8763/live');sys.exit(0)
except OSError:pass
if not (root/'encode_video').exists():
 subprocess.run(['swiftc','-module-cache-path','/tmp/astra-swift-module-cache',str(root/'encode_video.swift'),'-o',str(root/'encode_video')],check=True)
with (root/'runs/server.log').open('ab') as log:
 proc=subprocess.Popen([sys.executable,'-u',str(root/'harness/simsvc.py')],cwd=root,env={**os.environ,'SIMPORT':'8763','ASTRA_RUN':str(root/'runs')},stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
(root/'runs/server.pid').write_text(str(proc.pid))
for _ in range(30):
 if proc.poll() is not None:raise RuntimeError('Server exited; inspect runs/server.log')
 try:
  with urllib.request.urlopen('http://127.0.0.1:8763/health',timeout=1) as response:
   print(response.read().decode());break
 except OSError:time.sleep(.2)
else:raise RuntimeError('Server startup timed out')
print('Live: http://127.0.0.1:8763/live')
