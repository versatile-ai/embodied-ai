"""Adapter for the existing Ascend HTTP service; never retries inference."""
import base64
import json
import time
import uuid
import urllib.request
import urllib.parse
import urllib.error
import numpy as np
from ..io import require

CAMERAS = dict(cam_high='cam_high', cam_left_wrist='cam_left_wrist', cam_right_wrist='cam_right_wrist')

class Pi05HTTPClient:
    def __init__(self, url, *, allow_unverified=False, timeout=600, camera_schema="robodojo"):
        require(allow_unverified, 'Legacy NPU service has no attested weights identity; explicitly enable --allow-unverified-student')
        parsed=urllib.parse.urlsplit(url)
        require(parsed.scheme in ('http','https') and parsed.hostname and not parsed.username and not parsed.password and not parsed.query and not parsed.fragment, 'Invalid service URL')
        require(camera_schema in ('robodojo','aloha'), 'Unknown camera schema')
        self.cameras=CAMERAS if camera_schema=='robodojo' else dict(zip(CAMERAS,('base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb')))
        self.url=url.rstrip('/'); self.timeout=timeout; self.index=0
        self.opener=urllib.request.build_opener(urllib.request.ProxyHandler({})) if parsed.hostname in ('localhost','127.0.0.1','::1') else urllib.request.build_opener()
        health=self._request('/health')
        require(health.get('ok') is True and health.get('device')=='npu', 'Healthy NPU service required')
        self.metadata=dict(backend='PyTorch/Ascend-HTTP',config='robodojo_pi05_service',checkpoint=None,
            checkpoint_sha256=None, action_horizon=50,action_dim=14,identity_verified=False,
            service_url=self.url,camera_schema=camera_schema,health=health, fresh_server_verified=False)
    def _request(self, path, payload=None):
        data=None if payload is None else json.dumps(payload,allow_nan=False).encode()
        request=urllib.request.Request(self.url+path,data=data,headers={'Content-Type':'application/json'})
        try:
            with self.opener.open(request,timeout=self.timeout) as response:
                result=json.load(response)
        except urllib.error.HTTPError as exc:
            detail=exc.read(4096).decode('utf-8',errors='replace')
            raise RuntimeError(f'NPU HTTP {exc.code}: {detail}') from exc
        require(isinstance(result,dict) and 'error' not in result, 'NPU service returned an error')
        return result
    def infer(self, observation, output):
        start=time.monotonic(); images={}; shapes={}
        for src,dst in self.cameras.items():
            array=np.asarray(observation[src])
            require(array.dtype==np.uint8 and array.ndim==3 and array.shape[2]==3,'Expected RGB uint8 HWC')
            images[dst]=base64.b64encode(array.tobytes()).decode(); shapes[dst]=list(array.shape)
        state=np.asarray(observation['states'],np.float32)
        require(state.shape==(14,) and np.isfinite(state).all(),'Expected finite state14')
        result=self._request('/infer',dict(images=images,shapes=shapes,state=state.tolist(),prompt=observation['instruction']))
        raw=np.asarray(result['actions'],np.float32)
        if raw.shape==(1,50,14): raw=raw[0]
        require(raw.shape==(50,14) and np.isfinite(raw).all(),'Expected finite 50x14 NPU action chunk')
        actions=raw.copy();actions[:,[6,13]]=np.clip(actions[:,[6,13]],0,1)
        np.savez_compressed(output,raw_actions=raw,actions=actions,**{k:observation[k] for k in (*CAMERAS,'states')})
        prediction=dict(prediction_id=uuid.uuid4().hex,inference_index=self.index,
            inference_seconds=time.monotonic()-start,server_inference_ms=result.get('ms'),identity_verified=False)
        self.index+=1
        return actions,prediction
    def close(self):
        pass
