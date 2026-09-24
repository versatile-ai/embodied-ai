import numpy as np
import pytest
from .http_client import Pi05HTTPClient, CAMERAS

def test_identity_requires_explicit_opt_in():
    with pytest.raises(ValueError,match='attested'):Pi05HTTPClient('http://127.0.0.1:1')

@pytest.mark.parametrize('shape',[(50,14),(1,50,14)])
def test_wire_format_and_recording(tmp_path,monkeypatch,shape):
    calls=[]
    def request(self,path,payload=None):
        calls.append((path,payload))
        if path=='/health':return dict(ok=True,device='npu')
        actions=np.zeros(shape);actions[...,6]=1.5
        return dict(actions=actions.tolist(),ms=42)
    monkeypatch.setattr(Pi05HTTPClient,'_request',request)
    client=Pi05HTTPClient('http://localhost:8642',allow_unverified=True)
    obs={k:np.zeros((8,9,3),np.uint8) for k in CAMERAS}
    obs.update(states=np.zeros(14),instruction='test')
    actions,metadata=client.infer(obs,tmp_path/'a.npz')
    payload=calls[1][1]
    assert set(payload['images'])==set(CAMERAS)
    assert payload['shapes']['cam_high']==[8,9,3]
    assert actions.shape==(50,14) and np.all(actions[:,6]==1)
    assert np.all(np.load(tmp_path/'a.npz')['raw_actions'][:,6]==1.5)
    assert metadata['server_inference_ms']==42 and metadata['identity_verified'] is False
    assert client.metadata['checkpoint_sha256'] is None

def test_invalid_chunk_not_saved(tmp_path,monkeypatch):
    monkeypatch.setattr(Pi05HTTPClient,'_request',lambda self,path,payload=None: dict(ok=True,device='npu') if path=='/health' else dict(actions=[[0]*14]))
    client=Pi05HTTPClient('http://localhost:8642',allow_unverified=True)
    obs={k:np.zeros((2,2,3),np.uint8) for k in CAMERAS};obs.update(states=np.zeros(14),instruction='test')
    with pytest.raises(ValueError,match='50x14'):client.infer(obs,tmp_path/'bad.npz')
    assert not (tmp_path/'bad.npz').exists()
