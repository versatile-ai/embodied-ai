# coding: utf-8
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from eval_control import Controller, public_state, CAMERAS

class InterfaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name);self.sid = 'abcdef012345'
        self.directory = self.root/'runs'/self.sid/'policy';self.directory.mkdir(parents=True)
        (self.directory/'manifest.json').write_text(json.dumps({'sim':'http://sim','pi':'http://pi','mode':'hybrid'}))
        self.c = Controller(self.sid,self.root,'http://sim','http://pi')
        self.obs = {'t':0,'done':False,'remaining_steps':700,'observation_id':self.sid+':0',
                    'ee':{side:{'xyz':[0,0,1], 'quat_wxyz':[1,0,0,0], 'gripper':1} for side in ('left','right')},
                    'state':[0]*14,'instruction':'task','object_positions':{'secret':[0,0,0]},'grasped':{'left':'secret'}}
    def test_no_privileged_fields(self):
        clean = public_state(self.obs)
        self.assertNotIn('object_positions',clean);self.assertNotIn('grasped',clean)
    def test_follow_id_and_fresh_step(self):
        p=self.directory/'proposal_000000.json'
        p.write_text(json.dumps({'observation_id':self.sid+':0','actions':[[0]*14]*50}))
        with patch.object(self.c,'session_http',side_effect=[self.obs, {'t':5}]) as http:
            self.c.follow(5,'accept')
            body=http.call_args_list[-1].args[1]
            self.assertEqual(body['expected_step'],0);self.assertTrue(body['request_id']);self.assertEqual(len(body['joints']),5)
        p.write_text(json.dumps({'observation_id':'another:0','actions':[[0]*14]*50}))
        with patch.object(self.c,'session_http',return_value=self.obs) as http:
            with self.assertRaisesRegex(ValueError,'Stale'):self.c.follow(5,'accept')
            self.assertEqual(http.call_count,1)
    def test_eef_preflight_and_single_step(self):
        good={'expected_step':0,'goals':copy.deepcopy(self.obs['ee']),'steps':1,'decision_summary':'adjust'}
        for bad in [dict(good,goals={'left':good['goals']['left']}),dict(good,steps=6),dict(good,expected_step=1)]:
            with patch.object(self.c,'session_http',return_value=self.obs) as http:
                with self.assertRaises(ValueError):self.c.eef(bad)
                self.assertEqual(http.call_count,1)
        with patch.object(self.c,'session_http',side_effect=[self.obs,{'t':1}]) as http:
            self.c.eef(good);self.assertEqual(http.call_args_list[-1].args[1]['steps'],1)
    def test_pi_camera_contract_and_clipping(self):
        raw={**self.obs,'images':{k:'encoded' for k in CAMERAS},'shapes':{k:[1,1,3] for k in CAMERAS}}
        arr=np.zeros((50,14));arr[:,6]=1.01
        with patch.object(self.c,'session_http',return_value=raw),patch.object(self.c,'http',return_value={'actions':arr.tolist(),'ms':3}) as http:
            out=self.c.infer();sent=http.call_args.args[2]
            self.assertEqual(set(sent),{'images','shapes','state','prompt'})
            self.assertEqual(set(sent['images']),set(CAMERAS.values()))
            self.assertEqual(out['gripper_clips'],50)
    def test_direct_mode_cannot_infer(self):
        self.c.mode='direct'
        with patch.object(self.c,'http') as http:
            with self.assertRaises(ValueError):self.c.infer()
            http.assert_not_called()
    def test_start_requires_reset_evidence(self):
        (self.root/'layouts').mkdir();(self.root/'layouts/put_bottles_into_dustbin_0.json').write_text('{}')
        record=self.root/'server_record';record.mkdir();(record/'reset_check.json').write_text('{"passed": false}')
        c=Controller(root=self.root,sim='http://sim',pi='http://pi')
        with patch.object(c,'http',side_effect=[{'ok':True,'version':'astra-isolated-v5'}, {'session_id':'000000000001','t':0,'record_dir':str(record)}]):
            with self.assertRaisesRegex(ValueError,'Reset'):c.start('put_bottles',0,'direct')
        self.assertFalse((self.root/'runs/000000000001/policy').exists())
    def test_budget_stops_motion_but_allows_finish(self):
        (self.directory/'decision_count').write_text('180')
        with patch.object(self.c,'session_http',return_value={'t':0,'done':True}) as http:
            self.c.execute('act',self.obs,{'joints':[[0]*14]},'over budget')
            self.assertEqual(http.call_args.args[0],'finish')
        self.assertEqual((self.directory/'decision_count').read_text(),'180')
    def test_follow_records_discard(self):
        p=self.directory/'proposal_000000.json'
        p.write_text(json.dumps({'observation_id':self.sid+':0','actions':[[0]*14]*50}))
        with patch.object(self.c,'session_http',side_effect=[self.obs,{'t':15}]):self.c.follow(15,'accept')
        events=[json.loads(x) for x in (self.directory/'trace.jsonl').read_text().splitlines()]
        event=next(x for x in events if x['type']=='chunk_discard')
        self.assertEqual(event['discarded_steps'],35);self.assertFalse(p.exists())
    def test_eef_tracks_five_steps(self):
        payload={'expected_step':0,'goals':self.obs['ee'],'steps':5,'decision_summary':'track fixed target'}
        with patch.object(self.c,'session_http',side_effect=[self.obs,{'t':5}]) as http:
            self.c.eef(payload);self.assertEqual(http.call_args.args[1]['steps'],5)
    def test_session_path_rejected(self):
        with self.assertRaises(ValueError):Controller('../old',self.root)

if __name__=='__main__':unittest.main(verbosity=2)
