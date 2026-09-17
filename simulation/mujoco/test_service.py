"""Regression fixtures manipulate state only to test scoring, never benchmark policy."""
import json,sys,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'harness'))
import simsvc
from control import solve_step,rotation_error

class Regression(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.s=simsvc.Session(json.loads((ROOT/'layouts/put_bottles_into_dustbin_0.json').read_text()),'bottles')
  cls.initial=cls.s.data.qpos.copy()
 @classmethod
 def tearDownClass(cls):cls.s.renderer.close()
 def setUp(self):
  s=self.s;mujoco.mj_resetData(s.model,s.data);s.data.qpos[:]=self.initial;s.data.ctrl[s.ctrl_ids]=s.data.qpos[s.qpos_ids];mujoco.mj_forward(s.model,s.data)
  s.command_ctrl=s.data.ctrl[s.ctrl_ids].copy()
  s.grip_target_norm={'left':simsvc.grip_norm(s.data.qpos[s.qpos_ids[6]]),'right':simsvc.grip_norm(s.data.qpos[s.qpos_ids[13]])}
  s.commanded_grip=s.grip_target_norm.copy()
  s.t=0;s.done=False;s.success=False;s.score=0;s.error=None
 def test_scene_data_matches_rebuilt_model(self):
  s=self.s
  self.assertEqual(s.model.nq,len(s.data.qpos))
  for name,(pos,_) in s.obj_poses.items():np.testing.assert_allclose(s.data.xpos[s.model.body(name).id],pos,atol=1e-9)
  self.assertGreater(s.data.cam_xpos[s.cam_ids[0],2],1)
 def test_score_does_not_imply_success(self):
  self.s.score=100
  self.assertFalse(self.s._status()['success'])
 def test_timeout_even_with_closed_grippers(self):
  self.s.t=700;self.s.update_score()
  self.assertTrue(self.s.done);self.assertFalse(self.s.success)
  self.assertEqual(self.s._status()['termination'],'step_limit')
 def test_success_requires_return_home_and_open_grippers(self):
  s=self.s;binpos=s.data.xpos[s.bin_id].copy()
  for name in s.obj_body_ids:
   if name.startswith('bottle'):
    adr=s.model.jnt_qposadr[s.model.body(name).jntadr[0]];s.data.qpos[adr:adr+3]=binpos+[0,0,.2]
  s.data.qpos[s.qpos_ids[6]]=s.data.qpos[s.qpos_ids[13]]=.044
  s.data.qpos[s.qpos_ids[0]]=.3;mujoco.mj_forward(s.model,s.data);s.update_score()
  self.assertEqual(s.score,100);self.assertFalse(s.success)
  s.data.qpos[s.qpos_ids[0]]=0;mujoco.mj_forward(s.model,s.data);s.update_score();self.assertTrue(s.success)
 def test_bad_actions_do_not_advance(self):
  for rows in ([[0]*13],[[float('nan')]*14],[[0]*6+[2]+[0]*7]):
   with self.assertRaises(ValueError):self.s.act(rows)
  self.assertEqual(self.s.t,0)
 def test_gripper_interpolation_uses_metres_once(self):
  s=self.s;s.data.ctrl[s.ctrl_ids[6]]=.03
  s.command_ctrl=s.data.ctrl[s.ctrl_ids].copy()
  row=s.state14();row[6]=1;seen=[];original=mujoco.mj_step
  def step(m,d):seen.append(d.ctrl[s.ctrl_ids[6]]);original(m,d)
  with patch.object(simsvc.mujoco,'mj_step',step),patch.object(s,'capture'):
   s.act([row])
  # The normalized input is converted to metres once, then the physical
  # target is slew-limited and interpolated over the first 8 substeps.
  goal=.03+simsvc.CTRL_SMOOTH*simsvc.GRIP_CTRL_STEP
  self.assertAlmostEqual(seen[0],.03+(goal-.03)/simsvc.INTERP_SUBSTEPS)
  self.assertAlmostEqual(seen[-1],goal)

 def test_gripper_hysteresis_holds_noisy_commands(self):
  s=self.s; row=s.state14(); row[6]=0.0
  with patch.object(s,'capture'):
   s._step(row)
   self.assertEqual(s.commanded_grip['left'],0.0)
   row=s.state14(); row[6]=0.55  # policy noise in the deadband
   s._step(row)
   self.assertEqual(s.commanded_grip['left'],0.0)
   row=s.state14(); row[6]=0.95
   s._step(row)
   self.assertEqual(s.commanded_grip['left'],1.0)
 def test_eef_rejects_large_and_invalid_goals(self):
  goals=self.s.ee_poses();goals['left']['xyz'][0]+=.1
  with self.assertRaises(ValueError):solve_step(self.s.model,self.s.data,goals)
  goals=self.s.ee_poses();goals['left']['quat_wxyz']=[0]*4
  with self.assertRaises(ValueError):solve_step(self.s.model,self.s.data,goals)
 def test_eef_joint_and_cartesian_step_bounds(self):
  s=self.s;goals=s.ee_poses();goals['left']['xyz'][2]+=.02;goals['right']['xyz'][2]+=.01
  row=np.asarray(solve_step(s.model,s.data,goals));state=np.asarray(s.state14())
  self.assertLessEqual(np.max(np.abs(row-state)),.350001)
  d=mujoco.MjData(s.model);d.qpos[:]=s.data.qpos
  for col,qi in enumerate(s.qpos_ids):
   if col not in (6,13):d.qpos[qi]=row[col]
  mujoco.mj_forward(s.model,d)
  for side in ('left','right'):
   site=s.model.site(side+'_ee').id;self.assertLessEqual(np.linalg.norm(d.site_xpos[site]-s.data.site_xpos[site]),.050001)
 def test_classify_preserves_layout_pose_and_real_meshes(self):
  layout=json.loads((ROOT/'layouts/classify_objects_0.json').read_text())
  s=simsvc.Session(layout,'classify','classify_objects')
  try:
   for kind,items in layout['Rigid'].items():
    for entry in items:
     name=entry['label'];body=s.model.body(name)
     np.testing.assert_allclose(s.data.xpos[body.id],entry['default_pos'],atol=1e-8)
     np.testing.assert_allclose(s.data.xquat[body.id],entry['default_ori'],atol=1e-6)
     self.assertGreaterEqual(s.model.mesh('mesh_'+name).id,0)
   for entry in layout['Geometry']['basket']:
    np.testing.assert_allclose(s.data.xpos[s.model.body(entry['label']).id],entry['default_pos'],atol=1e-8)
  finally:s.renderer.close()

 def test_rotation_error_world_frame(self):
  q=np.array([np.cos(.05),0,0,np.sin(.05)]);mat=np.zeros(9);mujoco.mju_quat2Mat(mat,q)
  np.testing.assert_allclose(rotation_error(mat.reshape(3,3),np.eye(3)),[0,0,.1],atol=1e-8)

if __name__=='__main__':unittest.main(verbosity=2)
