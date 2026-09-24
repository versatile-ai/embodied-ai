import copy
import json
from pathlib import Path

import numpy as np
import pytest

from .io import InputError, write_json
from .robodojo_server.gpt_only_client import GPTOnlyTools, adaptive_stop_evidence, validate_action
from .robodojo_server.debug_recorder import DecisionTimeline
from .skill.run import tool_specs
from .skill.workspace import prepare_workspace
from .test_contract import Sim, POSE
from .test_case_ledger import attempt
from .test_evaluation import fake_panel_file
from .case_ledger import build_ledger, require_available


def invoke(tools, response=None):
    arguments=tools.next_call().copy();name=arguments.pop('tool')
    if response is not None:arguments['response']=response
    return (tools.start if name=='robodojo_start' else tools.act)(**arguments)


def action(request, mode='eef'):
    return dict(request_id=request['request_id'],mode=mode,steps=50,
        reason='Move toward the visible target.',
        target={arm:copy.deepcopy(POSE) for arm in ('left','right')})


@pytest.mark.parametrize('mode',['eef'])
def test_direct_loop_records_actions_without_pi05(tmp_path,mode):
    tools=GPTOnlyTools(tmp_path,'arrange_largest_number',rpc_factory=Sim,max_decisions=0)
    packet=invoke(tools)
    first=action(tools.request,mode)
    assert packet['next_call']['tool']=='robodojo_act'
    packet=invoke(tools,first)
    assert packet['step_id']==2 and tools.phase=='done'
    assert tools.next_call() is None
    assert packet['result']['complete'] is True
    assert tools.student is None and tools.counters['predictions']==0
    assert tools.counters['student_steps']==0
    assert not list(tmp_path.glob('proposal_*.npz'))
    assert 'fk_preview' not in tools.sim.calls
    assert sum(tools.counters[k] for k in ('gpt_joint_steps','gpt_eef_steps'))==2
    timeline=DecisionTimeline(tmp_path)
    assert len(timeline.segments)==1
    assert timeline.segments[0]['pi05_actions']==[]
    assert timeline.segments[0]['numeric_action_changed'] is None
    assert timeline.segments[0]['evaluation_method']=='gpt_only'


@pytest.mark.parametrize('mode',['joint','student','edit','stop'])
def test_forbidden_actions_reject_without_execution(tmp_path,mode):
    tools=GPTOnlyTools(tmp_path,'arrange_largest_number',rpc_factory=Sim)
    invoke(tools)
    with pytest.raises(InputError):invoke(tools,action(tools.request,mode))
    assert tools.sim.tick==0
    with pytest.raises(InputError):tools.infer()


def test_same_numeric_limits_and_bounded_eef():
    req=dict(request_id='r',current_eef={arm:copy.deepcopy(POSE) for arm in ('left','right')})
    r=action(req,'eef');r['steps']=51
    with pytest.raises(ValueError,match='exactly 50 maximum'):validate_action(r,req)
    r=action(req,'eef');r['steps']=49
    with pytest.raises(ValueError,match='exactly 50 maximum'):validate_action(r,req)
    r=action(req,'eef');r['steps']=50
    validate_action(r,req)
    r=action(req,'eef');r['target']['left']['position'][0]=0.051
    with pytest.raises(ValueError,match='5 cm'):validate_action(r,req)
    r=action(req);r['actions']=[[0.0]*14]
    with pytest.raises(ValueError,match='joint'):validate_action(r,req)


def test_adaptive_stop_requires_measured_target_and_stability():
    target={arm:copy.deepcopy(POSE) for arm in ('left','right')}
    diagnostics={arm:dict(target_position_error_m=.004,target_rotation_error_rad=.02)
                 for arm in ('left','right')}
    state=np.r_[np.zeros(6),1.,np.zeros(6),1.]
    trace=[dict(obs=dict(states=state.copy())) for _ in range(5)]
    evidence=adaptive_stop_evidence(trace,diagnostics,target)
    assert evidence['at_target'] and evidence['joints_stable'] and evidence['grippers_stable']
    diagnostics['left']['target_position_error_m']=.006
    assert adaptive_stop_evidence(trace,diagnostics,target)['at_target'] is False
    diagnostics['left']['target_position_error_m']=.004
    trace[-1]['obs']['states'][0]=.006
    assert adaptive_stop_evidence(trace,diagnostics,target)['joints_stable'] is False
    trace[-1]['obs']['states'][0]=0
    blocked={arm:dict(copy.deepcopy(POSE),gripper_closed=True) for arm in ('left','right')}
    for row in trace: row['obs']['states'][[6,13]]=.4
    evidence=adaptive_stop_evidence(trace,diagnostics,blocked)
    assert evidence['grippers_stable'] and evidence['max_gripper_error']==pytest.approx(.4)
    stalled_trace=[]
    for _ in range(5):
        stalled_trace.append(dict(obs=dict(states=state.copy()),ik_diagnostics={
            arm:dict(target_position_error_m=.02,target_rotation_error_rad=.04)
            for arm in ('left','right')}))
    stalled={arm:dict(target_position_error_m=.0198,target_rotation_error_rad=.039)
             for arm in ('left','right')}
    assert adaptive_stop_evidence(stalled_trace,stalled,target)['stalled_before_target']


def test_tool_schema_has_no_policy_inference_or_gate(tmp_path):
    specs=tool_specs('gpt_only')
    assert [s['name'] for s in specs]==['robodojo_start','robodojo_act']
    props=specs[1]['inputSchema']['properties']['response']['properties']
    assert props['mode']['enum']==['eef'] and 'assessment' not in props and 'actions' not in props
    assert props['steps']['minimum']==props['steps']['maximum']==50
    audit=tmp_path/'audit';audit.mkdir()
    agent=prepare_workspace(audit,Path(__file__).parent/'robodojo-gpt-only-rollout',method='gpt_only')
    assert (agent/'NOTES.md').is_file()
    assert not list(agent.rglob('gate_prompt.md'))


def test_two_methods_same_case_do_not_block_or_merge(tmp_path):
    _,_,panel=fake_panel_file(tmp_path)
    case=panel['cases'][6]
    batch=attempt(tmp_path,panel,case,success=True)
    require_available(tmp_path,panel,case['case_id'],method='gpt_only')
    assert build_ledger(tmp_path,panel,method='gpt_only')['cases'][6]['state']=='not_started'
    assert build_ledger(tmp_path,panel)['cases'][6]['state']=='completed'
    raw=json.loads(batch.read_text());raw['evaluation_method']='gpt_only';write_json(batch,raw)
    with pytest.raises(ValueError):require_available(tmp_path,panel,case['case_id'],method='gpt_only')
    assert build_ledger(tmp_path,panel)['cases'][6]['state']=='not_started'


def test_native_recorder_accepts_direct_joint_source(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from .robodojo_server import session as module
    from .test_native_evaluation import NativeLikeEnv
    sim = module.RoboDojoSession(NativeLikeEnv(), tmp_path, 'build_tower')
    monkeypatch.setattr(module, 'DualKinematics', lambda env: SimpleNamespace(check=lambda: {'passed': True}))
    monkeypatch.setattr(sim.env, 'render', lambda: None, raising=False)
    def observe(record=False):
        sim.obs = {'states': np.zeros(14, dtype=np.float32), 'instruction': 'Build a tower.'}
        return sim.obs
    monkeypatch.setattr(sim, '_observe', observe)
    sim.reset(seed=0, source='gpt_joint', policy_version='gpt_only')
    identity = dict(episode_id=sim.episode_id, step_id=0)
    sim.dispatch('switch_control_source', dict(identity, source='gpt_joint', reason='Direct action.'))
    response = dict(request_id='r', mode='joint', steps=1, reason='Direct action.', evaluation_method='gpt_only')
    sim.dispatch('record_codex_decision', dict(identity, decision=0, prediction_id=None, response=response))
    result = sim.dispatch('chunk_step', dict(identity, actions=np.zeros((1,14),dtype=np.float32)))
    assert result['steps'][0]['source'] == 'gpt_joint' and result['step_id'] == 1


def test_archive_checks_prove_direct_tools_and_absence_of_policy(tmp_path):
    from .artifact_manifest import build_manifest
    from .test_cluster_runtime import synthetic_archive
    synthetic_archive(tmp_path)
    write_json(tmp_path/'controller/run.json', dict(evaluation_method='gpt_only', pi05_enabled=False))
    result_path = tmp_path/'controller/result.json'
    result = json.loads(result_path.read_text())
    result.update(evaluation_method='gpt_only', pi05_enabled=False, predictions=0, student_steps=0)
    write_json(result_path, result)
    workspace = tmp_path/'controller/codex_workspace'; workspace.mkdir(exist_ok=True)
    write_json(workspace/'worker.json', dict(evaluation_method='gpt_only', tools=['robodojo_start','robodojo_act']))
    write_json(workspace/'call_0000_request.json', dict(tool='robodojo_start'))
    assert build_manifest(tmp_path)['status'] == 'verified'
    write_json(workspace/'call_0001_request.json', dict(tool='pi05_infer'))
    assert build_manifest(tmp_path)['checks']['gpt_only_direct_tool_contract'] is False
