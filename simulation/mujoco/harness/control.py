"""One measured-state IK solve per real control step; no physics rollout."""
import numpy as np
import mujoco


def rotation_error(target, current):
    q1, q2 = np.zeros(4), np.zeros(4)
    mujoco.mju_mat2Quat(q1, target.ravel())
    mujoco.mju_mat2Quat(q2, current.ravel())
    error = np.zeros(3)
    mujoco.mju_subQuat(error, q1, q2)
    # subQuat is in the current local frame; Jacobians are world-frame.
    return current @ error


def validate_goals(goals):
    if not isinstance(goals, dict) or set(goals) != {'left', 'right'}:
        raise ValueError('Provide both left and right world-frame targets')
    result = {}
    for side in ('left', 'right'):
        g = goals[side]
        pos, quat = np.asarray(g['xyz'],float), np.asarray(g['quat_wxyz'],float)
        grip = float(g['gripper'])
        if pos.shape != (3,) or quat.shape != (4,) or not np.isfinite(np.r_[pos,quat,grip]).all():
            raise ValueError('Targets must contain finite xyz, quat_wxyz, gripper')
        if abs(np.linalg.norm(quat)-1)>1e-3 or not 0<=grip<=1:
            raise ValueError('Quaternion must be unit length; gripper must be in [0,1]')
        mat=np.zeros(9);mujoco.mju_quat2Mat(mat,quat)
        result[side]=(pos,mat.reshape(3,3),grip)
    return result


def solve_step(model, measured, goals):
    targets=validate_goals(goals)
    data=mujoco.MjData(model);data.qpos[:]=measured.qpos
    mujoco.mj_forward(model,data)
    row=[]
    for side in ('left','right'):
        site=model.site(side+'_ee').id
        qids=np.array([model.joint(f'{side}_joint{i}').qposadr[0] for i in range(1,7)])
        dids=np.array([model.joint(f'{side}_joint{i}').dofadr[0] for i in range(1,7)])
        pos,rot,grip=targets[side]
        start_pos=data.site_xpos[site].copy();start_rot=data.site_xmat[site].reshape(3,3).copy()
        if np.linalg.norm(pos-start_pos)>.050001 or np.linalg.norm(rotation_error(rot,start_rot))>.350001:
            raise ValueError('Target exceeds 5 cm / 0.35 rad from the measured EEF pose')
        q0=data.qpos[qids].copy()
        # Solve the complete EEF pose.  Position-only IK lets the redundant
        # wrist joints drift into a different orientation while translating;
        # with the X5 meshes that turns the fingers into table/object
        # collisions.  A damped 6D solve with a small bounded joint increment
        # keeps the requested grasp orientation stable, including near home.
        for _ in range(80):
            mujoco.mj_forward(model,data)
            pos_err = pos-data.site_xpos[site]
            rot_err = rotation_error(rot, data.site_xmat[site].reshape(3,3))
            if np.linalg.norm(pos_err)<.0005 and np.linalg.norm(rot_err)<.005:
                break
            jp=np.zeros((3,model.nv)); jr=np.zeros((3,model.nv))
            mujoco.mj_jacSite(model,data,jp,jr,site)
            # The orientation row is slightly down-weighted, but still
            # actively regulated so the grasp frame stays fixed.
            j=np.vstack((jp[:,dids], 0.7*jr[:,dids]))
            err=np.r_[pos_err, 0.7*rot_err]
            dq=j.T@np.linalg.solve(j@j.T+np.eye(6)*1e-2,err)
            data.qpos[qids] += np.clip(0.5*dq, -0.025, 0.025)
            for qi,i in zip(qids,range(1,7)):
                joint=model.joint(f'{side}_joint{i}')
                if joint.limited[0]:data.qpos[qi]=np.clip(data.qpos[qi],*joint.range)
        mujoco.mj_forward(model,data)
        # Bound the final command, rather than merely each solver iteration.
        for _ in range(12):
            dp=np.linalg.norm(data.site_xpos[site]-start_pos)
            dr=np.linalg.norm(rotation_error(data.site_xmat[site].reshape(3,3),start_rot))
            if dp<=.05 and dr<=.35:break
            data.qpos[qids]=q0+(data.qpos[qids]-q0)*.5
            mujoco.mj_forward(model,data)
        row.extend(data.qpos[qids].tolist()+[grip])
    return row
