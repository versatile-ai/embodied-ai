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


def solve_step(model, measured, goals, *, check_initial_bound=True):
    """One bounded 6D DLS update; measured physics state is never written.

    5cm/.35rad validates a decision once. Subsequent tracking updates may
    compensate disturbances beyond that distance using the same small bounds.
    """
    targets = validate_goals(goals)
    data = mujoco.MjData(model)
    data.qpos[:] = measured.qpos
    mujoco.mj_forward(model, data)
    row = []
    def bounded(value, limit):
        return value * min(1.0, limit / max(np.linalg.norm(value), 1e-12))
    for side in ('left', 'right'):
        site = model.site(side + '_ee').id
        joints = [model.joint(f'{side}_joint{i}') for i in range(1, 7)]
        qids = np.array([j.qposadr[0] for j in joints])
        dids = np.array([j.dofadr[0] for j in joints])
        pos, rot, grip = targets[side]
        dp = pos - data.site_xpos[site]
        dr = rotation_error(rot, data.site_xmat[site].reshape(3, 3))
        if check_initial_bound and (np.linalg.norm(dp) > .050001 or np.linalg.norm(dr) > .350001):
            raise ValueError('Target exceeds 5 cm / 0.35 rad from the measured EEF pose')
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jp, jr, site)
        jac = np.vstack((jp[:, dids], jr[:, dids]))
        error = np.r_[bounded(dp, .02), bounded(dr, .1)]
        dq = jac.T @ np.linalg.solve(jac @ jac.T + .05**2 * np.eye(6), error)
        q0 = data.qpos[qids].copy()
        low, high = q0 - .05, q0 + .05
        for i, joint in enumerate(joints):
            if joint.limited[0]:
                low[i] = max(low[i], joint.range[0])
                high[i] = min(high[i], joint.range[1])
        if np.any(low > high):
            raise ValueError('Measured joint outside bounded valid interval')
        # Do not independently clip every joint.  For coupled Cartesian
        # motions (especially wrist rotation), component-wise clipping can
        # flatten unequal joint increments into equal ones and cancel the
        # intended EEF motion.  Scale the complete DLS direction until its
        # first joint reaches a per-tick or model limit, preserving the
        # kinematic coordination selected by the solver.
        positive = dq > 0
        negative = dq < 0
        fractions = [1.0]
        if np.any(positive):
            fractions.extend(((high[positive] - q0[positive]) / dq[positive]).tolist())
        if np.any(negative):
            fractions.extend(((low[negative] - q0[negative]) / dq[negative]).tolist())
        scale = float(np.clip(min(fractions), 0.0, 1.0))
        row.extend((q0 + scale * dq).tolist() + [grip])
    return row
