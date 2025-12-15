import numpy as np
import torch
from scipy.ndimage import affine_transform
from so2_equiv_rl.storage.transitions import Transition


def get_random_image_transform_params(image_size):
    theta = np.random.random() * 2 * np.pi
    trans = np.random.randint(0, image_size[0] // 10, 2) - image_size[0] // 20
    pivot = (image_size[1] / 2, image_size[0] / 2)
    return theta, trans, pivot


def get_image_transform(theta, trans, pivot=(0, 0)):
    pivot_t_image = np.array(
        [[1.0, 0.0, -pivot[0]], [0.0, 1.0, -pivot[1]], [0.0, 0.0, 1.0]]
    )
    image_t_pivot = np.array(
        [[1.0, 0.0, pivot[0]], [0.0, 1.0, pivot[1]], [0.0, 0.0, 1.0]]
    )
    transform = np.array(
        [
            [np.cos(theta), -np.sin(theta), trans[0]],
            [np.sin(theta), np.cos(theta), trans[1]],
            [0.0, 0.0, 1.0],
        ]
    )
    return np.dot(image_t_pivot, np.dot(transform, pivot_t_image))


def perturb(current_image, next_image, dxy, set_theta_zero=False, set_trans_zero=False):
    image_size = current_image.shape[-2:]

    # Compute random rigid transform.
    theta, trans, pivot = get_random_image_transform_params(image_size)
    if set_theta_zero:
        theta = 0.0
    if set_trans_zero:
        trans = [0.0, 0.0]
    transform = get_image_transform(theta, trans, pivot)
    transform_params = theta, trans, pivot

    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated_dxy = rot.dot(dxy)
    rotated_dxy = np.clip(rotated_dxy, -1, 1)

    # Apply rigid transform to image and pixel labels.
    current_image = affine_transform(
        current_image, np.linalg.inv(transform), mode="nearest", order=1
    )
    if next_image is not None:
        next_image = affine_transform(
            next_image, np.linalg.inv(transform), mode="nearest", order=1
        )
    return current_image, next_image, rotated_dxy, transform_params


def so2_augment_close_loop_env(t: Transition) -> Transition:

    obs, next_obs, dxy, _ = perturb(
        t.obs[0].copy(), t.next_obs[0].copy(), t.action[1:3].copy(), set_trans_zero=True
    )
    obs = obs.reshape(1, *obs.shape)
    next_obs = next_obs.reshape(1, *next_obs.shape)
    action = t.action.copy()
    action[1] = dxy[0]
    action[2] = dxy[1]
    return Transition(
        state=t.state,
        obs=obs,
        action=action,
        reward=t.reward,
        next_state=t.next_state,
        next_obs=next_obs,
        done=t.done,
    )


def normalize_transition(t: Transition) -> Transition:
    return t
    obs = np.clip(t.obs, 0, 0.32)
    obs = obs / 0.4 * 255
    obs = obs.astype(np.uint8)

    next_obs = np.clip(t.next_obs, 0, 0.32)
    next_obs = next_obs / 0.4 * 255
    next_obs = next_obs.astype(np.uint8)

    return Transition(
        state=t.state,
        obs=obs,
        action=t.action,
        reward=t.reward,
        next_state=t.next_state,
        next_obs=next_obs,
        done=t.done,
    )


def denormalize_observation(obs: torch.Tensor) -> torch.Tensor:
    return obs
    obs = obs.to(torch.float32)
    return obs / 255 * 0.4


def get_unscaled_action(action, action_range):
    """
    Maps action from true scale to (-1, 1)

    :param action: action in true scale
    :param action_range: tuple of (min, max) for the action
    """
    unscaled_action = (
        2 * (action - action_range[0]) / (action_range[1] - action_range[0]) - 1
    )
    return unscaled_action


def decode_actions(*args, action_sequence, dx, dy, dz, dr):
    """
    Decode unscaled actions to scaled actions
    :param args: unscaled actions
    :return: unscaled_actions (in range (-1, 1)), actions (in true scale)
    """
    unscaled_p, unscaled_dx, unscaled_dy, unscaled_dz = (
        args[0],
        args[1],
        args[2],
        args[3],
    )

    p = 0.5 * (unscaled_p + 1) * (1 - 0) + 0
    dx_val = 0.5 * (unscaled_dx + 1) * (2 * dx) - dx
    dy_val = 0.5 * (unscaled_dy + 1) * (2 * dy) - dy
    dz_val = 0.5 * (unscaled_dz + 1) * (2 * dz) - dz

    if "r" in action_sequence:
        unscaled_dr = args[4]
        dr_val = 0.5 * (unscaled_dr + 1) * (2 * dr) - dr
        actions = torch.stack([p, dx_val, dy_val, dz_val, dr_val], dim=1)
        unscaled_actions = torch.stack(
            [unscaled_p, unscaled_dx, unscaled_dy, unscaled_dz, unscaled_dr], dim=1
        )
    else:
        actions = torch.stack([p, dx_val, dy_val, dz_val], dim=1)
        unscaled_actions = torch.stack(
            [unscaled_p, unscaled_dx, unscaled_dy, unscaled_dz], dim=1
        )

    return unscaled_actions, actions


def get_action_from_plan(plan, action_sequence, dx, dy, dz, dr):
    """
    Get unscaled and scaled actions from scaled planner action
    :param plan: scaled planner action (in true scale)
    :return: unscaled_actions (in range (-1, 1)), actions (in true scale)
    """
    dx_range = (-dx, dx)
    dy_range = (-dy, dy)
    dz_range = (-dz, dz)
    dr_range = (-dr, dr)

    dx_val = plan[:, 1].clamp(*dx_range)
    p = plan[:, 0].clamp(0, 1)
    dy_val = plan[:, 2].clamp(*dy_range)
    dz_val = plan[:, 3].clamp(*dz_range)

    unscaled_p = get_unscaled_action(p, (0, 1))
    unscaled_dx = get_unscaled_action(dx_val, dx_range)
    unscaled_dy = get_unscaled_action(dy_val, dy_range)
    unscaled_dz = get_unscaled_action(dz_val, dz_range)

    if "r" in action_sequence:
        dr_val = plan[:, 4].clamp(*dr_range)
        unscaled_dr = get_unscaled_action(dr_val, dr_range)
        return decode_actions(
            unscaled_p,
            unscaled_dx,
            unscaled_dy,
            unscaled_dz,
            unscaled_dr,
            action_sequence=action_sequence,
            dx=dx,
            dy=dy,
            dz=dz,
            dr=dr,
        )
    else:
        return decode_actions(
            unscaled_p,
            unscaled_dx,
            unscaled_dy,
            unscaled_dz,
            action_sequence=action_sequence,
            dx=dx,
            dy=dy,
            dz=dz,
            dr=dr,
        )


def pack_state_obs(state, obs):
    if obs.ndim == 3:
        obs = obs[:, None, :, :]
    state = state.float()
    state_tile = state.reshape(state.size(0), 1, 1, 1).repeat(
        1, 1, obs.shape[2], obs.shape[3]
    )
    return torch.cat([obs, state_tile.to(obs.dtype)], dim=1)
