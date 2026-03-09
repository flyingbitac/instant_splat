import warnings
from typing import Optional, Union

import torch
from torch import Tensor
from torch.nn import functional as F
from pytorch3d import transforms as T
from omegaconf import DictConfig

from .base_dynamics import BaseDynamics, QuadrotorState
from .utils.math import (
    EulerIntegral,
    rk4,
    axis_rotmat,
    mvp,
    quat_mul,
    quat_from_two_vectors
)
from .utils.randomizer import build_randomizer
from .utils.spaces import CoordinateFrame

class PointMassModelBase(BaseDynamics):
    def __init__(self, cfg: DictConfig, device: torch.device):
        super().__init__(cfg, device)
        self.type = "pointmass"
        self.action_dim = 3
        self._vel_ema = torch.zeros(self.n_envs, self.n_agents, 3, device=device)
        self._orientation = torch.zeros(self.n_envs, self.n_agents, 3, device=device)
        self.align_yaw_with_target_direction: bool = cfg.align_yaw_with_target_direction
        self.align_yaw_with_vel_ema: bool = cfg.align_yaw_with_vel_ema
    
        self.vel_ema_factor = build_randomizer(cfg.vel_ema_factor, [self.n_envs, self.n_agents, 1], device=device)
        self._D = build_randomizer(cfg.D, [self.n_envs, self.n_agents, 1], device=device)
        self.lmbda = build_randomizer(cfg.lmbda, [self.n_envs, self.n_agents, 1], device=device)
        
        if self.n_agents == 1:
            self._state = self._state.squeeze(1)
            self._vel_ema.squeeze_(1)
            self._orientation.squeeze_(1)
            self.vel_ema_factor.value.squeeze_(1)
            self._D.value.squeeze_(1)
            self.lmbda.value.squeeze_(1)
        
        self.max_acc_xy = build_randomizer(cfg.max_acc.xy, [self.n_envs, self.n_agents], device=device)
        self.max_acc_z = build_randomizer(cfg.max_acc.z, [self.n_envs, self.n_agents], device=device)
        self.point_mass_quat_fn = torch.jit.script(point_mass_quat)
        
    @property
    def min_action(self) -> Tensor:
        zero = torch.zeros_like(self.max_acc_xy.value)
        min_action = torch.stack([-self.max_acc_xy.value, -self.max_acc_xy.value, zero], dim=-1)
        if self.n_agents == 1:
            min_action.squeeze_(1)
        return min_action

    @property
    def max_action(self) -> Tensor:
        max_action = torch.stack([self.max_acc_xy.value, self.max_acc_xy.value, self.max_acc_z.value], dim=-1)
        if self.n_agents == 1:
            max_action.squeeze_(1)
        return max_action
    
    def detach(self):
        super().detach()
        self._vel_ema.detach_()
    
    def reset_idx(self, env_idx: Tensor, p_new: Tensor) -> None:
        super().reset_idx(env_idx, p_new)
        mask = torch.zeros_like(self._vel_ema, dtype=torch.bool)
        mask[env_idx] = True
        self._vel_ema = torch.where(mask, 0., self._vel_ema)
        self._orientation[env_idx] = 0.
    
    def get_desired_orientation(self, target_vel: Optional[Tensor] = None):
        if self.align_yaw_with_target_direction and target_vel is not None:
            new_orientation = target_vel
        else:
            if self.align_yaw_with_vel_ema:
                new_orientation = self._vel_ema.detach()
            else:
                new_orientation = self.v
        new_orientation = F.normalize(new_orientation)
        if target_vel is None:
            return self._orientation.lerp_(new_orientation, 1.)
        target_vel_norm = torch.norm(target_vel, dim=-1, keepdim=True)
        update_rate = torch.sigmoid(target_vel_norm - 2)
        return self._orientation.lerp_(new_orientation, update_rate)
    
    def update_state(self, next_state: QuadrotorState, orientation: Tensor) -> None:
        self._state = self.grad_decay(next_state)
        self._vel_ema = torch.lerp(self._vel_ema, self._v, self.vel_ema_factor.value)
        self._state.a = self._a_thrust + self._G_vec - self._D.value * self._v
        with torch.no_grad():
            self._state.q = self.point_mass_quat_fn(self.a_thrust, orientation=orientation)

def continuous_point_mass_dynamics(
    X: QuadrotorState,
    U: Tensor,
    dt: Union[float, Tensor],
    G_vec: Tensor,
    D: Tensor,
    lmbda: Tensor,
    action_frame: CoordinateFrame
):
    """Dynamics function for continuous point mass model in local frame."""
    p, v, a_thrust = X.p, X.v, X.a_thrust
    p_dot = v
    fdrag = -D * v
    v_dot = a_thrust + G_vec + fdrag
    control_delay_factor = (1 - torch.exp(-lmbda * dt)) / dt
    a_thrust_cmd = U
    if action_frame == CoordinateFrame.LOCAL:
        a_thrust_cmd = mvp(X.Rz, a_thrust_cmd, clone=True)
    a_thrust_dot = control_delay_factor * (a_thrust_cmd - a_thrust)
    X_dot: QuadrotorState = X.new_zeros(X.shape)
    X_dot.p = p_dot # type: ignore
    X_dot.v = v_dot # type: ignore
    X_dot.a_thrust = a_thrust_dot # type: ignore
    return X_dot

class ContinuousPointMassModel(PointMassModelBase):
    def __init__(self, cfg: DictConfig, device: torch.device):
        super().__init__(cfg, device)
        self.n_substeps: int = cfg.n_substeps
        assert cfg.solver_type in ["euler", "rk4"]
        if cfg.solver_type == "euler":
            self.solver = EulerIntegral
        elif cfg.solver_type == "rk4":
            self.solver = rk4
        self.Rz_temp: Tensor
    
    def dynamics(self, X: QuadrotorState, U: Tensor) -> QuadrotorState:
        X_dot = continuous_point_mass_dynamics(
            X, U, self.dt/self.n_substeps, self._G_vec, self._D.value, self.lmbda.value, self.action_frame
        )
        return X_dot

    def step(
        self,
        action: Tensor,
        target_vel: Optional[Tensor] = None,
        latency: Optional[Tensor] = None,
        last_action: Optional[Tensor] = None
    ) -> None:
        if latency is not None and last_action is not None:
            stalled_dt = latency.float() / 1000. # ms to s
            for _ in range(self.n_substeps):
                next_state = self.solver(self.dynamics, self._state, last_action, dt=stalled_dt/self.n_substeps, M=1)
                self.update_state(next_state, orientation=self.get_desired_orientation(target_vel))
            
            delayed_dt = self.dt - stalled_dt
            for _ in range(self.n_substeps):
                next_state = self.solver(self.dynamics, self._state, action, dt=delayed_dt/self.n_substeps, M=1)
                self.update_state(next_state, orientation=self.get_desired_orientation(target_vel))
        else:
            for _ in range(self.n_substeps):
                next_state = self.solver(self.dynamics, self._state, action, dt=self.dt/self.n_substeps, M=1)
                self.update_state(next_state, orientation=self.get_desired_orientation(target_vel))

# @torch.jit.script
def discrete_point_mass_dynamics(
    X: QuadrotorState,
    U: Tensor,
    dt: Union[float, Tensor],
    G_vec: Tensor,
    D: Tensor,
    lmbda: Tensor,
    action_frame: CoordinateFrame
):
    """Dynamics function for discrete point mass model in local frame."""
    p, v, a_thrust = X.p, X.v, X.a_thrust
    next_p = p + dt * (v + 0.5 * (a_thrust + G_vec) * dt)
    control_delay_factor = 1 - torch.exp(-lmbda*dt)
    a_thrust_cmd = U
    if action_frame == CoordinateFrame.LOCAL:
        a_thrust_cmd = mvp(X.Rz, a_thrust_cmd, clone=True)
    next_a = torch.lerp(a_thrust, a_thrust_cmd, control_delay_factor) - D * v
    next_v = v + dt * (0.5 * (a_thrust + next_a) + G_vec)
    next_state: QuadrotorState = X.new_zeros(X.shape) # type: ignore
    next_state.p = next_p # type: ignore
    next_state.v = next_v # type: ignore
    next_state.a_thrust = next_a # type: ignore
    return next_state

class DiscretePointMassModel(PointMassModelBase):
    def step(
        self,
        action: Tensor,
        target_vel: Optional[Tensor] = None,
        latency: Optional[Tensor] = None,
        last_action: Optional[Tensor] = None
    ) -> None:
        if latency is not None and last_action is not None:
            stalled_dt = latency.float() / 1000. # ms to s
            for _ in range(self.n_substeps):
                next_state = discrete_point_mass_dynamics(
                    self._state, last_action, stalled_dt/self.n_substeps, self._G_vec, self._D.value, self.lmbda.value, self.action_frame)
                self.update_state(next_state, orientation=self.get_desired_orientation(target_vel))
            
            delayed_dt = self.dt - stalled_dt
            for _ in range(self.n_substeps):
                next_state = discrete_point_mass_dynamics(
                    self._state, action, delayed_dt/self.n_substeps, self._G_vec, self._D.value, self.lmbda.value, self.action_frame)
                self.update_state(next_state, orientation=self.get_desired_orientation(target_vel))
        else:
            for _ in range(self.n_substeps):
                next_state = discrete_point_mass_dynamics(
                    self._state, action, self.dt/self.n_substeps, self._G_vec, self._D.value, self.lmbda.value, self.action_frame)
                self.update_state(next_state, orientation=self.get_desired_orientation(target_vel))

# @torch.jit.script
def point_mass_quat(a: Tensor, orientation: Tensor) -> Tensor:
    """Compute the drone pose using target direction and thrust acceleration direction.

    Args:
        a (Tensor): the acceleration of the drone in world frame.
        orientation (Tensor): at which direction(yaw) the drone should be facing.

    Returns:
        Tensor: attitude quaternion of the drone with real part last.
    """
    up: Tensor = F.normalize(a, dim=-1)
    yaw = torch.atan2(orientation[..., 1], orientation[..., 0])
    mat_yaw = axis_rotmat("Z", yaw)
    new_up = (mat_yaw.transpose(-2, -1) @ up.unsqueeze(-1)).squeeze(-1)
    z = torch.zeros_like(new_up)
    z[..., -1] = 1.
    quat_pitch_roll = quat_from_two_vectors(z, new_up)
    yaw_half = yaw.unsqueeze(-1) / 2
    quat_yaw = torch.concat([torch.sin(yaw_half) * z, torch.cos(yaw_half)], dim=-1) # T.matrix_to_quaternion(mat_yaw)
    quat_xyzw = quat_mul(quat_yaw, quat_pitch_roll)
    return quat_xyzw
