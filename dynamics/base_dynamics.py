from abc import ABC, abstractmethod
from typing import Optional, Union

import torch
from torch import Tensor
import torch.autograd as autograd
import torch.nn.functional as F
import pytorch3d.transforms as T
from omegaconf import DictConfig
from tensordict import TensorDict

from .utils.math import (
    quat_rotate,
    quat_rotate_inverse,
    mvp,
    axis_rotmat,
    quaternion_to_euler,
    quat_standardize
)
from .utils.spaces import CoordinateFrame

class QuadrotorState(TensorDict):
    def __init__(
        self,
        n_envs: int,
        n_agents: int,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.float32
    ):
        super().__init__({
            "position":         torch.zeros(n_envs, n_agents, 3, device=device, dtype=dtype),
            "velocity":         torch.zeros(n_envs, n_agents, 3, device=device, dtype=dtype),
            "acceleration":     torch.zeros(n_envs, n_agents, 3, device=device, dtype=dtype),
            "acc_thrust":       torch.zeros(n_envs, n_agents, 3, device=device, dtype=dtype),
            "jerk":             torch.zeros(n_envs, n_agents, 3, device=device, dtype=dtype),
            "angular_velocity": torch.zeros(n_envs, n_agents, 3, device=device, dtype=dtype),
            "quat_xyzw":        torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=dtype).repeat(n_envs, n_agents, 1),
            "R":                torch.eye(3, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).repeat(n_envs, n_agents, 1, 1),
            "Rz":               torch.eye(3, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).repeat(n_envs, n_agents, 1, 1)
        }, batch_size=[n_envs, n_agents], device=device)
    
    @property
    def p(self) -> Tensor: return self["position"]
    @property
    def v(self) -> Tensor: return self["velocity"]
    @property
    def a(self) -> Tensor: return self["acceleration"]
    @property
    def a_thrust(self) -> Tensor: return self["acc_thrust"]
    @property
    def j(self) -> Tensor: return self["jerk"]
    @property
    def w(self) -> Tensor: return self["angular_velocity"]
    @property
    def q(self) -> Tensor: return self["quat_xyzw"]
    @property
    def R(self) -> Tensor: return self["R"]
    @property
    def Rz(self) -> Tensor: return self["Rz"]
    
    @p.setter
    def p(self, value: Tensor): self["position"] = value
    @v.setter
    def v(self, value: Tensor): self["velocity"] = value
    @a.setter
    def a(self, value: Tensor): self["acceleration"] = value
    @a_thrust.setter
    def a_thrust(self, value: Tensor): self["acc_thrust"] = value
    @j.setter
    def j(self, value: Tensor): self["jerk"] = value
    @w.setter
    def w(self, value: Tensor): self["angular_velocity"] = value
    @q.setter
    def q(self, value: Tensor):
        q: Tensor = quat_standardize(value)
        self["quat_xyzw"] = q
        self["R"] = T.quaternion_to_matrix(q.roll(1, -1))
        self["Rz"] = axis_rotmat("Z", quaternion_to_euler(q)[..., 2])
    
    def get_mask(self, env_idx: Tensor) -> TensorDict:
        mask: TensorDict = self.new_zeros(self.shape, dtype=torch.bool) # type: ignore
        mask[env_idx] = True # magic
        return mask
    
    def masked_assign(self, key: str, value: Tensor, mask: TensorDict):
        self[key] = torch.where(mask[key], value, self[key])
    
    def assign_with_idx(self, key: str, value: Tensor, env_idx: Tensor):
        mask = self.get_mask(env_idx)
        self.masked_assign(key, value, mask)
    
    def reset_pose(self, env_idx: Optional[Tensor] = None):
        qnew = torch.zeros_like(self.q)
        qnew[..., 3].fill_(1.)
        Rnew = torch.zeros_like(self.R)
        Rnew[..., 0, 0].fill_(1.)
        Rnew[..., 1, 1].fill_(1.)
        Rnew[..., 2, 2].fill_(1.)
        
        if env_idx is None:
            self.q.copy_(qnew, non_blocking=True)
            self.R.copy_(Rnew, non_blocking=True)
            self.Rz.copy_(Rnew, non_blocking=True)
        else:
            mask = self.get_mask(env_idx)
            self.masked_assign("quat_xyzw", qnew, mask)
            self.masked_assign("R", Rnew, mask)
            self.masked_assign("Rz", Rnew, mask)
    
    # def update_rotation_matrices(self):
    #     self["R"].copy_(T.quaternion_to_matrix(self.q.roll(1, -1)), non_blocking=True)
    #     self["Rz"].copy_(axis_rotmat("Z", quaternion_to_euler(self.q)[..., 2]), non_blocking=True)
    
    # def update_quaternion_from_R(self):
    #     self["quat_xyzw"].copy_(T.matrix_to_quaternion(self.R).roll(-1, -1), non_blocking=True)
    #     self["Rz"].copy_(axis_rotmat("Z", quaternion_to_euler(self.q)[..., 2]), non_blocking=True)


class GradientDecay(autograd.Function):
    @staticmethod
    def forward(ctx, state: QuadrotorState, alpha: float, dt: float):
        ctx.save_for_backward(torch.tensor(-alpha * dt, device=state.device).exp())
        return state
    
    @staticmethod
    def backward(ctx, grad_state: QuadrotorState):
        decay_factor = ctx.saved_tensors[0]
        if ctx.needs_input_grad[0]:
            grad_state = grad_state * decay_factor
        return grad_state, None, None


class BaseDynamics(ABC):
    def __init__(self, cfg: DictConfig, device: torch.device):
        self.type: str
        self.state_dim: int
        self.action_dim: int
        self.device = device
        self.n_agents: int = cfg.n_agents
        self.n_envs: int = cfg.n_envs
        self.dt: float = cfg.dt
        self.alpha: float = cfg.alpha
        self.n_substeps: int = 1
        self._state = QuadrotorState(self.n_envs, self.n_agents, device)
        self.action_frame = CoordinateFrame(cfg.action_frame)
        
        self._G = torch.tensor(cfg.g, device=device, dtype=torch.float32)
        self._G_vec = torch.tensor([0.0, 0.0, -self._G], device=device, dtype=torch.float32)
        if self.n_agents > 1:
            self._G_vec.unsqueeze_(0)

    def detach(self):
        """Detach the state to prevent backpropagation through released computation graphs."""
        self._state = self._state.detach()
    
    def grad_decay(self, state: QuadrotorState) -> QuadrotorState:
        if self.alpha > 0 and state.requires_grad:
            state = GradientDecay.apply(state, self.alpha, self.dt/self.n_substeps) # type: ignore
        return state
    
    @abstractmethod
    def step(
        self,
        action: Tensor,
        target_vel: Optional[Tensor] = None,
        latency: Optional[Tensor] = None,
        last_action: Optional[Tensor] = None
    ) -> None:
        """Step the model with the given action U.

        Args:
            action (Tensor): The action tensor of shape (n_envs, n_agents, 3).
            target_vel (Tensor, optional): The target velocity tensor of shape (n_envs, n_agents, 3). Defaults to None.
            latency (Tensor, optional): The latency (ms) tensor of shape (n_envs,). Defaults to None.
            last_action (Tensor, optional): The last action tensor of shape (n_envs,
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    # Action ranges
    @property
    @abstractmethod
    def min_action(self) -> Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def max_action(self) -> Tensor:
        raise NotImplementedError
    
    def reset_idx(self, env_idx: Tensor, p_new: Tensor):
        mask = self._state.get_mask(env_idx)
        zero = torch.zeros_like(self._state.v)
        self._state.masked_assign("position", p_new, mask)
        self._state.masked_assign("velocity", zero, mask)
        self._state.masked_assign("acceleration", zero, mask)
        self._state.masked_assign("acc_thrust", -self._G_vec.expand_as(self.a_thrust), mask)
        self._state.masked_assign("jerk", zero, mask)
        self._state.masked_assign("angular_velocity", zero, mask)
        self._state.reset_pose(env_idx)

    # Properties of agents, requires_grad=True if stepped with undetached action inputs
    @property
    def _p(self) -> Tensor: return self._state.p
    @property
    def _v(self) -> Tensor: return self._state.v
    @property
    def _a(self) -> Tensor: return self._state.a
    @property
    def _a_thrust(self) -> Tensor: return self._state.a_thrust
    @property
    def _w(self) -> Tensor: return self._state.w
    @property
    def _q(self) -> Tensor: return self._state.q
    @property
    def _R(self) -> Tensor: return self._state.R
    @property
    def _Rz(self) -> Tensor: return self._state.Rz
    
    # Detached versions of properties
    @property
    def p(self) -> Tensor: return self._p.detach()
    @property
    def v(self) -> Tensor: return self._v.detach()
    @property
    def a(self) -> Tensor: return self._a.detach()
    @property
    def a_thrust(self) -> Tensor: return self._a_thrust.detach()
    @property
    def w(self) -> Tensor: return self._w.detach()
    @property
    def q(self) -> Tensor: return self._q.detach()
    @property
    def R(self) -> Tensor: return self._R.detach()
    @property
    def Rz(self) -> Tensor: return self._Rz.detach()
    
    @property
    def ux(self) -> Tensor:
        "Unit vector along the x-axis of the body frame in world frame."
        return self.R[..., 0]
    
    @property
    def uy(self) -> Tensor:
        "Unit vector along the y-axis of the body frame in world frame."
        return self.R[..., 1]
    
    @property
    def uz(self) -> Tensor:
        "Unit vector along the z-axis of the body frame in world frame."
        return self.R[..., 2]
    
    def world2body(self, vec_w: Tensor) -> Tensor:
        """
        Convert vector from world frame to body frame.
        Args:
            vec_w (Tensor): vector in world frame
        Returns:
            Tensor: vector in body frame
        """
        return quat_rotate_inverse(self.q, vec_w)
    
    def body2world(self, vec_b: Tensor) -> Tensor:
        """
        Convert vector from body frame to world frame.
        Args:
            vec_b (Tensor): vector in body frame
        Returns:
            Tensor: vector in world frame
        """
        return quat_rotate(self.q, vec_b)

    def world2local(self, vec_w: Tensor) -> Tensor:
        """
        Convert vector from world frame to local frame.
        Args:
            vec_w (Tensor): vector in world frame
        Returns:
            Tensor: vector in local frame
        """
        # Logger.debug(mvp(self.Rz.transpose(-1, -2), self.ux)[0][..., 1].cpu(), "should be around 0")
        return mvp(self.Rz.transpose(-1, -2), vec_w)
    
    def local2world(self, vec_l: Tensor) -> Tensor:
        """
        Convert vector from local frame to world frame.
        Args:
            vec_l (Tensor): vector in local frame
        Returns:
            Tensor: vector in world frame
        """
        return mvp(self.Rz, vec_l)

    def get_desired_orientation(self, target_vel: Optional[Tensor] = None):
        raise NotImplementedError
