"""
基于体素占据栅格 (Voxel Occupancy Grid) 的碰撞检测模块。

从 3DGS 场景的 Gaussian 点云构建栅格，运行时 O(1) 查表判定碰撞。
坐标系约定:
    FLU  (前-左-上)  — 无人机动力学
    3DGS (x=-y_flu, y=z_flu, z=-x_flu) — 场景渲染
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor


# ── 数据结构 ────────────────────────────────────────────────────────────────

@dataclass
class CollisionResult:
    """单次碰撞检测结果 (3DGS 坐标系)。"""
    collided: bool
    collision_point_3dgs: Optional[Tensor] = None
    surface_normal: Optional[Tensor] = None
    penetration_depth: float = 0.0


@dataclass
class TrajectoryCollisionResult:
    """一架无人机 N 个位姿的批量碰撞检测结果。"""
    has_collision: bool
    collision_mask: Optional[Tensor] = None        # (N,) bool
    first_collision_idx: int = -1
    collision_count: int = 0
    first_collision_point_3dgs: Optional[Tensor] = None


# ── 碰撞检测器 ──────────────────────────────────────────────────────────────

class CollisionDetector:
    """基于体素占据栅格的碰撞检测器，工作在 3DGS 坐标系中。"""

    def __init__(
        self,
        voxel_size: float = 0.05,
        opacity_threshold: float = 0.5,
        scale_max_threshold: float = 1.0,
        drone_radius: float = 0.15,
        padding: int = 2,
        device: torch.device = torch.device("cuda"),
    ):
        self.voxel_size = voxel_size
        self.opacity_threshold = opacity_threshold
        self.scale_max_threshold = scale_max_threshold
        self.drone_radius = drone_radius
        self.padding = padding
        self.device = device

        self.grid: Optional[Tensor] = None                     # bool 3D tensor
        self.grid_origin: Optional[Tensor] = None              # 栅格原点 (3DGS)
        self.grid_shape: Optional[Tuple[int, int, int]] = None

    # ── 栅格构建 ──────────────────────────────────────────────────────────

    def build_from_gaussians(
        self,
        xyz: Tensor,
        opacity_raw: Tensor,
        scaling_raw: Tensor,
    ) -> None:
        """从 Gaussian 点云构建体素占据栅格。

        Args:
            xyz:          Gaussian 中心 (N, 3)，3DGS 坐标系
            opacity_raw:  原始 opacity (N, 1)，需 sigmoid 激活
            scaling_raw:  原始 scale   (N, 3)，需 exp 激活
        """
        with torch.no_grad():
            # ---- 过滤: 高 opacity & 非过大 scale ----
            opacity = torch.sigmoid(opacity_raw).squeeze(-1)
            scales = torch.exp(scaling_raw)
            mask = (opacity > self.opacity_threshold) & \
                   (scales.max(dim=-1).values < self.scale_max_threshold)

            obs_xyz = xyz[mask]
            obs_scales = scales[mask]
            if obs_xyz.shape[0] == 0:
                print("[Collision] WARNING: No obstacle points found!")
                return

            print(f"[Collision] {obs_xyz.shape[0]}/{xyz.shape[0]} gaussians → obstacles")

            # ---- 栅格边界 (2σ + drone_radius padding) ----
            eff_r = 2.0 * obs_scales                         # 有效半径
            pts_min = (obs_xyz - eff_r).min(dim=0).values
            pts_max = (obs_xyz + eff_r).max(dim=0).values
            pad = self.padding * self.voxel_size + self.drone_radius

            self.grid_origin = pts_min - pad
            extent = pts_max + pad - self.grid_origin

            # ---- 栅格尺寸 (自动缩放防 OOM) ----
            dims = torch.ceil(extent / self.voxel_size).long()
            nx, ny, nz = dims[0].item(), dims[1].item(), dims[2].item()
            max_dim = 2048
            if max(nx, ny, nz) > max_dim:
                self.voxel_size *= max(nx, ny, nz) / max_dim
                dims = torch.ceil(extent / self.voxel_size).long()
                nx, ny, nz = dims[0].item(), dims[1].item(), dims[2].item()
                print(f"[Collision] Voxel size → {self.voxel_size:.4f}m (clamped)")

            self.grid_shape = (nx, ny, nz)
            self.grid = torch.zeros(nx, ny, nz, dtype=torch.bool, device=self.device)
            print(f"[Collision] Grid {nx}×{ny}×{nz}, "
                  f"voxel={self.voxel_size:.4f}m, "
                  f"~{nx * ny * nz / 1024 / 1024:.1f}MB")

            # ---- 填充体素 (Minkowski 膨胀: 2σ + drone_radius) ----
            batch = 50000
            for s in range(0, obs_xyz.shape[0], batch):
                bxyz = obs_xyz[s:s + batch]
                inflate = 2.0 * obs_scales[s:s + batch] + self.drone_radius
                rel = bxyz - self.grid_origin
                lo = torch.floor((rel - inflate) / self.voxel_size).long().clamp(min=0)
                hi = torch.ceil((rel + inflate) / self.voxel_size).long()
                hi[:, 0].clamp_(max=nx); hi[:, 1].clamp_(max=ny); hi[:, 2].clamp_(max=nz)
                for i in range(bxyz.shape[0]):
                    lx, ly, lz = lo[i].tolist()
                    hx, hy, hz = hi[i].tolist()
                    if lx < hx and ly < hy and lz < hz:
                        self.grid[lx:hx, ly:hy, lz:hz] = True

            occ = self.grid.sum().item()
            print(f"[Collision] Occupied {occ}/{nx * ny * nz} "
                  f"({100.0 * occ / (nx * ny * nz):.2f}%)")

    # ── 碰撞查询 ──────────────────────────────────────────────────────────

    def _pos_to_voxel(self, pos_3dgs: Tensor) -> Tensor:
        """3DGS 坐标 → 体素索引。"""
        return torch.floor((pos_3dgs - self.grid_origin) / self.voxel_size).long()

    def check_collision(self, pos_3dgs: Tensor) -> CollisionResult:
        """检查 3DGS 坐标位置是否与场景碰撞。"""
        if self.grid is None:
            return CollisionResult(collided=False)

        ix, iy, iz = self._pos_to_voxel(pos_3dgs).tolist()
        nx, ny, nz = self.grid_shape

        # 超出栅格 → 无碰撞
        if not (0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz):
            return CollisionResult(collided=False)
        if not self.grid[ix, iy, iz]:
            return CollisionResult(collided=False)

        return CollisionResult(
            collided=True,
            collision_point_3dgs=pos_3dgs.clone(),
            surface_normal=self._estimate_normal(ix, iy, iz),
            penetration_depth=self.voxel_size,
        )

    def _estimate_normal(self, ix: int, iy: int, iz: int) -> Tensor:
        """通过相邻体素占据梯度估计碰撞面法向量 (指向空闲侧)。"""
        nx, ny, nz = self.grid_shape
        g = self.grid
        normal = torch.tensor([
            float(g[max(ix-1, 0), iy, iz]) - float(g[min(ix+1, nx-1), iy, iz]),
            float(g[ix, max(iy-1, 0), iz]) - float(g[ix, min(iy+1, ny-1), iz]),
            float(g[ix, iy, max(iz-1, 0)]) - float(g[ix, iy, min(iz+1, nz-1)]),
        ], device=self.device)
        norm = normal.norm()
        return normal / norm if norm > 1e-6 else torch.tensor([0., 1., 0.], device=self.device)

# ── 碰撞响应 ────────────────────────────────────────────────────────────────

def apply_collision_response(
    dynamics,
    prev_pos_flu: Tensor,
    collision_result: CollisionResult,
    restitution: float = 0.0,
) -> None:
    """碰撞响应: 回退位置并处理速度。

    Args:
        dynamics:         动力学模型 (ContinuousPointMassModel)
        prev_pos_flu:     碰撞前 FLU 位置 (3,)
        collision_result: 碰撞检测结果
        restitution:      恢复系数 (0=非弹性, 1=完全弹性)
    """
    dynamics._state["position"][0] = prev_pos_flu

    if restitution <= 0:
        dynamics._state["velocity"][0] = torch.zeros(3, device=dynamics.device)
    else:
        # 弹性反弹: 法向量 3DGS→FLU 后沿法向反射
        n = collision_result.surface_normal
        n_flu = torch.stack([-n[2], -n[0], n[1]], dim=-1)
        v = dynamics._state["velocity"][0]
        v_n = torch.dot(v, n_flu) * n_flu
        dynamics._state["velocity"][0] = v - v_n - restitution * v_n

    dynamics._state["acceleration"][0] = torch.zeros(3, device=dynamics.device)


# ── 批量轨迹碰撞检测 ────────────────────────────────────────────────────────

def check_trajectory_collision(
    detector: CollisionDetector,
    positions_flu: Tensor,
    scene_center: Tensor,
) -> TrajectoryCollisionResult:
    """检测一架无人机 N 个位姿中是否存在碰撞 (向量化)。

    Args:
        detector:       已构建的 CollisionDetector
        positions_flu:  FLU 坐标 (N, 3)
        scene_center:   场景中心偏移 (3,)，用于 FLU→3DGS 转换
    """
    N, device = positions_flu.shape[0], positions_flu.device
    empty = TrajectoryCollisionResult(
        has_collision=False,
        collision_mask=torch.zeros(N, dtype=torch.bool, device=device),
    )
    if detector.grid is None:
        return empty

    # FLU → 3DGS
    x, y, z = positions_flu[:, 0], positions_flu[:, 1], positions_flu[:, 2]
    pos_3dgs = torch.stack([-y, z, -x], dim=-1) + scene_center  # (N, 3)

    # 体素索引 + 边界检查
    voxel = torch.floor((pos_3dgs - detector.grid_origin) / detector.voxel_size).long()
    nx, ny, nz = detector.grid_shape
    in_bounds = (
        (voxel[:, 0] >= 0) & (voxel[:, 0] < nx) &
        (voxel[:, 1] >= 0) & (voxel[:, 1] < ny) &
        (voxel[:, 2] >= 0) & (voxel[:, 2] < nz)
    )

    # 查表 (clamp 后查询，再用 in_bounds 掩码)
    ix = voxel[:, 0].clamp(0, nx - 1)
    iy = voxel[:, 1].clamp(0, ny - 1)
    iz = voxel[:, 2].clamp(0, nz - 1)
    collision_mask = in_bounds & detector.grid[ix, iy, iz]

    count = collision_mask.sum().item()
    first_idx = -1
    first_pt = None
    if count > 0:
        first_idx = collision_mask.nonzero(as_tuple=False)[0, 0].item()
        first_pt = pos_3dgs[first_idx].clone()

    return TrajectoryCollisionResult(
        has_collision=count > 0,
        collision_mask=collision_mask,
        first_collision_idx=first_idx,
        collision_count=count,
        first_collision_point_3dgs=first_pt,
    )
