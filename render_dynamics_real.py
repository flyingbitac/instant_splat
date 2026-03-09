#!/usr/bin/env python3
"""使用真实的diffaero连续时间质点动力学模型渲染视频，使用四元数确定相机朝向"""
from typing import List
from pathlib import Path
import os
import sys
import importlib.util as _ilu

_REPO = Path(__file__).parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch
from pytorch3d.transforms import euler_angles_to_matrix
import numpy as np
import math
import subprocess
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import argparse
from omegaconf import OmegaConf, DictConfig
from arguments import PipelineParams
from scene.gaussian_model import GaussianModel
from line_profiler import LineProfiler

# Load Renderer from gaussian_renderer/__init__3dgs.py (non-standard filename)
_spec = _ilu.spec_from_file_location("_renderer3dgs", _REPO / "gaussian_renderer" / "__init__3dgs.py")
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
Renderer = _mod.Renderer

from dynamics.pointmass import ContinuousPointMassModel
from dynamics.collision import CollisionDetector, apply_collision_response

# Repository root: the directory containing this script
REPO_ROOT = Path(__file__).parent.resolve()

def flu_to_3dgs(pos_flu: torch.Tensor):
    """FLU (X-front, Y-left, Z-up) -> 3DGS (X-right, Y-up, Z-back)"""
    x_flu, y_flu, z_flu = pos_flu.unbind(dim=-1)
    return torch.stack([-y_flu, z_flu, -x_flu], dim=-1)

def quaternion_to_euler(quat_xyzw):
    x, y, z, w = quat_xyzw.unbind(dim=-1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x**2 + y**2))
    pitch = torch.asin(2.0 * (w * y - x * z))
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))
    return torch.stack([yaw, pitch, roll], dim=-1)

class CustomCamera:
    """Camera built from explicit world->camera rotation matrix."""
    def __init__(
        self,
        fovx: float,
        fovy: float,
        width: int,
        height: int,
        device: torch.device = torch.device("cuda"),
    ):
        self.image_width = width
        self.image_height = height
        self.FoVx = fovx
        self.FoVy = fovy
        self.znear, self.zfar = 0.01, 100.0
        self.image_name = "dynamics"
        self.device = device

        tanY = torch.tan(torch.tensor(self.FoVy, device=self.device) / 2)
        tanX = torch.tan(torch.tensor(self.FoVx, device=self.device) / 2)
        self.P = torch.zeros((4, 4), device=self.device)
        self.P[0, 0] = 1.0 / tanX
        self.P[1, 1] = 1.0 / tanY
        self.P[2, 2] = self.zfar / (self.zfar - self.znear)
        self.P[2, 3] = -(self.zfar * self.znear) / (self.zfar - self.znear)
        self.P[3, 2] = 1.0

    def update_transformation(
        self,
        camera_pos_flu: torch.Tensor,
        quat_xyzw_flu: torch.Tensor,
        scene_center: torch.Tensor
    ):
        euler = quaternion_to_euler(quat_xyzw_flu) # [yaw, pitch, roll]
        # Axis mapping FLU → 3DGS: Y_flu(left)=-X_3dgs, X_flu(front)=-Z_3dgs.
        # euler_angles_to_matrix('YXZ') rotates around +X/+Z, but pitch/roll axes map to
        # -X_3dgs/-Z_3dgs, so both angles must be negated to flip the axis sign.
        # Note: in FLU with right-hand rule, positive pitch = nose DOWN (Ry(+α) tilts X toward -Z).
        euler[1] = -euler[1]  # negate pitch: Y_flu → -X_3dgs axis
        euler[2] = -euler[2]  # negate roll:  X_flu → -Z_3dgs axis
        R = euler_angles_to_matrix(euler, 'YXZ').squeeze(0)
        R[:, 1] = -R[:, 1]
        R[:, 2] = -R[:, 2]

        T = -R.T @ (flu_to_3dgs(camera_pos_flu) + scene_center)

        Rt = torch.eye(4, device=self.device)
        Rt[:3, :3] = R.T
        Rt[:3, 3] = T

        self.world_view_transform = Rt.T
        self.full_proj_transform = (self.P @ Rt).T
        self.camera_center = flu_to_3dgs(camera_pos_flu) + scene_center

def load_ply(ply_path, device=torch.device("cuda"), opacity_clip=6.0):
    """Load PLY with NaN handling and opacity clipping (from render.py)."""
    from plyfile import PlyData
    plydata = PlyData.read(ply_path)
    vd = plydata['vertex'].data

    xyz = torch.from_numpy(np.vstack([vd['x'], vd['y'], vd['z']]).T).to(device)

    dc = np.column_stack([vd['f_dc_0'], vd['f_dc_1'], vd['f_dc_2']])
    features_dc = torch.tensor(dc, dtype=torch.float32, device=device).view(-1, 1, 3)

    has_rest = 'f_rest_0' in vd.dtype.names
    sh_degree = 3 if has_rest else 0

    if has_rest:
        rest_fields = sorted([f for f in vd.dtype.names if f.startswith('f_rest_')],
                             key=lambda x: int(x.split('_')[2]))
        n = len(rest_fields) // 3
        rest = np.stack([vd[f] for f in rest_fields], axis=1)
        r = rest[:, :n, np.newaxis]
        g = rest[:, n:2*n, np.newaxis]
        b = rest[:, 2*n:, np.newaxis]
        features_rest = torch.tensor(np.concatenate([r, g, b], axis=2),
                                     dtype=torch.float32, device=device)
    else:
        features_rest = torch.zeros((len(xyz), 0, 3), dtype=torch.float32, device=device)

    opacity_raw = np.nan_to_num(vd['opacity'].reshape(-1, 1).astype(np.float32),
                                nan=0.0, posinf=10.0, neginf=-10.0)
    opacity = torch.tensor(np.clip(opacity_raw, -opacity_clip, opacity_clip),
                           dtype=torch.float32, device=device)

    scales = torch.tensor(np.vstack([vd['scale_0'], vd['scale_1'], vd['scale_2']]).T,
                          dtype=torch.float32, device=device)
    rotations = torch.tensor(np.vstack([vd['rot_0'], vd['rot_1'], vd['rot_2'], vd['rot_3']]).T,
                             dtype=torch.float32, device=device)

    gaussians = GaussianModel(sh_degree=sh_degree)
    gaussians._xyz = torch.nn.Parameter(xyz)
    gaussians._features_dc = torch.nn.Parameter(features_dc)
    gaussians._features_rest = torch.nn.Parameter(features_rest)
    gaussians._opacity = torch.nn.Parameter(opacity)
    gaussians._scaling = torch.nn.Parameter(scales)
    gaussians._rotation = torch.nn.Parameter(rotations)

    center = xyz.mean(dim=0)
    radius = torch.norm(xyz - center, dim=1).max().item()
    print(f"Loaded {len(xyz)} gaussians, center={center}, radius={radius:.2f}")
    return gaussians, center, radius

def main():
    parser = argparse.ArgumentParser()
    # Use relative path: World.ply in the same directory as this script
    parser.add_argument("--ply_path", default=str(REPO_ROOT / "World.ply"))
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--fov", type=float, default=80.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--output", type=str, default="outputs/trajectory_real")
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--draw", action="store_true")
    # 碰撞检测参数
    parser.add_argument("--collision", action="store_true", help="Enable collision detection")
    parser.add_argument("--voxel_size", type=float, default=0.05, help="Voxel grid resolution (m)")
    parser.add_argument("--drone_radius", type=float, default=0.15, help="Drone collision sphere radius (m)")
    parser.add_argument("--opacity_thresh", type=float, default=0.5, help="Opacity threshold for obstacles")
    parser.add_argument("--restitution", type=float, default=0.0, help="Bounce coefficient (0=stop, 1=full bounce)")
    args = parser.parse_args()

    # GPU selection
    if args.gpu < 0:
        try:
            r = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'],
                               capture_output=True, text=True, timeout=5)
            gpu_id = int(np.argmax([int(x) for x in r.stdout.strip().split('\n')])) if r.returncode == 0 else 0
        except Exception:
            gpu_id = 0
    else:
        gpu_id = args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device('cuda:0')

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    dt = 1.0 / args.fps
    num_steps = int(args.duration * args.fps)

    # Load dynamics: use relative path to diffaero config
    # Assumes diffaero is a sibling directory to 3dgs_dynamics
    dynamics_config_path = REPO_ROOT / "dynamics" / "pmc.yaml"
    cfg: DictConfig = OmegaConf.load(str(dynamics_config_path))  # type: ignore
    cfg.n_envs = 1
    cfg.n_agents = 1
    cfg.dt = dt
    cfg.action_frame = "local"

    print(f"Dynamics config: action_frame={cfg.action_frame}, solver={cfg.solver_type}, g={cfg.g}")
    dynamics = ContinuousPointMassModel(cfg, device)

    # Initialize inside the scene: FLU(-1.5, -0.5, 0.25) is in a street between buildings
    dynamics.reset_idx(
        env_idx=torch.tensor([0], device=device), 
        p_new=torch.tensor([-1.5, -0.5, 0.25], device=device)
    )

    # Fly forward (+x_flu) through the street, then turn left into another corridor
    # Stronger forward thrust to reach buildings within the scene
    action_forward = torch.tensor([[3.0, 0, 9.81]], device=device)
    action_left = torch.tensor([[1.0, 1.5, 9.81]], device=device)

    # Load PLY scene (moved before simulation so collision detector can be built)
    gaussians, scene_center, scene_radius = load_ply(args.ply_path, device)

    # Build collision detector
    collision_detector = None
    if args.collision:
        collision_detector = CollisionDetector(
            voxel_size=args.voxel_size,
            opacity_threshold=args.opacity_thresh,
            drone_radius=args.drone_radius,
            device=device,
        )
        collision_detector.build_from_gaussians(
            xyz=gaussians.get_xyz.detach(),
            opacity_raw=gaussians._opacity.detach(),
            scaling_raw=gaussians._scaling.detach(),
        )

    # Simulate trajectory (with collision detection)
    collision_count = 0
    positions_flu: List[torch.Tensor] = []
    quats: List[torch.Tensor] = []
    for i in range(num_steps):
        pos = dynamics.p[0].clone()
        quat_xyzw = dynamics.q[0].clone()
        positions_flu.append(pos)
        quats.append(quat_xyzw)

        prev_pos_flu = pos.clone()

        if i < num_steps // 6 or i >= 5 * num_steps // 6:
            dynamics.step(action_forward)
        else:
            dynamics.step(action_left)

        # Collision detection (after dynamics step)
        if collision_detector is not None:
            new_pos_flu = dynamics.p[0]
            new_pos_3dgs = flu_to_3dgs(new_pos_flu) + scene_center
            result = collision_detector.check_collision(new_pos_3dgs)
            if result.collided:
                collision_count += 1
                apply_collision_response(
                    dynamics, prev_pos_flu, result,
                    restitution=args.restitution,
                )
                if collision_count <= 10 or collision_count % 50 == 0:
                    print(f"  [Collision #{collision_count}] step={i}, "
                          f"pos_flu={prev_pos_flu.cpu().numpy()}")

    if collision_detector is not None:
        print(f"[CollisionDetector] Total collisions: {collision_count}/{num_steps} steps")

    pos_flu_tensor = torch.stack(positions_flu, dim=0)  # (N, 3)
    quat_xyzw_flu_tensor = torch.stack(quats, dim=0)  # (N, 4)

    print(f"Trajectory FLU: {positions_flu[0]} -> {positions_flu[-1]}")
    print(f"Final quaternion (FLU): {quats[-1]}")

    if args.draw:
        # Extract Euler angles (ZYX: yaw, pitch, roll) from rotation matrices for plotting
        euler_all = quaternion_to_euler(quat_xyzw_flu_tensor).cpu().numpy()  # (N, 3): yaw, pitch, roll
        euler_deg = np.degrees(euler_all)
        t = np.arange(num_steps) * dt

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for ax, data, label in zip(axes,
                                    [euler_deg[:, 1], euler_deg[:, 0], euler_deg[:, 2]],
                                    ['Pitch (deg)', 'Yaw (deg)', 'Roll (deg)']):
            ax.plot(t, data)
            ax.set_ylabel(label)
            ax.grid(True)
        axes[-1].set_xlabel('Time (s)')
        fig.suptitle('Euler Angles (FLU ZYX)')
        fig.tight_layout()
        fig.savefig(output_path / 'euler_angles.png', dpi=150)
        plt.close(fig)
        print(f"Euler angles plot saved to {output_path / 'euler_angles.png'}")

    # Pipeline
    pipe_parser = argparse.ArgumentParser()
    pipeline = PipelineParams(pipe_parser)
    pipe_parser.parse_args([])
    pipeline.antialiasing = True

    bg = torch.zeros(3, dtype=torch.float32, device=device)

    fovx = math.radians(args.fov)
    fovy = 2.0 * np.arctan(np.tan(fovx * 0.5) * (args.height / args.width))

    video_path = output_path / "trajectory_real.mp4"
    print(f"Rendering and encoding {num_steps} frames to {video_path}...")
    writer = imageio.get_writer(
        str(video_path),
        fps=args.fps,
        codec="libx264",
        pixelformat="yuv420p",
        ffmpeg_params=["-crf", "18"],
    )

    camera = CustomCamera(fovx, fovy, args.width, args.height, torch.device("cuda"))
    renderer = Renderer(pc=gaussians, pipe=pipeline, bg_color=bg)
    with torch.no_grad():
        try:
            for i in range(num_steps):
                camera.update_transformation(pos_flu_tensor[i], quat_xyzw_flu_tensor[i], scene_center)
                # result = render(viewpoint_camera=camera, pc=gaussians, pipe=pipeline,
                #                 bg_color=bg, scaling_modifier=1.0,
                #                 separate_sh=False, override_color=None, use_trained_exp=False)
                result = renderer.render(viewpoint_camera=camera, scaling_modifier=1.0,
                                separate_sh=False, override_color=None, use_trained_exp=False)
                img = (result["render"].permute(1, 2, 0).clamp(0, 1) * 255).cpu().numpy().astype(np.uint8)
                writer.append_data(img)
                if (i + 1) % 30 == 0:
                    print(f"  [{i+1}/{num_steps}] pos_flu={pos_flu_tensor[i]}, pos_3dgs={flu_to_3dgs(pos_flu_tensor[i]) + scene_center}")
        finally:
            writer.close()

    print(f"Done: {video_path} ({video_path.stat().st_size / 1048576:.1f} MB)")


if __name__ == "__main__":
    profiler = LineProfiler()
    main = profiler(main)
    profiler.add_function(CustomCamera.__init__)
    main()
    profiler.print_stats(output_unit=1e-3)
