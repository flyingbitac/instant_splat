#!/usr/bin/env python3
"""
批量渲染脚本：使用 rasterize_gaussians_batch_forward 一次渲染多相机
cam = env = action-seq，每个cam对应一条独立的action序列，输出C条视频
"""
import sys
from pathlib import Path
import argparse
import time
import json
import os
_REPO = Path(__file__).parent

if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import importlib.util as _ilu

import torch
from torch import Tensor
import numpy as np
import math
from pytorch3d.transforms import euler_angles_to_matrix
import imageio.v2 as imageio
from omegaconf import OmegaConf

from gaussian_renderer.__init__3dgs import Renderer
from arguments import PipelineParams
from scene.gaussian_model import GaussianModel
from utils.sh_utils import SH2RGB, C0
from dynamics.pointmass import ContinuousPointMassModel
from dynamics.utils.math import mvp

# Load Renderer from gaussian_renderer/__init__3dgs.py (non-standard filename)
# _spec = _ilu.spec_from_file_location("_renderer3dgs", _REPO / "gaussian_renderer" / "__init__3dgs.py")
# _mod = _ilu.module_from_spec(_spec)
# _spec.loader.exec_module(_mod)
# Renderer = _mod.Renderer


def load_ply_ref(ply_path: str, device: torch.device):
    """Load a 3DGS Gaussian PLY into a GaussianModel and return (model, center, radius)."""
    from plyfile import PlyData
    plydata = PlyData.read(ply_path)
    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    n_rest = len(extra_f_names)
    sh_degree = int(round(((n_rest + 3) / 3) ** 0.5)) - 1

    gaussians = GaussianModel(sh_degree=sh_degree)
    gaussians.load_ply(ply_path)

    xyz = gaussians._xyz.data
    center = xyz.mean(dim=0)
    radius = torch.norm(xyz - center, dim=1).max().item()
    return gaussians, center.to(device), radius

REPO_ROOT = Path(__file__).parent.resolve()

def flu_to_3dgs(pos_flu):
    x, y, z = pos_flu.unbind(-1)
    return torch.stack([-y, z, -x], -1)

def quaternion_to_euler(q):
    x, y, z, w = q.unbind(-1)
    roll = torch.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    pitch = torch.asin(2*(w*y-x*z))
    yaw = torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    return torch.stack([yaw, pitch, roll], -1)

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
    
    def _calculate_transformation(
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

        world_view_transform = Rt.T
        full_proj_transform = (self.P @ Rt).T
        camera_center = flu_to_3dgs(camera_pos_flu) + scene_center
        return world_view_transform, full_proj_transform, camera_center

    def update_transformation(
        self,
        camera_pos_flu: torch.Tensor,
        quat_xyzw_flu: torch.Tensor,
        scene_center: torch.Tensor
    ):
        world_view_transform, full_proj_transform, camera_center = self._calculate_transformation(camera_pos_flu, quat_xyzw_flu, scene_center)
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        self.camera_center = camera_center

    def update_transformation_batch(
        self,
        pos_flu: Tensor,
        quat_flu: Tensor,
        scene_center: Tensor
    ):
        e = quaternion_to_euler(quat_flu)
        e[..., 1], e[..., 2] = -e[..., 1], -e[..., 2]
        R = euler_angles_to_matrix(e, 'YXZ')
        R[..., 1], R[..., 2] = -R[..., 1], -R[..., 2]
        T = -torch.matmul(R.transpose(-1, -2), (flu_to_3dgs(pos_flu) + scene_center.unsqueeze(0)).unsqueeze(-1)).squeeze(-1)
        Rt = torch.eye(4, device=self.device).unsqueeze(0).expand(R.shape[0], -1, -1).clone()
        Rt[:, :3] = torch.concat([R.transpose(-2, -1), T.unsqueeze(-1)], dim=-1)
        self.world_view_transform = Rt.transpose(-2, -1)
        self.full_proj_transform = torch.matmul(self.P.unsqueeze(0), Rt).transpose(-2, -1)
        self.camera_center = flu_to_3dgs(pos_flu) + scene_center.unsqueeze(0)

def load_ply_flexible(ply_path: str, device: torch.device):
    """兼容两类PLY：
    1) 3DGS高斯PLY（含 f_dc_0 / scale_* / rot_* / opacity）
    2) 普通点云PLY（x,y,z,red,green,blue）
    """
    from plyfile import PlyData
    vd = PlyData.read(ply_path)['vertex'].data
    names = set(vd.dtype.names)

    if 'f_dc_0' in names:
        return load_ply_ref(ply_path, device)

    xyz = torch.from_numpy(np.vstack([vd['x'], vd['y'], vd['z']]).T.astype(np.float32)).to(device)

    if {'red', 'green', 'blue'}.issubset(names):
        rgb = np.vstack([vd['red'], vd['green'], vd['blue']]).T.astype(np.float32) / 255.0
    elif {'r', 'g', 'b'}.issubset(names):
        rgb = np.vstack([vd['r'], vd['g'], vd['b']]).T.astype(np.float32) / 255.0
    else:
        rgb = np.full((xyz.shape[0], 3), 0.7, dtype=np.float32)

    dc = (rgb - 0.5) / float(C0)
    features_dc = torch.tensor(dc, dtype=torch.float32, device=device).view(-1, 1, 3)
    features_rest = torch.zeros((xyz.shape[0], 0, 3), dtype=torch.float32, device=device)

    center = xyz.mean(dim=0)
    radius = torch.norm(xyz - center, dim=1).max().item()
    base_scale = max(radius * 0.0015, 1e-4)
    log_s = float(np.log(base_scale))
    scales = torch.full((xyz.shape[0], 3), log_s, dtype=torch.float32, device=device)

    rotations = torch.zeros((xyz.shape[0], 4), dtype=torch.float32, device=device)
    rotations[:, 0] = 1.0

    opacity = torch.full((xyz.shape[0], 1), 2.1972246, dtype=torch.float32, device=device)

    gaussians = GaussianModel(sh_degree=0)
    gaussians._xyz = torch.nn.Parameter(xyz)
    gaussians._features_dc = torch.nn.Parameter(features_dc)
    gaussians._features_rest = torch.nn.Parameter(features_rest)
    gaussians._opacity = torch.nn.Parameter(opacity)
    gaussians._scaling = torch.nn.Parameter(scales)
    gaussians._rotation = torch.nn.Parameter(rotations)

    print(f'[compat] Loaded point-cloud PLY as gaussians: N={xyz.shape[0]}, center={center}, radius={radius:.3f}, base_scale={base_scale:.6f}')
    return gaussians, center, radius

def build_action_sequences(n_cams: int, n_steps: int, device: torch.device) -> torch.Tensor:
    """
    构建动作序列 [T, C, 3]
    与 render_dynamics_real.py 相同：对所有 cam 使用相同的 action_forward / action_left
    action_forward = [1.5, 0, 9.81]
    action_left = [0.2, 0.3, 9.81]
    """
    action_forward = [1.5, 0, 9.81]
    action_left = [0.2, 0.3, 9.81]
    
    all_actions = []
    for cam_idx in range(n_cams):
        cam_actions = []
        for t in range(n_steps):
            if t < n_steps // 6 or t >= 5 * n_steps // 6:
                a = action_forward
            else:
                a = action_left
            cam_actions.append(a)
        all_actions.append(cam_actions)
    
    # shape: [C, T, 3] -> transpose to [T, C, 3]
    actions = torch.tensor(all_actions, dtype=torch.float32, device=device).transpose(0, 1)
    return actions


def simulate_trajectories(dyn, actions: torch.Tensor):
    """
    模拟C条轨迹
    actions: [T, C, 3]
    Returns: pos[T, C, 3], quat[T, C, 4]
    """
    n_steps = actions.shape[0]
    n_cams = actions.shape[1]
    pos, quat = [], []
    for t in range(n_steps):
        p, q = dyn.p, dyn.q
        if p.dim() == 1:
            p, q = p.unsqueeze(0), q.unsqueeze(0)
        pos.append(p.cpu())
        quat.append(q.cpu())
        dyn.step(actions[t])
    return torch.stack(pos, 0), torch.stack(quat, 0)

def render_batch(
    gaussians,
    bg: Tensor,
    fovx: float,
    fovy: float,
    pos_t: Tensor,
    quat_t: Tensor,
    scene_center: Tensor,
    out_dir: Path,
    n_cams: int,
    n_steps: int,
    width: int,
    height: int,
    device: torch.device,
    fps: int = 30
):
    """
    批量渲染模式：使用 rasterize_gaussians_batch_forward 一次渲染所有相机
    
    check_align: 若为True，在第0帧进行batch vs串行对齐测试
    debug_frame0_stats: 若为True，在第0帧打印调试统计信息
    """
    writers = []
    for c in range(n_cams):
        w = imageio.get_writer(str(out_dir / f'trajectory_real_cam_{c:03d}.mp4'), fps=fps, codec='libx264', pixelformat='yuv420p', ffmpeg_params=['-crf', '18'])
        writers.append(w)
    
    pipe_parser = argparse.ArgumentParser()
    pipeline = PipelineParams(pipe_parser)
    pipe_parser.parse_args([])
    pipeline.antialiasing = True
    camera = CustomCamera(fovx, fovy, width, height, device)
    renderer = Renderer(pc=gaussians, pipe=pipeline, bg_color=bg)
    
    t_first_render = None
    frame_times = []
    
    with torch.no_grad():
        for f in range(n_steps):
            t_frame_start = time.time()
            camera.update_transformation_batch(
                pos_t[f].to(device),
                quat_t[f].to(device),
                scene_center.to(device)
            )
            result = renderer.render_batch(
                viewpoint_camera=camera,
                scaling_modifier=1.0,
                separate_sh=False,
                override_color=None,
                use_trained_exp=False
            )
            for c in range(n_cams):
                img = result["render"][c].permute(1, 2, 0).clamp(0, 1).mul(255).cpu().numpy().astype(np.uint8)
                writers[c].append_data(img)
            
            frame_times.append(time.time() - t_frame_start)
            
            if (f + 1) % max(1, n_steps // 10) == 0:
                print(f'    frame {f+1}/{n_steps}')
    
    for w in writers:
        w.close()
    
    avg_frame_time = sum(frame_times) / len(frame_times) if frame_times else 0
    print(f'    [batch] avg frame time: {avg_frame_time*1000:.1f}ms')
    
    return t_first_render, avg_frame_time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ply_path', default=str(REPO_ROOT/'World.ply'))
    p.add_argument('--duration', type=float, default=15.0)
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--fov', type=float, default=80.0)
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--height', type=int, default=720)
    p.add_argument('--output', default='outputs/trajectory_real_batch')
    p.add_argument('--num_cameras', type=int, default=1, help='C: cam数量=env数量=action序列数')
    p.add_argument('--debug_compare', action='store_true', help='Enable deterministic debug compare mode')
    p.add_argument('--debug_log', type=str, default='', help='Optional path to debug log file')
    args = p.parse_args()

    if args.debug_compare:
        print('[DEBUG] Enabling deterministic mode for reproducible comparison')
        torch.manual_seed(42)
        np.random.seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        print('[DEBUG] Deterministic settings applied: torch/np seeds=42, cudnn.deterministic=True')

    device = torch.device('cuda:0')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    debug_logger = None
    if args.debug_log:
        debug_log_path = Path(args.debug_log)
        debug_log_path.parent.mkdir(parents=True, exist_ok=True)
        debug_logger = open(debug_log_path, 'w')
        print(f'[DEBUG] Logging to {args.debug_log}')
    n_steps = int(args.duration * args.fps)
    n_cams = args.num_cameras

    print(f'=== Batch Render: C={n_cams}, frames={n_steps}, res={args.width}x{args.height} ===')
    if n_steps <= 0:
        print('ERROR: need at least 1 frame')
        return

    t0_total = time.time()

    print('[1] Init dynamics...')
    t1 = time.time()
    cfg = OmegaConf.load(str(REPO_ROOT/'dynamics'/'pmc.yaml'))
    cfg.n_envs = n_cams
    cfg.n_agents = 1
    cfg.dt = 1.0 / args.fps
    cfg.action_frame = 'local'
    dyn = ContinuousPointMassModel(cfg, device)
    for e in range(n_cams):
        dyn.reset_idx(torch.tensor([e], device=device), torch.tensor([-2., 0., 0.], device=device))
    print(f'    dyn init in {time.time()-t1:.2f}s')

    print('[2] Build action sequences and simulate...')
    t2 = time.time()
    actions = build_action_sequences(n_cams, n_steps, device)
    print(f'    actions shape: {actions.shape}')
    pos_t, quat_t = simulate_trajectories(dyn, actions)
    print(f'    pos={pos_t.shape}, quat={quat_t.shape}, sim in {time.time()-t2:.2f}s')

    print('[3] Load PLY...')
    t3 = time.time()
    gsm, scene_center, scene_radius = load_ply_flexible(args.ply_path, device)

    print('[4] Init batch renderer...')
    t4 = time.time()
    bg = torch.zeros(3, device=device)
    print(f'    renderer init in {time.time()-t4:.2f}s')

    fovx = math.radians(args.fov)
    fovy = 2 * np.arctan(np.tan(fovx * 0.5) * (args.height / args.width))

    print(f'[5] Batch render ({n_steps} frames)...')
    t5 = time.time()
    t_first, avg_frame = render_batch(gsm, bg, fovx, fovy, pos_t, quat_t, scene_center, out, n_cams, n_steps, args.width, args.height, device, fps=args.fps)
    t_render = time.time() - t5

    if debug_logger:
        debug_logger.close()
        print(f'[DEBUG] Closed debug log: {args.debug_log}')

    print('[Finish]')
    total_time = time.time() - t0_total

    print(f'=== Timing Summary ===')
    print(f'  Total: {total_time:.2f}s')
    print(f'  Render (all frames): {t_render:.2f}s')
    print(f'  Avg frame time: {avg_frame*1000:.1f}ms')
    if t_first:
        print(f'  First frame render: {t_first - t5:.2f}s')
    print(f'  Avg fps: {n_steps * n_cams / t_render:.1f}')

    print('\nOutput files:')
    for c in range(n_cams):
        mp4 = out / f'trajectory_real_cam_{c:03d}.mp4'
        if mp4.exists():
            size_bytes = mp4.stat().st_size
            size_mb = size_bytes / 1e6
            print(f'  {mp4} ({size_bytes} bytes, {size_mb:.6f} MB)')


if __name__ == '__main__':
    main()