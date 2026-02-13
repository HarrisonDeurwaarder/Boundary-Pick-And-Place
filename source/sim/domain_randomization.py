import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

from isaaclab.managers import EventTermCfg
from isaaclab.envs import mdp, ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

from isaaclab.assets import AssetBaseCfg

from isaacsim.core.utils import prims as prim_utils
from isaacsim.core.utils.stage import get_current_stage
from pxr import Sdf, Gf, UsdGeom, Vt

import torch
import torch.nn.functional as F
from random import choices
from typing import Sequence

import source.utils.config as config


thickness: float = config.config['scene']['room']['wall_thickness']


def randomize_room_dimensions(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfgs: Sequence[SceneEntityCfg],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
) -> None:
    '''
    Randomize the cuboid dimensions to form the closed space around an agent.
    
    Args:
        env (ManagerBasedEnvironment): Environment object containing the cuboids
        env_ids (Tensor | None): IDs of each environment instance
        asset_cfgs (Sequence[SceneEntityCfg]): Boundaries of the room (i.e. six cuboids)
        x_range (tuple[float, float]): Bounds of the room's x-axis length
        y_range (tuple[float, float]): Bounds of the room's y-axis length
        z_range (tuple[float, float]): Bounds of the room's z-axis length
    '''
    # Check if sim is running
    if env.sim.is_playing():
        raise RuntimeError('USD property modification attempted while simulation was running')
    # Get stage for USD manipulation
    stage = get_current_stage()
    
    # Resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device='cpu')
    else:
        env_ids = env_ids.cpu()
        
    # Extract the cuboid assets
    assets: list[sim_utils.CuboidCfg] = [
        env.scene[asset_cfg.name] for asset_cfg in asset_cfgs
    ]
    # Resolve environment IDs
    if env_ids is None:
        env_ids = torch.arange(
            env.scene.num_envs,
            device=asset_cfgs[0].device
        )
        
    # Get environment origins to translate assets properly
    origins = env.scene.env_origins.to(env.device)
    
    ''' Compute scales '''
    ranges: torch.Tensor = torch.cat(
        (
            torch.tensor(x_range).unsqueeze(-1).to(device='cpu'),
            torch.tensor(y_range).unsqueeze(-1).to(device='cpu'),
            torch.tensor(z_range).unsqueeze(-1).to(device='cpu'),
        ),
        dim=1
    ).unsqueeze(0)
    # Sample random room sizes
    sizes: torch.Tensor = ranges[:, 0, :] + torch.rand(
        env.num_envs, 3, device='cpu',
    ) * (ranges[:, 1, :] - ranges[:, 0, :]) # Place random values on desired range
    
    # Compute values required to adjust shape
    sizes_x: torch.Tensor = sizes.clone()
    sizes_x[:, 0] = thickness
    
    sizes_y: torch.Tensor = sizes.clone()
    sizes_y[:, 1] = thickness
    
    sizes_z: torch.Tensor = sizes.clone()
    sizes_z[:, 2] = thickness
    
    ''' Compute translations & write cuboid attributes '''
    # Sdf changeblock enables faster processing of USD properties
    with Sdf.ChangeBlock():
        for i, env_id in enumerate(env_ids):
            # Map prim to corresponding scale vect
            prim_to_scale: dict[str, tuple[float, float, float]] = {
                'wallx1': (
                    thickness,
                    sizes[i][1].item(),
                    sizes[i][2].item(),
                ),
                'wallx2': (
                    thickness,
                    sizes[i][1].item(),
                    sizes[i][2].item(),
                ),
                'wally1': (
                    sizes[i][0].item(),
                    thickness,
                    sizes[i][2].item(),
                ),
                'wally2': (
                    sizes[i][0].item(),
                    thickness,
                    sizes[i][2].item(),
                ),
                'wallz1': (
                    sizes[i][0].item(),
                    sizes[i][1].item(),
                    thickness,
                ),
                'wallz2': (
                    sizes[i][0].item(),
                    sizes[i][1].item(),
                    thickness,
                ),
            }
            # Map prim to corresponding translation vect
            prim_to_translation: dict[str, tuple[float, float, float]] = {
                'wallx1': (
                    -sizes[env_id, 0].item() / 2.0, 
                    0.0, 
                    sizes[env_id, 2].item() / 2.0,
                ),
                'wallx2': (
                    sizes[env_id, 0].item() / 2.0, 
                    0.0, 
                    sizes[env_id, 2].item() / 2.0,
                ),
                'wally1': (
                    0.0, 
                    -sizes[env_id, 1].item() / 2.0, 
                    sizes[env_id, 2].item() / 2.0,
                ),
                'wally2': (
                    0.0, 
                    sizes[env_id, 1].item() / 2.0, 
                    sizes[env_id, 2].item() / 2.0,
                ),
                'wallz1': (
                    0.0, 
                    0.0, 
                    0.0,
                ),
                'wallz2': (
                    0.0, 
                    0.0, 
                    sizes[env_id, 2].item(),
                )
            }
            # Root environment path
            env_root: str = f"/World/envs/env_{env_id}/"
            # Iter through all prims
            for prim in prim_to_translation.keys():
                prim_path = env_root + prim
                prim_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)
                
                # Get scale attribute
                scale_spec = prim_spec.GetAttributeAtPath(prim_path + '.xformOp:scale')
                # If attribute doesn't exist, create it
                if scale_spec is None:
                    scale_spec = Sdf.AttributeSpec(
                        prim_spec,
                        prim_path + '.xformOp:scale',
                        Sdf.ValueTypeNames.Double3,
                    )
                # Set new scale
                scale_spec.default = Gf.Vec3f(*prim_to_scale[prim])
                
                # Get translation attribute
                translate_spec = prim_spec.GetAttributeAtPath(prim_path + '.xformOp:translate')
                # If attribute doesn't exist, create it
                if translate_spec is None:
                    translate_spec = Sdf.AttributeSpec(
                        prim_spec,
                        prim_path + '.xformOp:translate',
                        Sdf.ValueTypeNames.Double3,
                    )
                # Set new scale
                translate_spec.default = Gf.Vec3f(*prim_to_translation[prim])
                
                # Ensure operations are done in correct order
                # If neither attributes are missing, assume order is correct
                if scale_spec is None or translate_spec is None:
                    op_order_spec = prim_spec.GetAttributeAtPath(prim_path + '.xformOpOrder')
                    if op_order_spec is None:
                        op_order_spec = Sdf.AttributeSpec(
                            prim_spec, UsdGeom.Tokens.xformOpOrder, Sdf.ValueTypeNames.TokenArray
                        )
                    op_order_spec.default = Vt.TokenArray([
                        'xformOp:translate',
                        'xformOp:orient',
                        'xformOp:scale',
                    ])
                    
                    
def randomize_object_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfgs: Sequence[SceneEntityCfg],
    distance_range_from_origin: tuple[float],
) -> None:
    '''
    Randomize a spawned asset
    
    Args:
        env (ManagerBasedEnvironment): Environment object containing the cuboids
        env_ids (Tensor | None): IDs of each environment instance
        asset_cfg (SceneEntityCfg): Targetted asset
        distance_range_from_origin (tuple[float]): Min/max distance that the asset can be translated from the origin
    '''
    
    # Check if sim is running
    if env.sim.is_playing():
        raise RuntimeError('USD property modification attempted while simulation was running')
    
    # Resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device='cpu')
    else:
        env_ids = env_ids.cpu()
    # Get stage for USD manipulation
    stage = get_current_stage()
    
    # Compute polar offset
    dist_from_origin: torch.Tensor = distance_range_from_origin[0] + torch.rand((env.num_envs, len(asset_cfgs),), device=env.sim.device) * (distance_range_from_origin[1] - distance_range_from_origin[0])
    theta: torch.Tensor = torch.rand((env.num_envs, len(asset_cfgs),), device=env.sim.device) * 2 * torch.pi
    # Convert to cart
    offset: torch.Tensor = torch.tensor([0.6, 0.0, 0.1]).repeat(env.num_envs, len(asset_cfgs), 1,)
    '''offset: torch.Tensor = torch.stack((
            dist_from_origin * torch.cos(theta),
            dist_from_origin * torch.sin(theta),
            torch.full(
                (env.num_envs, len(asset_cfgs),),
                config.config['scene']['object']['vertical_placement'],
                device=env.sim.device,
            ),
        ),
        dim=2,
    )'''
    
    # Sample raw quaternions
    raw_quat_rots: torch.Tensor = torch.rand((env.num_envs, len(asset_cfgs), 4,), device=env.sim.device)
    # Normalize quats
    quat_rots: torch.Tensor = F.normalize(raw_quat_rots, dim=2,)
    
    # Iter through all prims
    with Sdf.ChangeBlock():
        for i, asset_cfg in enumerate(asset_cfgs):
            # Resolve prim paths for spawning and cloning
            asset = env.scene[asset_cfg.name]
            prim_paths = sim_utils.find_matching_prim_paths(asset.cfg.prim_path)
            
            for j, env_id in enumerate(env_ids):
                prim_path = prim_paths[env_id]
                prim_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)
                
                # Get orient attribute
                orient_spec = prim_spec.GetAttributeAtPath(prim_path + '.xformOp:orient')
                # If attribute doesn't exist, create it
                if orient_spec is None:
                    orient_spec = Sdf.AttributeSpec(
                        prim_spec,
                        prim_path + '.xformOp:scale',
                        Sdf.ValueTypeNames.Quatd,
                    )
                # Set new orient
                orient_spec.default = Gf.Quatd(*quat_rots[j, i].tolist())
                
                # Get translation attribute
                translate_spec = prim_spec.GetAttributeAtPath(prim_path + '.xformOp:translate')
                # If attribute doesn't exist, create it
                if translate_spec is None:
                    translate_spec = Sdf.AttributeSpec(
                        prim_spec,
                        prim_path + '.xformOp:translate',
                        Sdf.ValueTypeNames.Double3,
                    )
                # Set new scale
                translate_spec.default = Gf.Vec3f(*offset[j, i].tolist())
                
                # Ensure operations are done in correct order
                # If neither attributes are missing, assume order is correct
                if orient_spec is None or translate_spec is None:
                    op_order_spec = prim_spec.GetAttributeAtPath(prim_path + '.xformOpOrder')
                    if op_order_spec is None:
                        op_order_spec = Sdf.AttributeSpec(
                            prim_spec, UsdGeom.Tokens.xformOpOrder, Sdf.ValueTypeNames.TokenArray
                        )
                    op_order_spec.default = Vt.TokenArray([
                        'xformOp:translate',
                        'xformOp:orient',
                        'xformOp:scale',
                    ])