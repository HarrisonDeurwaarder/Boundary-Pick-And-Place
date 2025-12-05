import isaaclab.sim as sim_utils
import torch
from utils.config import HPARAMS


thickness: float = HPARAMS['scene']['room']['wall_thickness']


def get_rects(dims: torch.Tensor,) -> tuple[sim_utils.CuboidCfg]:
    '''
    Converts randomized room dimensions into room boundary (cuboid) prims.
    
    Args:
        dim (torch.Tensor): (x, y, z) room dimensions
        
    Returns:
        tuple[CuboidCfg, ...]: A tuple containing:
            - boundary_x: Primative cuboid for the x-dim walls
            - boundary_y: Primative cuboid for the y-dim walls
            - boundary_z: Primative cuboid for the z-dim walls
    '''
    get_boundary = lambda scale: sim_utils.CuboidCfg(
        size=scale,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            kinematic_enabled=True,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.8, dynamic_friction=0.6, restitution=0.1,
        ),
    )
    boundary_x: sim_utils.CuboidCfg = get_boundary((thickness, float(dims[1].item()), float(dims[2].item())))
    boundary_y: sim_utils.CuboidCfg = get_boundary((float(dims[0].item()), thickness, float(dims[2].item())))
    boundary_z: sim_utils.CuboidCfg = get_boundary((float(dims[0].item()), float(dims[1].item()), thickness))
    
    return boundary_x, boundary_y, boundary_z