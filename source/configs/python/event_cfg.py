import torch

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

from isaaclab.managers import EventTermCfg
from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from source.sim.domain_randomization import randomize_object_pose, randomize_room_dimensions
from source.utils.assets import get_assets
import source.utils.config as config


'''IMAGE_EXTS: tuple[str] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr", ".hdr", ".bmp")
texture_paths: list[str] = get_assets(ISAACLAB_NUCLEUS_DIR, IMAGE_EXTS)'''

@configclass
class EventCfg:
    '''
    Event config. Handles most domain randomization
    '''
    
    '''### ROBOT RANDOMIZATION ###
    robot_phys_material: EventTermCfg = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode='reset',
        params={
            'asset_cfg': SceneEntityCfg('robot', body_names='.*'),
            'static_friction_range': config.config['event']['robot']['material']['static_friction_range'],
            'dynamic_friction_range': config.config['event']['robot']['material']['dynamic_friction_range'],
            'restitution_range': config.config['event']['robot']['material']['restitution_range'],
            'num_buckets': config.config['event']['robot']['material']['num_buckets'],
        },
    )
    robot_stiffness_damping: EventTermCfg = EventTermCfg(
        func=mdp.randomize_actuator_gains,
        mode='reset',
        params={
            'asset_cfg': SceneEntityCfg('robot', body_names='.*'),
            'stiffness_distribution_params': config.config['event']['robot']['actuator_gains']['stiffness_distribution_params'],
            'damping_distribution_params': config.config['event']['robot']['actuator_gains']['damping_distribution_params'],
            'operation': 'scale',
            'distribution': 'log_uniform',
        },
    )
    robot_mass: EventTermCfg = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode='reset',
        params={
            'asset_cfg': SceneEntityCfg('robot', body_names='.*'),
            'mass_distribution_params': config.config['event']['robot']['mass']['mass_distribution_params'],
            'operation': 'scale',
            'distribution': 'log_uniform',
        },
    )
    robot_joint_params: EventTermCfg = EventTermCfg(
        func=mdp.randomize_joint_parameters,
        mode='reset',
        params={
            'asset_cfg': SceneEntityCfg('robot', body_names='.*'),
            'friction_distribution_params': config.config['event']['robot']['joint_params']['friction_distribution_params'],
            'armature_distribution_params': config.config['event']['robot']['joint_params']['armature_distribution_params'],
            'operation': 'scale',
            'distribution': 'log_uniform',
        },
    )
    robot_ext_force_torque: EventTermCfg = EventTermCfg(
        func=mdp.apply_external_force_torque,
        mode='reset',
        params={
            'asset_cfg': SceneEntityCfg('robot', body_names='.*'),
            'force_range': config.config['event']['robot']['force_torque']['force_range'],
            'torque_range': config.config['event']['robot']['force_torque']['torque_range'],
        },
    )
    robot_joints_scale: EventTermCfg = EventTermCfg(
        func=mdp.reset_joints_by_scale,
        mode='reset',
        params={
            'asset_cfg': SceneEntityCfg('robot', body_names='.*'),
            'position_range': config.config['event']['robot']['reset_joints']['position_range'],
            'velocity_range': config.config['event']['robot']['reset_joints']['velocity_range'],
        },
    )
    
    ### PHYSICS RANDOMIZATION ###
    gravity: EventTermCfg = EventTermCfg(
        func=mdp.randomize_physics_scene_gravity,
        mode='reset',
        params={
            'gravity_distribution_params': config.config['event']['physics']['gravity']['gravity_distribution_params'],
            'operation': 'add',
            'distribution': 'gaussian',
        },
    )'''
    
    '''### OBJECT & ROOM RANDOMIZATION ###
    room_scale: EventTermCfg = EventTermCfg(
        func=randomize_room_dimensions,
        mode='reset',
        params={
            'asset_cfgs': map(
                lambda prim: SceneEntityCfg(prim),
                ['wall_x1', 'wall_x2', 'wall_y1', 'wall_y2', 'wall_z1', 'wall_z2'],
            ),
            'x_range': config.config['scene']['room']['x_range'],
            'y_range': config.config['scene']['room']['y_range'],
            'z_range': config.config['scene']['room']['z_range'],
        },
    )'''
    
    object_pose: EventTermCfg = EventTermCfg(
        func=randomize_object_pose,
        mode='reset',
        params={
            'asset_cfgs': [SceneEntityCfg(f'object_{i}') for i in range(config.config['scene']['object']['n_assets'])],
            'distance_range_from_origin': tuple(config.config['scene']['object']['distance_range_from_origin']),
        }
    )
    
    '''visual_color_x1: EventTermCfg = EventTermCfg(
        func=mdp.randomize_visual_color,
        mode='reset',
        params={
            'asset_cfg': SceneEntityCfg('wall_x1'),
            'event_name': 'randomize_x1_color',
            'colors': {'r': (-1.0, 1.0), 'g': (-1.0, 1.0),'b': (-1.0, 1.0)},
        }
    )
    
    visual_color_x2: EventTermCfg = EventTermCfg(
        func=mdp.randomize_visual_color,
        mode='reset',
        params={
            'asset_cfg': SceneEntityCfg('wall_x2'),
            'event_name': 'randomize_x2_color',
            'colors': {'r': (-1.0, 1.0), 'g': (-1.0, 1.0),'b': (-1.0, 1.0)},
        }
    )
    
    visual_color_y1: EventTermCfg = EventTermCfg(
        func=mdp.randomize_visual_color,
        mode='reset',
        params={
            'asset_cfg': SceneEntityCfg('wall_y1'),
            'event_name': 'randomize_y1_color',
            'colors': {'r': (-1.0, 1.0), 'g': (-1.0, 1.0),'b': (-1.0, 1.0)},
        }
    )
    
    visual_color_y2: EventTermCfg = EventTermCfg(
        func=mdp.randomize_visual_color,
        mode='reset',
        params={
            'asset_cfg': SceneEntityCfg('wall_y2'),
            'event_name': 'randomize_y2_color',
            'colors': {'r': (-1.0, 1.0), 'g': (-1.0, 1.0),'b': (-1.0, 1.0)},
        }
    )
    
    visual_color_z1: EventTermCfg = EventTermCfg(
        func=mdp.randomize_visual_color,
        mode='reset',
        params={
            'asset_cfg': SceneEntityCfg('wall_z1'),
            'event_name': 'randomize_z1_color',
            'colors': {'r': (-1.0, 1.0), 'g': (-1.0, 1.0),'b': (-1.0, 1.0)},
        }
    )
    
    visual_color_z2: EventTermCfg = EventTermCfg(
        func=mdp.randomize_visual_color,
        mode='reset',
        params={
            'asset_cfg': SceneEntityCfg('wall_z2'),
            'event_name': 'randomize_z2_color',
            'colors': {'r': (-1.0, 1.0), 'g': (-1.0, 1.0),'b': (-1.0, 1.0)},
        }
    )'''
    
    '''visual_texture_x1: EventTermCfg = EventTermCfg(
        func=mdp.randomize_visual_texture_material,
        mode='reset',
        params={
            'asset_cfg': SceneEntityCfg('wall_x1'),
            'event_name': 'randomize_x1_texture',
            'texture_paths': texture_paths,
            'texture_rotation': (0.0, 2 * torch.pi),
        }
    )
    
    visual_texture_x2: EventTermCfg = EventTermCfg(
        func=mdp.randomize_visual_texture_material,
        mode='reset',
        params={
            'asset_cfg': SceneEntityCfg('wall_x2'),
            'event_name': 'randomize_x2_texture',
            'texture_paths': texture_paths,
            'texture_rotation': (0.0, 2 * torch.pi),
        }
    )
    
    visual_texture_y1: EventTermCfg = EventTermCfg(
        func=mdp.randomize_visual_texture_material,
        mode='reset',
        params={
            'asset_cfg': SceneEntityCfg('wall_y1'),
            'event_name': 'randomize_y1_texture',
            'texture_paths': texture_paths,
            'texture_rotation': (0.0, 2 * torch.pi),
        }
    )
    
    visual_texture_y2: EventTermCfg = EventTermCfg(
        func=mdp.randomize_visual_texture_material,
        mode='reset',
        params={
            'asset_cfg': SceneEntityCfg('wall_y2'),
            'event_name': 'randomize_y2_texture',
            'texture_paths': texture_paths,
            'texture_rotation': (0.0, 2 * torch.pi),
        }
    )
    
    visual_texture_z1: EventTermCfg = EventTermCfg(
        func=mdp.randomize_visual_texture_material,
        mode='reset',
        params={
            'asset_cfg': SceneEntityCfg('wall_z1'),
            'event_name': 'randomize_z1_texture',
            'texture_paths': texture_paths,
            'texture_rotation': (0.0, 2 * torch.pi),
        }
    )
    
    visual_texture_z2: EventTermCfg = EventTermCfg(
        func=mdp.randomize_visual_texture_material,
        mode='reset',
        params={
            'asset_cfg': SceneEntityCfg('wall_z2'),
            'event_name': 'randomize_z2_texture',
            'texture_paths': texture_paths,
            'texture_rotation': (0.0, 2 * torch.pi),
        }
    )'''