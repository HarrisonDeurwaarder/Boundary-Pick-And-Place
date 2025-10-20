import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

from isaaclab.managers import EventTermCfg
from isaaclab.envs import mdp
from isaaclab.scene import SceneEntityCfg


@configclass
class EventCfg:
    '''
    Event config. Handles most domain randomization
    '''
    
    ### PANDA RANDOMIZATION ###
    panda_phys_material: EventTermCfg = EventTermCfg(
        func = mdp.randomize_rigid_body_material,
        mode='reset',
        params={
            'asset_cfg': SceneEntityCfg('panda', body_names='.*'),
            'static_friction_range': (0.5, 1.5),
            'dynamic_friction_range': (1.0, 1.0),
            'restitution_range': (1.0, 1.0),
            'num_buckets': 250,
        },
    )
    panda_stiffness_damping: EventTermCfg = EventTermCfg(
        func=mdp.randomize_actuator_gains,
        mode='reset',
        params={
            'asset_cfg': SceneEntityCfg('panda', body_names='.*'),
            'stiffness_distribution_params': (0.75, 1.5),
            'damping_distribution_params': (0.3, 3.0),
            'operation': 'scale',
            'distribution': 'log_uniform',
        },
    )
    panda_mass: EventTermCfg = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode='reset',
        params={
            'asset_cfg': SceneEntityCfg('panda', body_names='.*'),
            'mass_distribution_params': (0.85, 1.15),
            'operation': 'scale',
            'distribution': 'log_uniform',
        },
    )
    panda_mass: EventTermCfg = EventTermCfg(
        func=mdp.randomize_joint_parameters,
        mode='reset',
        params={
            'asset_cfg': SceneEntityCfg('panda', body_names='.*'),
            'friction_distribution_params': (0.85, 1.15),
            'armature_distribution_params': (0.75, 1.25),
            'operation': 'scale',
            'distribution': 'log_uniform',
        },
    )
    panda_ext_force_torque: EventTermCfg = EventTermCfg(
        func=mdp.randomize_joint_parameters,
        mode='reset',
        params={
            'asset_cfg': SceneEntityCfg('panda', body_names='.*'),
            'force_range': (-2.0, 2.0),
            'torque_range': (-0.4, 0.4),
        },
    )
    panda_joints_scale: EventTermCfg = EventTermCfg(
        func=mdp.randomize_joint_parameters,
        mode='reset',
        params={
            'asset_cfg': SceneEntityCfg('panda', body_names='.*'),
            'position_range': (0.7, 1.3),
            'velocity_range': (0.5, 1.5),
        },
    )
    
    ### PHYSICS RANDOMIZATION ###
    gravity: EventTermCfg = EventTermCfg(
        func=mdp.randomize_physics_scene_gravity,
        mode='reset',
        params={
            'gravity_distribution_params': ([0.0, 0.0, 0.0], [0.0, 0.0, 0.4]),
            'operation': 'add',
            'distribution': 'gaussian',
        },
    )
    
    ### OBJECT RANDOMIZATION ###
    '''
    Note:
    Use randomize_visual_color, randomize_visual_texture_material, randomize_rigid_body_material 
    '''