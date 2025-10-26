import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

from isaaclab.managers import EventTermCfg
from isaaclab.envs import mdp
from isaaclab.scene import SceneEntityCfg

from utils.hyperparams import HPARAMS


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
            'static_friction_range': HPARAMS['event']['panda']['material']['static_friction_range'],
            'dynamic_friction_range': HPARAMS['event']['panda']['material']['dynamic_friction_range'],
            'restitution_range': HPARAMS['event']['panda']['material']['restitution_range'],
            'num_buckets': HPARAMS['event']['panda']['material']['num_buckets'],
        },
    )
    panda_stiffness_damping: EventTermCfg = EventTermCfg(
        func=mdp.randomize_actuator_gains,
        mode='reset',
        params={
            'asset_cfg': SceneEntityCfg('panda', body_names='.*'),
            'stiffness_distribution_params': HPARAMS['event']['panda']['actuator_gains']['stiffness_distribution_params'],
            'damping_distribution_params': HPARAMS['event']['panda']['actuator_gains']['damping_distribution_params'],
            'operation': 'scale',
            'distribution': 'log_uniform',
        },
    )
    panda_mass: EventTermCfg = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode='reset',
        params={
            'asset_cfg': SceneEntityCfg('panda', body_names='.*'),
            'mass_distribution_params': HPARAMS['event']['panda']['actuator_gains']['mass_distribution_params'],
            'operation': 'scale',
            'distribution': 'log_uniform',
        },
    )
    panda_mass: EventTermCfg = EventTermCfg(
        func=mdp.randomize_joint_parameters,
        mode='reset',
        params={
            'asset_cfg': SceneEntityCfg('panda', body_names='.*'),
            'friction_distribution_params': HPARAMS['event']['panda']['actuator_gains']['friction_distribution_params'],
            'armature_distribution_params': HPARAMS['event']['panda']['actuator_gains']['armature_distribution_params'],
            'operation': 'scale',
            'distribution': 'log_uniform',
        },
    )
    panda_ext_force_torque: EventTermCfg = EventTermCfg(
        func=mdp.randomize_joint_parameters,
        mode='reset',
        params={
            'asset_cfg': SceneEntityCfg('panda', body_names='.*'),
            'force_range': HPARAMS['event']['panda']['actuator_gains']['force_range'],
            'torque_range': HPARAMS['event']['panda']['actuator_gains']['torque_range'],
        },
    )
    panda_joints_scale: EventTermCfg = EventTermCfg(
        func=mdp.randomize_joint_parameters,
        mode='reset',
        params={
            'asset_cfg': SceneEntityCfg('panda', body_names='.*'),
            'position_range': HPARAMS['event']['panda']['actuator_gains']['position_range'],
            'velocity_range': HPARAMS['event']['panda']['actuator_gains']['velocity_range'],
        },
    )
    
    ### PHYSICS RANDOMIZATION ###
    gravity: EventTermCfg = EventTermCfg(
        func=mdp.randomize_physics_scene_gravity,
        mode='reset',
        params={
            'gravity_distribution_params': HPARAMS['event']['physics']['gravity']['gravity_distribution_params'],
            'operation': 'add',
            'distribution': 'gaussian',
        },
    )
    
    ### OBJECT RANDOMIZATION ###
    '''
    Note:
    Use randomize_visual_color, randomize_visual_texture_material, randomize_rigid_body_material 
    '''