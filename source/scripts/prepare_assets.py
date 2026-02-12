import os
from huggingface_hub import snapshot_download
from datasets import load_dataset
from source.utils.file_io import read_list, write_list
from source.utils.logger import logging
from random import shuffle

from source.sim.launch_app import launch_app

sim_app, args_cli = launch_app()

from pxr import Usd, UsdPhysics, UsdGeom, Gf, PhysxSchema, Sdf


def main() -> None:
    filtered_paths: list[str] = read_list('source/data/usd_paths.txt')
    shuffle(filtered_paths)
    
    rb_paths: list[str] = []
    
    logging.info('Starting USD conversion...')
    # Add rigidbody api
    for i, usd_path in enumerate(filtered_paths):
        usd_name: str = usd_path.split('/')[-1]
        
        logging.info(f'Converting ({usd_name})...')
        # Open stage
        stage = Usd.Stage.Open(usd_path)
        root_prim = stage.GetDefaultPrim()
        logging.info(f'Stage loaded for ({usd_name})')
        
        if not root_prim.IsValid():
            continue
        
        # Create the APIs
        UsdPhysics.RigidBodyAPI.Apply(root_prim)
        UsdPhysics.CollisionAPI.Apply(root_prim)
        
        # Set RB attributes
        rb_api = UsdPhysics.RigidBodyAPI(root_prim)
        rb_api.CreateRigidBodyEnabledAttr(True)
        rb_api.CreateKinematicEnabledAttr(False)
        # Set mass attributes
        mass_api = UsdPhysics.MassAPI.Apply(root_prim)
        mass_api.CreateMassAttr(1.0)
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
        logging.info(f'APIs created for ({usd_name})')
        
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(prim)
                #mesh_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                attr = prim.CreateAttribute("physxMeshCollision:approximation", Sdf.ValueTypeNames.Token)
                attr.Set("convexHull")
            
                # Explicitly check for 'None' and overwrite it
                if prim.GetAttribute("physxMeshCollision:approximation").Get() != "convexHull":
                    prim.GetAttribute("physxMeshCollision:approximation").Set("convexHull")
        logging.info(f'Meshes created for ({usd_name})')
        # Create new path
        new_path = usd_path.replace('.usd', '_rb_modified.usd')
        if new_path == usd_path:
            print(f'Ignored USD ({usd_path})')
            continue # Don't override default usds
        
        rb_paths.append(new_path)
        # Export
        rb_stage = stage.Flatten()
        rb_stage.Export(new_path)
        
        logging.info(f'Successfully converted asset #{i+1} ({usd_name})')
    
    write_list(rb_paths, 'source/data/rb_usd_paths.txt')
        

if __name__ == '__main__':
    main()
    sim_app.close()