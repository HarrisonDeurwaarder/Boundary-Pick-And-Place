import omni.client
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from pxr import Usd, UsdPhysics, UsdGeom

from source.utils.timer import timer


@timer
def get_assets(root_url: str, exts: tuple[str] = ()) -> str:
    '''
    Get a list of all nucleus assets paths satisfying the parameters 
    
    Args:
        root_url (str): Root url
        exts (tuple[str]): Allowed extensions
        
    Returns:
        list[str]: Asset paths
    '''
    assets: list[str] = []
    stack: list[str] = [root_url.rstrip("/")]
    # Recursively iterate
    while stack:
        url = stack.pop()
        result, entries = omni.client.list(url)
        
        if result != omni.client.Result.OK:
            continue
        
        for e in entries:
            child: str = f'{url}/{e.relative_path}'
            # Append if child entries as present
            if e.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN:
                stack.append(child)
            else:
                # Append to output if ext is matched
                if not exts or child.lower().endswith(exts):
                    assets.append(child)
    return assets


def filter_single_meshes(usd_paths: list[str],) -> list[str]:
    '''
    Filters out non-meshes from USD paths
    
    Args:
        usd_paths (list[str]): Unfiltered path list
        
    Returns
        list[str]: Filtered path list
    '''
    unpermitted_patterns: tuple[str] = ('.thumbs')
    valid_paths: list[str] = []
    # Get valid USD paths
    for usd_path in usd_paths:
        try:
            stage = Usd.Stage.Open(usd_path)
            root = stage.GetDefaultPrim()
            
            if not root or not root.IsValid():
                root = stage.GetPseudoRoot()
                
            # Check that no unpermitted patterns exist in the path
            if any(pattern in usd_path for pattern in unpermitted_patterns):
                continue
                
            # Count rigidbodies
            rigid_bodies = 0
            has_mesh = False
            for prim in Usd.PrimRange(root):
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    rigid_bodies += 1
                if prim.IsA(UsdGeom.Mesh):
                    has_mesh = True
            # Exit if count exceeds one or no meshes exist
            if rigid_bodies > 1 or not has_mesh:
                continue
            # Otherwise, add path
            valid_paths.append(usd_path)
        except:
            pass
        
    return valid_paths