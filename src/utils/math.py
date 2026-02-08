import torch
import torch.nn.functional as F


def vect_to_quat(
    vector: torch.Tensor,
    axis: torch.Tensor,
) -> torch.Tensor:
    """
    Converts a vector [E, 3] to a quaternion [E, 4] along the axis [3]

    Args:
        vector (torch.Tensor): Vector
        axis (torch.Tensor): Axis

    Returns:
        torch.Tensor: Quaternion
    """
    # Compute quat components
    c = (vector * axis).sum(dim=1).unsqueeze(1)
    v = torch.cross(vector, axis, dim=1)
    # Concat components
    target_quat = F.normalize(torch.concat((1.0 + c, v,), dim=1))
    return target_quat