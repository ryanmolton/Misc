import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_FOV_RAD = 0.9424777960769379  # 54 degrees by default


def to_hom(X):
    # get homogeneous coordinates of the input
    X_hom = torch.cat([X, torch.ones_like(X[..., :1])], dim=-1)
    return X_hom


def to_hom_pose(pose):
    # get homogeneous coordinates of the input pose
    if pose.shape[-2:] == (3, 4):
        pose_hom = torch.eye(4, device=pose.device)[None].repeat(pose.shape[0], 1, 1)
        pose_hom[:, :3, :] = pose
        return pose_hom
    return pose


def get_default_intrinsics(
    fov_rad=DEFAULT_FOV_RAD,
    aspect_ratio=1.0,
):
    if not isinstance(fov_rad, torch.Tensor):
        fov_rad = torch.tensor(
            [fov_rad] if isinstance(fov_rad, (int, float)) else fov_rad
        )
    if aspect_ratio >= 1.0:  # W >= H
        focal_x = 0.5 / torch.tan(0.5 * fov_rad)
        focal_y = focal_x * aspect_ratio
    else:  # W < H
        focal_y = 0.5 / torch.tan(0.5 * fov_rad)
        focal_x = focal_y / aspect_ratio
    intrinsics = focal_x.new_zeros((focal_x.shape[0], 3, 3))
    intrinsics[:, torch.eye(3, device=focal_x.device, dtype=bool)] = torch.stack(
        [focal_x, focal_y, torch.ones_like(focal_x)], dim=-1
    )
    intrinsics[:, :, -1] = torch.tensor(
        [0.5, 0.5, 1.0], device=focal_x.device, dtype=focal_x.dtype
    )
    return intrinsics


def get_image_grid(img_h, img_w):
    # add 0.5 is VERY important especially when your img_h and img_w
    # is not very large (e.g., 72)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    y_range = torch.arange(img_h, dtype=torch.float32).add_(0.5)
    x_range = torch.arange(img_w, dtype=torch.float32).add_(0.5)
    Y, X = torch.meshgrid(y_range, x_range, indexing="ij")  # [H,W]
    xy_grid = torch.stack([X, Y], dim=-1).view(-1, 2)  # [HW,2]
    return to_hom(xy_grid)  # [HW,3]


def img2cam(X, cam_intr):
    return X @ cam_intr.inverse().transpose(-1, -2)


def cam2world(X, pose):
    X_hom = to_hom(X)
    pose_inv = torch.linalg.inv(to_hom_pose(pose))[..., :3, :4]
    return X_hom @ pose_inv.transpose(-1, -2)


def get_center_and_ray(img_h, img_w, pose, intr):  # [HW,2]
    # given the intrinsic/extrinsic matrices, get the camera center and ray directions]
    # assert(opt.camera.model=="perspective")

    # compute center and ray
    grid_img = get_image_grid(img_h, img_w)  # [HW,3]
    grid_3D_cam = img2cam(grid_img.to(intr.device), intr.float())  # [B,HW,3]
    center_3D_cam = torch.zeros_like(grid_3D_cam)  # [B,HW,3]

    # transform from camera to world coordinates
    grid_3D = cam2world(grid_3D_cam, pose)  # [B,HW,3]
    center_3D = cam2world(center_3D_cam, pose)  # [B,HW,3]
    ray = grid_3D - center_3D  # [B,HW,3]

    return center_3D, ray, grid_3D_cam


def get_plucker_coordinates(
    extrinsics_src,
    extrinsics,
    intrinsics=None,
    fov_rad=DEFAULT_FOV_RAD,
    target_size=[72, 72],
):
    if intrinsics is None:
        intrinsics = get_default_intrinsics(fov_rad).to(extrinsics.device)
    else:
        if not (
            torch.all(intrinsics[:, :2, -1] >= 0)
            and torch.all(intrinsics[:, :2, -1] <= 1)
        ):
            intrinsics[:, :2] /= intrinsics.new_tensor(target_size).view(1, -1, 1) * 8
        assert torch.all(intrinsics[:, :2, -1] >= 0) and torch.all(
            intrinsics[:, :2, -1] <= 1
        ), (
            "Intrinsics should be expressed in resolution-independent normalized image coordinates."
        )

    c2w_src = torch.linalg.inv(extrinsics_src)
    # transform coordinates from the source camera's coordinate system to the coordinate system of the respective camera
    extrinsics_rel = torch.einsum(
        "vnm,vmp->vnp", extrinsics, c2w_src[None].repeat(extrinsics.shape[0], 1, 1)
    )

    intrinsics[:, :2] *= extrinsics.new_tensor(
        [
            target_size[1],  # w
            target_size[0],  # h
        ]
    ).view(1, -1, 1)
    centers, rays, grid_cam = get_center_and_ray(
        img_h=target_size[0],
        img_w=target_size[1],
        pose=extrinsics_rel[:, :3, :],
        intr=intrinsics,
    )

    rays = torch.nn.functional.normalize(rays, dim=-1)
    plucker = torch.cat((rays, torch.cross(centers, rays, dim=-1)), dim=-1)
    plucker = plucker.permute(0, 2, 1).reshape(plucker.shape[0], -1, *target_size)
    return plucker


def normalize(x):
    """Normalization helper function."""
    return x / np.linalg.norm(x)


def normalize_K(K_px, H, W):
    K = K_px.clone().float()
    K[:, 0, 0] /= W
    K[:, 1, 1] /= H
    K[:, 0, 2] /= W
    K[:, 1, 2] /= H
    return K


def backward_warp(
    img: torch.Tensor,
    disp: torch.Tensor,
    *,
    padding_mode: str = "zeros",
    align_corners: bool = True,
):
    B, C, H, W = img.shape

    if disp.dim() == 4:
        disp = disp.squeeze(1)
    else:
        disp = disp

    device, dtype = img.device, img.dtype
    v, u = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )  # each is (H, W)
    v = v.unsqueeze(0).expand(B, -1, -1)  # (B, H, W)
    u = u.unsqueeze(0).expand(B, -1, -1)

    u_src = u - disp  # (B, H, W)
    v_src = v  # (B, H, W)

    x_norm = (u_src / (W - 1)) * 2 - 1  # (B, H, W)
    y_norm = (v_src / (H - 1)) * 2 - 1  # (B, H, W)

    grid = torch.stack((x_norm, y_norm), dim=-1)

    left_warped = F.grid_sample(
        img,  # source
        grid,  # sampling grid
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=align_corners,
    )  # → (B, C, H, W)

    mask = (x_norm >= -1) & (x_norm <= 1) & (y_norm >= -1) & (y_norm <= 1)
    mask = mask.unsqueeze(1).expand(-1, C, -1, -1)  # [B, 3, H, W]

    return left_warped, mask


def lr_consistency_mask(
    src_disp: torch.Tensor,  # [B,1,H,W]
    tgt_disp: torch.Tensor,  # [B,1,H,W]
    thresh: float = 1.0,
    padding_mode: str = "zeros",
    align_corners: bool = True,
) -> torch.Tensor:
    """
    Returns a binary mask [B,C,H,W] telling which LEFT pixels are stereo-consistent.
    """
    B, C, H, W = src_disp.shape
    device, dtype = src_disp.device, src_disp.dtype

    v, u = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )  # each (H,W)
    u = u.unsqueeze(0).expand(B, -1, -1)  # (B,H,W)
    v = v.unsqueeze(0).expand(B, -1, -1)

    u_r = u - src_disp.squeeze(1)  # target column in RIGHT img
    x_norm = (u_r / (W - 1)) * 2 - 1  # → [-1,1]
    y_norm = (v / (H - 1)) * 2 - 1

    grid = torch.stack((x_norm, y_norm), dim=-1)  # (B,H,W,2)

    disp_r_warped = F.grid_sample(
        tgt_disp,
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=align_corners,
    ).squeeze(1)  # (B,H,W)

    lrc_error = torch.abs(src_disp.squeeze(1) - (-disp_r_warped))  # sign!

    inside = (x_norm >= -1) & (x_norm <= 1) & (y_norm >= -1) & (y_norm <= 1)

    mask = ((lrc_error < thresh) & inside).unsqueeze(1)  # [B,1,H,W]
    mask = mask.expand(-1, C, -1, -1)
    return mask
