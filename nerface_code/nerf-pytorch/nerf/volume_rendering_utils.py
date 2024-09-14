import torch

from .nerf_helpers import cumprod_exclusive



def volume_render_radiance_field(
    radiance_field,
    depth_values,
    ray_directions,
    radiance_field_noise_std=0.0,
    white_background=False,
    background_prior = None
):
    # TESTED
    one_e_10 = torch.tensor(
        [1e10], dtype=ray_directions.dtype, device=ray_directions.device
    )
    dists = torch.cat(
        (
            depth_values[..., 1:] - depth_values[..., :-1],
            one_e_10.expand(depth_values[..., :1].shape),
        ),
        dim=-1,
    )
    dists = dists * ray_directions[..., None, :].norm(p=2, dim=-1)


    if background_prior is not None:
        rgb = torch.sigmoid(radiance_field[:, :-1, :3])
        rgb = torch.cat((rgb, radiance_field[:, -1, :3].unsqueeze(1)), dim=1)
    else:
        rgb = torch.sigmoid(radiance_field[..., :3])

    # Experimental:
    #torch.autograd.set_detect_anomaly(True)
    #rgb = torch.sigmoid(radiance_field[..., :3])
    #if background_prior is not None:
    #    rgb[:,-1,:] = torch.ones(4096,3)#background_prior

    noise = 0.0
    if radiance_field_noise_std > 0.0:
        noise = (
            torch.randn(
                radiance_field[..., 3].shape,
                dtype=radiance_field.dtype,
                device=radiance_field.device,
            )
            * radiance_field_noise_std
        )
        # noise = noise.to(radiance_field)
    sigma_a = torch.nn.functional.relu(radiance_field[..., 3] + noise)
    # sigma_a[:,-1] += 1e-6 # todo commented this for FCB demo !!!!!!
    alpha = 1.0 - torch.exp(-sigma_a * dists)
    weights = alpha * cumprod_exclusive(1.0 - alpha + 1e-10)
    #weights[:, -1] *= 10
    #weights[:,:] = 0
    #weights[:,-1] = 1
    #weights = torch.softmax(weights, dim=1)
    #argmax_sigma = torch.argmax(sigma_a,dim=1)
    #surface_depth = torch.gather(depth_values, 1, argmax_sigma.unsqueeze(1))
    surface_depth = None
    rgb_map = weights[..., None] * rgb
    rgb_map = rgb_map.sum(dim=-2)
    depth_map = weights * depth_values
    depth_map = depth_map.sum(dim=-1)
    # depth_map = (weights * depth_values).sum(dim=-1)
    acc_map = weights.sum(dim=-1)
    disp_map = 1.0 / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / acc_map)

    if white_background:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    #return rgb_map, disp_map, acc_map, weights, depth_map
    return rgb_map, disp_map, acc_map, weights, surface_depth


