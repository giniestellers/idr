import torch.nn as nn
import torch

class SampleNetwork(nn.Module):
    '''
    Represent the intersection (sample) point as differentiable function of the implicit geometry and camera parameters.
    See equation 3 in the paper for more details.
    '''

    def forward(self, surface_output, surface_sdf_values, surface_points_grad, surface_dists, surface_cam_loc, surface_ray_dirs):
        # Args:
        #   surface_output: f(x) = f(c + t_0 * v)
        #   surface_sdf_values: f(x_0)
        #   surface_points_grad: gradient of implicit function f at current intersection point \nabla f(x_0, \theta_0)
        #   surface_dists: distance along the ray to the surface (t_0 in the paper)
        #   surface_cam_loc: camera centers (c in the paper)
        #   surface_ray_dirs: direction of the rays (v in the paper)

        # Compute t(\theta, c, v) = t_0 - f(c + t_0 * v)/(\nabla f(x_0, \theta_0) \cdot v_0)
        surface_ray_dirs_0 = surface_ray_dirs.detach() # dettach to get v_0 (not tracked during construction of computational graph)
        surface_points_dot = torch.bmm(surface_points_grad.view(-1, 1, 3),
                                       surface_ray_dirs_0.view(-1, 3, 1)).squeeze(-1)
        surface_dists_theta = surface_dists - (surface_output - surface_sdf_values) / surface_points_dot

        # x(theta,c,v) = c + t(\theta, c, v) * v
        surface_points_theta_c_v = surface_cam_loc + surface_dists_theta * surface_ray_dirs

        return surface_points_theta_c_v
