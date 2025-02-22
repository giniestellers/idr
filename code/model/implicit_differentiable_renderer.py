import torch
import torch.nn as nn
import numpy as np

from utils import rend_util
from model.embedder import *
from model.ray_tracing import RayTracing
from model.sample_network import SampleNetwork

class ImplicitNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,# dimension of feature vector z that accounts for global illumination effects
            d_in, # 3, dimension of the domain of the implicit function f that parametrizes the surface by its zero-level set
            d_out, # 1, dimension of the values taken by implicit function f(x)
            dims, # list with dimensions of MLP layers
            geometric_init=True,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            multires=0
    ):
        super().__init__()
        # the implicit function f is extended to return a global feature
        # vector z s.t. for each input coordinate x the network computes
        # F(x) = [f(x), z(z)]. z accounts for global illumination effects that cannot
        # be modelled by a rendering function M(x, n, v) that computes rgb
        # values as a function of the surface point, normal, and viewind direction
        dims = [d_in] + dims + [d_out + feature_vector_size]

        self.embed_fn = None
        # positional encoding uses a frequency decompositions of the input
        # signal (xyz coordinates) to parametrize the input
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)

    def forward(self, input, compute_grad=False):
        # given x,y,z coordinate, compute the value of the function at this location, F(x) = [f(x), z(z)]
        if self.embed_fn is not None:
            input = self.embed_fn(input)

        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        return x

    def gradient(self, x):
        # given x,y,z coordinate, compute the value of the implicit function f at
        # this location, that is, \nabla f (x) or the normal to the surface at point x
        x.requires_grad_(True)
        y = self.forward(x)[:,:1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)

class RenderingNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size, # dimension of feature vector z that accounts for global illumination effects
            mode, # one of 'idr', 'no_view_dir', 'no_normal' that allows for different input sizes (d_in)
            d_in, # 9 (idr) or 6 (no_view_dir, no_normal) dimension of input to the network that approximates the function that describes the color of a surfacepoint as a function of the surface point, normal, and viewing direction (3 for coordinate x, 3 for normal n, 3 for viewing direction v)
            d_out,# 3 (for RGB output)
            dims, # list with dimensions of internal MLP layers
            weight_norm=True,
            multires_view=0
    ):
        super().__init__()

        self.mode = mode
        # the input to the rendering network is the location of the point in the surface x,
        # the normal at this point n, and the viewing direction v, and a feature vector z(x)
        # accounts for global illumination effects that cannot be modelled by a rendering
        # function M(x, n, v) that computes rgb values as a function of x, n, and v only. The
        # rendering network approximates a function M(x, n, v, z)
        dims = [d_in + feature_vector_size] + dims + [d_out]

        self.embedview_fn = None
        # embedding for viewing direction v
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, points, normals, view_dirs, feature_vectors):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        x = self.tanh(x)
        return x

class IDRNetwork(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.feature_vector_size = conf.get_int('feature_vector_size')
        self.implicit_network = ImplicitNetwork(self.feature_vector_size, **conf.get_config('implicit_network'))
        self.rendering_network = RenderingNetwork(self.feature_vector_size, **conf.get_config('rendering_network'))
        self.ray_tracer = RayTracing(**conf.get_config('ray_tracer'))
        self.sample_network = SampleNetwork()
        self.object_bounding_sphere = conf.get_float('ray_tracer.object_bounding_sphere')

    def forward(self, input):

        # Parse model input
        intrinsics = input["intrinsics"]
        uv = input["uv"]
        pose = input["pose"]
        object_mask = input["object_mask"].reshape(-1)

        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)

        batch_size, num_pixels, _ = ray_dirs.shape

        # get location of approximate location of the intersection of each ray with
        # direction v starting at camera center c with the surface x_0 = c + t_0 v, where
        # t_0 (dists) is the distance to the surface along each ray and x_0 the intersection (points)
        self.implicit_network.eval()
        with torch.no_grad():
            points, network_object_mask, dists = self.ray_tracer(sdf=lambda x: self.implicit_network(x)[:, 0],
                                                                 cam_loc=cam_loc,
                                                                 object_mask=object_mask,
                                                                 ray_directions=ray_dirs)
        self.implicit_network.train()

        points = (cam_loc.unsqueeze(1) + dists.reshape(batch_size, num_pixels, 1) * ray_dirs).reshape(-1, 3)

        sdf_output = self.implicit_network(points)[:, 0:1]
        ray_dirs = ray_dirs.reshape(-1, 3)

        # during training, we need a differentiable way to find the location of the points where the rays
        # emanating from each pixel intersect the surface and define the pixel RGB color. To use the paper
        # first-order approximation, we need to compute
        # x = c + (t_0 - f(c + t_0 v; \theta)/ \nabla f(x_0; \theta_0) \cdot v_0) v
        # where
        #   \theta_0 current network parameters
        #   c_0 current estimate of camera center (as it depends on camera pose)
        #   v_0 current estimate of ray direction (as it depends on current estimate of camera center)
        #   t_0 current estimate of distance to the surface along each ray
        #   x_0 = c_0 + t_0 v_0 current estimate of ray intersection (given by sphere tracing algorithm)
        if self.training:
            surface_mask = network_object_mask & object_mask
            surface_points = points[surface_mask] # x
            surface_dists = dists[surface_mask].unsqueeze(-1) #t_0
            surface_ray_dirs = ray_dirs[surface_mask] # v_0
            surface_cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)[surface_mask] # c
            surface_output = sdf_output[surface_mask] # f(x)
            N = surface_points.shape[0]

            output = self.implicit_network(surface_points) #f(x)
            # dettach to get f(x_0) from f(x), which tells pytorch that the computation of f(x_0) is not tracked during construction of computational graph
            surface_sdf_values = output[:N, 0:1].detach()

            # Sample points for the eikonal loss
            eik_bounding_box = self.object_bounding_sphere
            n_eik_points = batch_size * num_pixels // 2
            eikonal_points = torch.empty(n_eik_points, 3).uniform_(-eik_bounding_box, eik_bounding_box).cuda()
            eikonal_pixel_points = points.clone()
            eikonal_pixel_points = eikonal_pixel_points.detach()
            eikonal_points = torch.cat([eikonal_points, eikonal_pixel_points], 0) # x_bar
            # x_bar, points where eikonal equation needs to be imposed to gurantee a SDF. We use all the points that are involved in computations
            # plus a set of points randomly distributed in the sphere to guarantee SDF behavior in computations

            points_all = torch.cat([surface_points, eikonal_points], dim=0) # [x_0, x_bar]

            g = self.implicit_network.gradient(points_all)
            # dettach to get \nabla f(x_0) from \nabla f(x), which tells pytorch that the computation of \nabla f(x_0) is not tracked during construction of computational graph
            surface_points_grad = g[:N, 0, :].clone().detach() # \nabla f(x_0; \theta_0)
            grad_theta = g[N:, 0, :] # \nabla f(x_bar) where x_bar = [points_uniformly_distributed_sphere, surface_points_or_with_minimal_sdf_along_ray]

            differentiable_surface_points = self.sample_network(surface_output, # f(x)
                                                                surface_sdf_values,# f(x_0) dettached
                                                                surface_points_grad,# \nabla f(x_0; \theta_0) dettached
                                                                surface_dists, # t_0
                                                                surface_cam_loc, # c
                                                                surface_ray_dirs) # v

        else:
            surface_mask = network_object_mask
            differentiable_surface_points = points[surface_mask]
            grad_theta = None

        view = -ray_dirs[surface_mask]

        rgb_values = torch.ones_like(points).float().cuda()
        if differentiable_surface_points.shape[0] > 0:
            rgb_values[surface_mask] = self.get_rbg_value(differentiable_surface_points, view)

        output = {
            'points': points,
            'rgb_values': rgb_values,
            'sdf_output': sdf_output,
            'network_object_mask': network_object_mask,
            'object_mask': object_mask,
            'grad_theta': grad_theta
        }

        return output

    def get_rbg_value(self, points, view_dirs):
        output = self.implicit_network(points)
        g = self.implicit_network.gradient(points)
        normals = g[:, 0, :]

        feature_vectors = output[:, 1:]
        rgb_vals = self.rendering_network(points, normals, view_dirs, feature_vectors)

        return rgb_vals
