import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List
from pointnet2_ops import pointnet2_utils

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    xyz = xyz.contiguous()
    B, N, C = xyz.shape
    S = npoint
    # fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    fps_idx = pointnet2_utils.furthest_point_sample(xyz, npoint).long()
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points

class PointNetSetAbstraction_grid(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, use_xyz, normalize_xyz, group_all):
        super(PointNetSetAbstraction_grid, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.use_xyz = use_xyz
        self.normalize_xyz = normalize_xyz
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        if self.use_xyz:
            in_channel += 3
        last_channel = in_channel
        self.sharedmlp3d = [ ]
        self.sharedmlp3d.append(last_channel)
        for out_channel in mlp:
            # self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            # self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            self.sharedmlp3d.append(out_channel)
            last_channel = out_channel
        self.group_all = group_all

        self.grid_num = 5
        self.grid_render = GridRender(grid_num=self.grid_num)
        # print('self.sharedmlp3d:',self.sharedmlp3d)

        self.mlp_convs_3d = nn.ModuleList()
        self.mlp_bns_3d = nn.ModuleList()
        for i in range(len(self.sharedmlp3d)-1):
            if i < len(self.sharedmlp3d) - 2:
                self.mlp_convs_3d.append(nn.Conv3d(self.sharedmlp3d[i], self.sharedmlp3d[i+1], (3,3,3)))
                self.mlp_bns_3d.append(nn.BatchNorm3d(self.sharedmlp3d[i+1]))
            else:
                self.mlp_convs_3d.append(nn.Conv3d(self.sharedmlp3d[i], self.sharedmlp3d[i+1], (1,1,1)))
                self.mlp_bns_3d.append(nn.BatchNorm3d(self.sharedmlp3d[i+1]))


        self.mlp_convs_2d = nn.ModuleList()
        self.mlp_bns_2d = nn.ModuleList()
        #   add conv2d block as dual channel
        for i in range(len(self.sharedmlp3d)-1):
            self.mlp_convs_2d.append(nn.Conv2d(self.sharedmlp3d[i], self.sharedmlp3d[i+1], 1))
            self.mlp_bns_2d.append(nn.BatchNorm2d(self.sharedmlp3d[i+1]))




        # print(self.mlp_convs_3d)


    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, grouped_xyz, grouped_points, fps_idx = self.ballquery_and_group(self.npoint, self.radius, self.nsample, xyz, points,\
                                                                                     self.use_xyz, self.normalize_xyz)
        # new_xyz: sampled points position data, [B, npoint, 3]
        # grouped_xyz: [B, npoint, nsample, 3]
        # grouped_points: sampled points data, [B, npoint, nsample, C]

        grid_feats = self.grid_render(grouped_points.permute(0, 3, 1, 2), grouped_xyz.permute(0, 3, 1, 2))  #   (B*npoint, C, grid_num, grid_num, grid_num)


        for i in range(len(self.sharedmlp3d)-1):
            grid_feats = self.mlp_convs_3d[i](grid_feats)
            grid_feats = F.relu(self.mlp_bns_3d[i](grid_feats))
        new_features = F.max_pool3d(grid_feats, kernel_size=[grid_feats.size(2), grid_feats.size(3), grid_feats.size(4)])   #   (B*npoint, C, 1, 1, 1)
        new_features = new_features.reshape(grouped_points.size(0), grouped_points.size(1), -1).permute(0, 2, 1).contiguous()       #   (B, mlp[-1], npoint)


        #   conv2d block
        grouped_points_conv2d = grouped_points.permute(0, 3, 1, 2)  #   [B, C, npoint, nsample]
        for i in range(len(self.sharedmlp3d)-1):
            grouped_points_conv2d = self.mlp_convs_2d[i](grouped_points_conv2d)
            grouped_points_conv2d = F.relu(self.mlp_bns_2d[i](grouped_points_conv2d))
        #   grouped_points_conv2d: [B, mlp[-1], npoint, nsample]
        new_features_conv2d = torch.max(grouped_points_conv2d, 3)[0]    #   [B, mlp[-1], npoint]
        new_features = new_features + new_features_conv2d   #   [B, mlp[-1], npoint]






        new_xyz = new_xyz.permute(0, 2, 1)
        new_points = new_features
        return new_xyz, new_points

    def ballquery_and_group(self, npoint, radius, nsample, xyz, features, use_xyz=True, normalize_xyz=True):
        """
        Input:
            npoint:
            radius:
            nsample:
            xyz: input points position data, [B, N, 3]
            features: input points data, [B, N, D]
        Return:
            new_xyz: [B, npoint, 3]
            grouped_xyz: sampled points position data, [B, npoint, nsample, 3]
            grouped_features: sampled points data, [B, npoint, nsample, D]
            fps_idx: [B, npoint]
        """
        xyz = xyz.contiguous()
        B, N, C = xyz.shape
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, npoint).long()                 #   [B, npoint]
        new_xyz = index_points(xyz, fps_idx)                                                #   [B, npoint, 3]
        idx_ballquery = pointnet2_utils.ball_query(radius, nsample, xyz, new_xyz).long()    #   [B, npoint, nsample]
        grouped_xyz = index_points(xyz, idx_ballquery)                                      #   [B, npoint, nsample, 3]
        grouped_xyz_norm = grouped_xyz - new_xyz.view(B, npoint, 1, C)                      #   [B, npoint, nsample, 3]
        if normalize_xyz:
            grouped_xyz_norm = grouped_xyz_norm / radius

        if features is not None:
            grouped_features = index_points(features, idx_ballquery)                            #   [B, npoint, nsample, C]
            if use_xyz:
                grouped_features = torch.cat([grouped_xyz_norm, grouped_features], dim=-1)      #   [B, npoint, nsample, 3+C]

        else:
            grouped_features = grouped_xyz_norm

        return new_xyz, grouped_xyz_norm, grouped_features, fps_idx







class GridRender(nn.Module):
    def __init__(self, grid_num = 16):
        super(GridRender, self).__init__()
        self.grid_num = grid_num
        self.split_num = 1
        self.radius_x, self.radius_y, self.radius_z = 1.0/(self.grid_num-1), 1.0/(self.grid_num-1),  1.0/(self.grid_num-1)
        self.radius_xyz = math.sqrt(self.radius_x**2 + self.radius_y**2 + self.radius_z**2)
        self.line_x = torch.linspace(-1.0, 1.0, steps=self.grid_num).cuda()
        self.line_y = torch.linspace(-1.0, 1.0, steps=self.grid_num).cuda()
        self.line_z = torch.linspace(-1.0, 1.0, steps=self.grid_num).cuda()
        self.grid_x, self.grid_y, self.grid_z = torch.meshgrid(self.line_x, self.line_y, self.line_z)
        # self.grid_x,y,z: [grid_num, grid_num, grid_num]
        self.grid_xyz_tmp = torch.cat([self.grid_x.reshape(1, -1, 1), self.grid_y.reshape(1, -1, 1), self.grid_z.reshape(1, -1, 1)], dim=-1)
        # self.grid_xyz: [1, grid_num**3, 3]

    def forward(self, feats, locs):
        '''
        Input:
            feat: [B, C, N, k]
            locs: [B, 3, N, k]
        Output:
            grid_feats: [B*N, C, grid_num, grid_num, grid_num]
        '''
        B, C, NP, NS = feats.shape

        feats = feats.permute(0, 2, 3, 1).reshape(B*NP, NS, C)      #   feats: [B*N, k, C]
        locs  = locs.permute(0, 2, 3, 1).reshape(B*NP, NS, 3)       #   locs:  [B*N, k, 3]
        grid_feats_all = torch.zeros(B*NP, C, self.grid_num**3).to(feats.device)     #   [B*NP, C, grid_num**3]
        T = int( feats.size(0) / self.split_num)
        grid_xyz = self.grid_xyz_tmp.unsqueeze(0).repeat(T, NS, 1, 1).to(locs.device)
        # grid_xyz: [T, NS, grid_num**3, 3]

        for i in range(self.split_num):
            locs_i = locs[i*T:i*T+T, :, :].unsqueeze(2).repeat(1, 1, self.grid_num**3, 1)
            # locs_i: [T, NS, grid_num**3, 3]
            rend_dist = torch.abs(locs_i - grid_xyz)
            # rend_dist: [T, NS, grid_num**3, 3]
            # rend_dx = F.relu(1.0 - rend_dist[:,:,:,0] / self.radius_x)
            # # rend_dx: [T, NS, grid_num**3]
            # rend_dy = F.relu(1.0 - rend_dist[:,:,:,1] / self.radius_y)
            # # rend_dy: [T, NS, grid_num**3]
            # rend_dz = F.relu(1.0 - rend_dist[:,:,:,2] / self.radius_z)
            # # rend_dz: [T, NS, grid_num**3]
            # rend_dxyz = (rend_dx * rend_dy * rend_dz)
            # # rend_dxyz: [T, NS, grid_num**3]

            rend_dist_l2 = torch.norm(rend_dist, p=2, dim=-1)
            # rend_dist_l2: [T, NS, grid_num**3]
            rend_dist_l2 = F.relu(1.0 - rend_dist_l2 / self.radius_xyz)
            # rend_dist_l2: [T, NS, grid_num**3]
            # rend_dist_l2 = rend_dist_l2 * torch.where(rend_dxyz > 0, torch.ones_like(rend_dxyz).cuda(), torch.zeros_like(rend_dxyz).cuda()).cuda()
            # # rend_dist_l2: [T, NS, grid_num**3]

            feats_i = feats[i*T:i*T+T, :, :]
            # feats_i: [T, NS, C]
            # feat_wi = feats_i.unsqueeze(2).repeat(1, 1, self.grid_num**3, 1) * rend_dxyz.unsqueeze(-1)
            # # feat_wi: [T, NS, grid_num**3, C]

            # rend_dxyz = rend_dxyz / (torch.sum(rend_dxyz, dim=1, keepdim=True) + 1e-6)
            rend_dist_l2 = rend_dist_l2 / (torch.sum(rend_dist_l2, dim=1, keepdim=True) + 1e-6)
            # rend_dxyz: [T, NS, grid_num**3]

            # grid_feats = torch.matmul(feats_i.permute(0,2,1), rend_dxyz)
            grid_feats = torch.matmul(feats_i.permute(0,2,1), rend_dist_l2)
            # grid_feats: [T, C, grid_num**3]

            # grid_feats = torch.matmul(rend_dxyz.permute(0,2,1).unsqueeze(2), feat_wi.permute(0,2,1,3)).squeeze(-2).permute(0,2,1)
            # # grid_feats: [T, C, grid_num**3]

            # feats_i = feats[i*T:i*T+T, :, :].unsqueeze(-1).repeat(1, 1, 1, self.grid_num**3)
            # # feats_i: [T, NS, C, grid_num**3]
            # grid_feats =  rend_dxyz * feats_i
            # # grid_feats: [T, NS, C, grid_num**3]
            # grid_feats = torch.max(grid_feats, dim=1, keepdim=False)[0]
            # # grid_feats: [T, C, grid_num**3]

            grid_feats_all[i*T:i*T+T] = grid_feats

        grid_feats_all = grid_feats_all.view(B*NP, C, self.grid_num, self.grid_num, self.grid_num)
        # grid_feats: [B*NP, C, grid_num, grid_num, grid_num]

        return grid_feats_all


class PointnetSAModuleVotesRender(nn.Module):
    ''' Modified based on _PointnetSAModuleBase and PointnetSAModuleMSG
    with extra support for returning point indices for getting their GT votes '''

    def __init__(
            self,
            *,
            mlp: List[int],
            npoint: int = None,
            radius: float = None,
            nsample: int = None,
            bn: bool = True,
            use_xyz: bool = True,
            pooling: str = 'max',
            sigma: float = None, # for RBF pooling
            normalize_xyz: bool = False, # noramlize local XYZ with radius
            sample_uniformly: bool = False,
            ret_unique_cnt: bool = False,
            grid_num: int = 5,
            use_nonlocal: bool = False
    ):
        super().__init__()

        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.pooling = pooling
        self.mlp_module = None
        self.use_xyz = use_xyz
        self.sigma = sigma
        if self.sigma is None:
            self.sigma = self.radius/2
        self.normalize_xyz = normalize_xyz
        self.ret_unique_cnt = ret_unique_cnt
        self.grid_num = grid_num
        self.use_nonlocal = use_nonlocal

        if npoint is not None:
            self.grouper = pointnet2_utils.QueryAndGroup(radius, nsample,
                use_xyz=use_xyz, ret_grouped_xyz=True, normalize_xyz=normalize_xyz,
                sample_uniformly=sample_uniformly, ret_unique_cnt=ret_unique_cnt, use_cube=False)
        else:
            self.grouper = pointnet2_utils.GroupAll(use_xyz, ret_grouped_xyz=True)

        mlp_spec = mlp
        if use_xyz and len(mlp_spec)>0:
            mlp_spec[0] += 3
        # self.mlp_module = pt_utils.SharedMLP(mlp_spec, bn=bn)
        self.mlp_module_3d = pt_utils.SharedMLP3D(mlp_spec, bn=bn)
        # self.mlp_module_3d = pt_utils.SharedMLP3DDW(mlp_spec, bn=bn)

        # if self.use_nonlocal:
        #     self.nl_module= NonLocal(in_channels=256, use_relu=True, use_bn=True, use_shortcut=True)

        self.grid_render = GridRender(grid_num=self.grid_num)

        # self.conv3d1  = torch.nn.Conv3d(mlp_spec[0], mlp_spec[0], kernel_size=3, groups=256, padding=0)
        # self.bn3d1 = torch.nn.BatchNorm3d(mlp_spec[0])
        # self.conv3d1_ = torch.nn.Conv3d(mlp_spec[0], mlp_spec[1], kernel_size=1)
        # self.bn3d1_ = torch.nn.BatchNorm3d(mlp_spec[1])
        # self.conv3d2  = torch.nn.Conv3d(mlp_spec[2], mlp_spec[2], kernel_size=3, groups=128, padding=0)
        # self.bn3d2 = torch.nn.BatchNorm3d(mlp_spec[2])
        # self.conv3d2_ = torch.nn.Conv3d(mlp_spec[2], mlp_spec[3], kernel_size=1)
        # self.bn3d2_ = torch.nn.BatchNorm3d(mlp_spec[3])

        # self.conv3d1  = torch.nn.Conv3d(mlp_spec[0], mlp_spec[1], kernel_size=3, padding=0)
        # self.bn3d1 = torch.nn.BatchNorm3d(mlp_spec[1])
        # self.conv3d2  = torch.nn.Conv3d(mlp_spec[2], mlp_spec[3], kernel_size=3, padding=0)
        # self.bn3d2 = torch.nn.BatchNorm3d(mlp_spec[3])

    def forward(self, xyz: torch.Tensor,
                features: torch.Tensor = None,
                inds: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, C, N) tensor of the descriptors of the the features
        inds : torch.Tensor
            (B, npoint) tensor that stores index to the xyz points (values in 0-N-1)

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B, \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        inds: torch.Tensor
            (B, npoint) tensor of the inds
        """
        xyz = xyz.transpose(1, 2).contiguous()
        features = features.contiguous()

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        if inds is None:
            inds = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
        else:
            assert(inds.shape[1] == self.npoint)
        new_xyz = pointnet2_utils.gather_operation(
            xyz_flipped, inds
        ).transpose(1, 2).contiguous() if self.npoint is not None else None

        if not self.ret_unique_cnt:
            grouped_features, grouped_xyz = self.grouper(
                xyz, new_xyz, features
            )  # (B, C, npoint, nsample)
        else:
            grouped_features, grouped_xyz, unique_cnt = self.grouper(
                xyz, new_xyz, features
            )  # (B, C, npoint, nsample), (B,3,npoint,nsample), (B,npoint)

        grid_feats = self.grid_render(grouped_features, grouped_xyz)
        # grid_feats (B*npoint, C, grid_num, grid_num, grid_num)

        # grid_feats = F.relu(self.bn3d1_(self.conv3d1_(self.bn3d1(self.conv3d1(grid_feats)))))
        # grid_feats = F.relu(self.bn3d2_(self.conv3d2_(self.bn3d2(self.conv3d2(grid_feats)))))

        # grid_feats = F.relu(self.bn3d1(self.conv3d1(grid_feats)))
        # grid_feats = F.relu(self.bn3d2(self.conv3d2(grid_feats)))
        # grid_feats = self.mlp_module_3d(grid_feats)
        grid_feats = self.mlp_module_3d(grid_feats)
        new_features = F.max_pool3d(grid_feats, kernel_size=[grid_feats.size(2), grid_feats.size(3), grid_feats.size(4)])
        # new_features = (B*npoint, C, 1, 1, 1)
        # new_features = new_features.squeeze(-1).squeeze(-1).squeeze(-1)
        new_features = new_features.reshape(grouped_features.size(0), grouped_features.size(2), new_features.size(1)).permute(0, 2, 1).contiguous()
        # new_features = (B, mlp[-1], npoint)

        new_xyz = new_xyz.transpose(1, 2).contiguous()

        if not self.ret_unique_cnt:
            # return new_xyz, new_features, inds
            return new_xyz, new_features
        else:
            return new_xyz, new_features, inds, unique_cnt

#   0.754
class LgrNet4_3(nn.Module):
    def __init__(self,num_classes=40,normal_channel=False, **kwargs):
        super().__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        # self.sa1 = PointnetSAModuleVotesRender(npoint=512, radius=0.2, nsample=32, mlp=[in_channel-3, 64, 64, 128], use_xyz=True, normalize_xyz=True)
        # self.sa2 = PointnetSAModuleVotesRender(npoint=128, radius=0.4, nsample=64, mlp=[128, 128, 128, 256], use_xyz=True, normalize_xyz=True)
        # self.sa3 = PointnetSAModuleVotes(npoint=None, radius=None, nsample=None, mlp=[256, 256, 512, 1024], use_xyz=True, normalize_xyz=True)
        self.sa1 = PointNetSetAbstraction_grid(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128],\
                                               use_xyz=False, normalize_xyz=False, group_all=False)
        self.sa2 = PointNetSetAbstraction_grid(npoint=128, radius=0.4, nsample=64, in_channel=128, mlp=[128, 128, 256], \
                                               use_xyz=True, normalize_xyz=False, group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024],\
                                          group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x

#   0.749
class LgrNet4_4(nn.Module):
    def __init__(self,num_classes=40,normal_channel=False, **kwargs):
        super().__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        # self.sa1 = PointnetSAModuleVotesRender(npoint=512, radius=0.2, nsample=32, mlp=[in_channel-3, 64, 64, 128], use_xyz=True, normalize_xyz=True)
        # self.sa2 = PointnetSAModuleVotesRender(npoint=128, radius=0.4, nsample=64, mlp=[128, 128, 128, 256], use_xyz=True, normalize_xyz=True)
        # self.sa3 = PointnetSAModuleVotes(npoint=None, radius=None, nsample=None, mlp=[256, 256, 512, 1024], use_xyz=True, normalize_xyz=True)
        self.sa1 = PointNetSetAbstraction_grid(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128],\
                                               use_xyz=False, normalize_xyz=False, group_all=False)
        self.sa2 = PointNetSetAbstraction_grid(npoint=128, radius=0.4, nsample=64, in_channel=128, mlp=[128, 128, 256], \
                                               use_xyz=False, normalize_xyz=False, group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024],\
                                          group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x

#   bug
class LgrNet4_5(nn.Module):
    def __init__(self,num_classes=40,normal_channel=False, **kwargs):
        super().__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        # self.sa1 = PointnetSAModuleVotesRender(npoint=512, radius=0.2, nsample=32, mlp=[in_channel-3, 64, 64, 128], use_xyz=True, normalize_xyz=True)
        # self.sa2 = PointnetSAModuleVotesRender(npoint=128, radius=0.4, nsample=64, mlp=[128, 128, 128, 256], use_xyz=True, normalize_xyz=True)
        # self.sa3 = PointnetSAModuleVotes(npoint=None, radius=None, nsample=None, mlp=[256, 256, 512, 1024], use_xyz=True, normalize_xyz=True)
        self.sa1 = PointNetSetAbstraction_grid(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128],\
                                               use_xyz=True, normalize_xyz=False, group_all=False)
        self.sa2 = PointNetSetAbstraction_grid(npoint=128, radius=0.4, nsample=64, in_channel=128, mlp=[128, 128, 256], \
                                               use_xyz=False, normalize_xyz=False, group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024],\
                                          group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x
#   bug
class LgrNet4_6(nn.Module):
    def __init__(self,num_classes=40,normal_channel=False, **kwargs):
        super().__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        # self.sa1 = PointnetSAModuleVotesRender(npoint=512, radius=0.2, nsample=32, mlp=[in_channel-3, 64, 64, 128], use_xyz=True, normalize_xyz=True)
        # self.sa2 = PointnetSAModuleVotesRender(npoint=128, radius=0.4, nsample=64, mlp=[128, 128, 128, 256], use_xyz=True, normalize_xyz=True)
        # self.sa3 = PointnetSAModuleVotes(npoint=None, radius=None, nsample=None, mlp=[256, 256, 512, 1024], use_xyz=True, normalize_xyz=True)
        self.sa1 = PointNetSetAbstraction_grid(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128],\
                                               use_xyz=True, normalize_xyz=False, group_all=False)
        self.sa2 = PointNetSetAbstraction_grid(npoint=128, radius=0.4, nsample=64, in_channel=128, mlp=[128, 128, 256], \
                                               use_xyz=True, normalize_xyz=False, group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024],\
                                          group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    data = torch.rand(2, 3, 1024).cuda()
    model = LgrNet4_3().cuda()
    out = model(data)
    print(out.shape)

    # xyz = torch.rand(2, 3, 256, 24).cuda()
    # points = torch.rand(2, 64, 256, 24).cuda()
    # model = GridRender(grid_num=5).cuda()
    # out = model(points, xyz)
    # print(out.shape)



