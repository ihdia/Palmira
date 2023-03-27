import torch
import torch.nn as nn

from defgrid.models.GNN import superpixel_grid

EPS = 1e-8


def defaulttensor(sub_batch_size, device):
    return torch.zeros(sub_batch_size).to(device)


# def gtmask2onehot(gtmask):
#     batch_size, channel_num, h, w = gtmask.shape

#     max_index = gtmask.max().long() + 1
#     onehot = torch.zeros(batch_size, max_index, h, w, device=gtmask.device)
#     onehot.scatter_(dim=1, index=gtmask.long(), value=1.0)

#     return onehot


class DeformableGrid(nn.Module):
    def __init__(self, cfg, device):
        super(DeformableGrid, self).__init__()
        # self.debug = args.debug
        self.device = device

        self.resolution = cfg.MODEL.DEFGRID_MASK_HEAD.GRID_SIZE
        self.state_dim = cfg.MODEL.DEFGRID_MASK_HEAD.STATE_DIM
        self.grid_size = cfg.MODEL.DEFGRID_MASK_HEAD.GRID_SIZE
        self.feature_channel_num = cfg.MODEL.DEFGRID_MASK_HEAD.FEATURE_CHANNEL_NUM
        self.deform_layer_num = cfg.MODEL.DEFGRID_MASK_HEAD.DEFORM_LAYER_NUM
        self.out_dim = cfg.MODEL.DEFGRID_MASK_HEAD.OUT_DIM

        # self.mlp_expansion = args.mlp_expansion
        # self.concat_channels = args.concat_channels
        # self.final_dim = args.final_dim
        self.out_feature_channel_num = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM

        self.grid_type = cfg.MODEL.DEFGRID_MASK_HEAD.GRID_TYPE
        self.add_mask_variance = True
        # self.dataset = args.dataset
        # self.encoder_backbone = args.encoder_backbone
        # self.update_last = args.update_last
        # self.grid_pos_layer = args.grid_pos_layer
        # self.mask_coef = cfg.MODEL.DEFGRID_MASK_HEAD.MASK_COEF

        # try:
        #     self.feature_aggregate_type = args.feature_aggregate_type
        # except:
        #     self.feature_aggregate_type = 'mean'
        #     print(f'Warning: feature_aggregate_type set to {self.feature_aggregate_type}')

        self.gamma = cfg.MODEL.DEFGRID_MASK_HEAD.GAMMA
        # self.sigma = cfg.MODEL.DEFGRID_MASK_HEAD.SIGMA

        # self.input_channel = 3

        # self.use_final_encoder = self.resolution[0] == 512
        # if self.encoder_backbone == 'affinity_net':
        #     self.encoder = MyEncoder(
        #         nr_channel=self.feature_channel_num, input_channel=self.input_channel
        #     ).to(self.device)
        #     self.out_feature_channel_num = self.feature_channel_num
        #     self.final_dim = self.out_feature_channel_num
        # elif self.encoder_backbone == 'deeplab':
        #     self.encoder = DeepLabResnet(
        #         input_channel=self.input_channel,
        #         concat_channels=self.concat_channels,
        #         final_dim=self.final_dim,
        #         final_resolution=self.resolution,
        #         use_final=self.use_final_encoder,
        #         update_last=self.update_last,
        #     ).to(self.device)
        #     self.out_feature_channel_num = self.final_dim
        # elif self.encoder_backbone == 'simplenn':
        #     self.encoder = SimpleEncoder()
        #     self.out_feature_channel_num = 512
        #     self.final_dim = 512

        # if self.grid_pos_layer == 5:
        #     self.deformer_encoder = NetHead(self.final_dim, self.final_dim).to(self.device)
        # else:
        #     self.deformer_encoder = NetHead(
        #         self.grid_pos_layer * self.concat_channels, self.final_dim
        #     ).to(self.device)

        self.deformer = superpixel_grid.DeformGNN(
            state_dim=self.state_dim,
            feature_channel_num=self.out_feature_channel_num,
            out_dim=self.out_dim,
            layer_num=self.deform_layer_num,
        ).to(self.device)
        # self.superpixel = LatticeVariance(
        #     self.resolution[0],
        #     self.resolution[1],
        #     sigma=self.sigma,
        #     device=device,
        #     add_seg=self.add_mask_variance,
        #     mask_coef=self.mask_coef,
        # )

    def forward(
        self,
        net_input=None,
        base_point=None,
        base_normalized_point_adjacent=None,
        base_point_mask=None,
        base_triangle2point=None,
        base_area_mask=None,
        base_triangle_mask=None,
        crop_gt=None,
        inference=False,
        # timing=False,
        grid_size=20,
    ):

        net_input.shape[0]
        net_input.device
        # variance = torch.zeros(sub_batch_size, device=device)
        # laplacian_loss = torch.zeros(sub_batch_size, device=device)
        # area_variance = torch.zeros(sub_batch_size, device=device)
        # reconstruct_loss = torch.zeros(sub_batch_size, device=device)

        # if timing:
        #     tt('start', sub_batch_size)

        # final_features, final_cat_features = self.encoder(deepcopy(net_input))

        # #############################################################################
        # # Grid Deformation
        # if self.grid_pos_layer == 5:
        #     deformer_feature = self.deformer_encoder(final_features)
        # else:
        #     deformer_feature = self.deformer_encoder(
        #         final_cat_features[:, : self.concat_channels * self.grid_pos_layer, :, :]
        #     )

        # deformer_feature: 1, 512, 128, 256
        # base_point, base_normalized_point_adjacent, base_point_mask : constants
        # output = self.deformer(
        #     deformer_feature, base_point, base_normalized_point_adjacent, base_point_mask
        # )
        output = self.deformer(
            net_input, base_point, base_normalized_point_adjacent, base_point_mask
        )

        # if timing:
        #     tt('get deformation', sub_batch_size)

        output['pred_points']
        output['gcn_pred_points']

        return output
        # return pred_points, gcn_pred_points

        #############################################################################
        # if timing:
        #     tt('get curve prediction', sub_batch_size)

        # n_row_area_normalize = self.grid_size[0]
        # n_column_area_normalize = self.grid_size[1]

        # if not inference:
        #     if self.add_mask_variance:
        #         tmp_gt_mask = deepcopy(crop_gt)
        #         tmp_gt_mask = tmp_gt_mask.long()
        #         gt_mask = gtmask2onehot(tmp_gt_mask).permute(0, 2, 3, 1)
        #         superpixel_ret = self.superpixel(
        #             grid_pos=pred_points,
        #             img_fea=net_input[:, :3, ...].permute(0, 2, 3, 1),
        #             base_triangle2point=base_triangle2point,
        #             base_area_mask=base_area_mask,
        #             base_triangle_mask=base_triangle_mask,
        #             area_normalize=(n_row_area_normalize, n_column_area_normalize),
        #             semantic_mask=gt_mask,
        #             inference=inference,
        #             grid_size=self.grid_size,
        #         )
        # else:
        #     superpixel_ret = self.superpixel(
        #         grid_pos=pred_points,
        #         img_fea=net_input[:, :3, ...].permute(0, 2, 3, 1),
        #         base_triangle2point=base_triangle2point,
        #         base_area_mask=base_area_mask,
        #         base_triangle_mask=base_triangle_mask,
        #         area_normalize=(n_row_area_normalize, n_column_area_normalize),
        #         inference=inference,
        #         grid_size=self.grid_size,
        #     )
        # else:
        #     superpixel_ret = defaultdict(None)

        # if not inference:
        #     condition = superpixel_ret['condition']
        #     laplacian_loss += output['laplacian_energy']
        #     variance += superpixel_ret['variance']
        #     area_variance += superpixel_ret['area_variance']
        #     reconstruct_loss += superpixel_ret['reconstruct_loss']
        # else:
        #     condition = None
        # return condition, laplacian_loss, variance, area_variance, reconstruct_loss, pred_points
