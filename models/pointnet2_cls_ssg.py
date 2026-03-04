import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction
from typing import Sequence, Optional
from pointnet_utils import PointNetEncoder, feature_transform_reguliarzer



###PointNet++
class get_model(nn.Module):
    """
    Standard PointNet++ classification network (geometry branch only).

    Args:
        num_class:       Number of output classes.
        normal_channel:  Kept for API compatibility. We assume input is (B, 3, N)
                         with XYZ only and do not explicitly use normals here.
    """

    def __init__(self, num_class: int, normal_channel: bool = True) -> None:
        super().__init__()
        self.normal_channel = normal_channel  # not used directly, kept for compatibility

        # ---- PointNet++ backbone (geometry only) ----
        # SA layers follow the classic PointNet++ MSG/SSG style
        self.sa1 = PointNetSetAbstraction(
            npoint=512,
            radius=0.2,
            nsample=32,
            in_channel=3,              # xyz only
            mlp=[64, 64, 128],
            group_all=False,
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=128,
            radius=0.4,
            nsample=64,
            in_channel=128 + 3,        # features + xyz
            mlp=[128, 128, 256],
            group_all=False,
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=256 + 3,        # features + xyz
            mlp=[256, 512, 1024],
            group_all=True,
        )

        # ---- Classification head ----
        # Global feature dimension is fixed to 1024 from the last SA layer
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)

        self.fc3 = nn.Linear(256, num_class)

    def forward(self, points_3: torch.Tensor):
        """
        Forward pass.

        Args:
            points_3: Tensor of shape (B, 3, N)
                      Normalised per-point XYZ coordinates.

        Returns:
            logits:    (B, num_class) classification scores.
            l3_points: (B, 1024, 1) global feature map from the last SA layer.
        """
        B, _, _ = points_3.shape
        xyz = points_3
        feats = None  # no additional point-wise features, xyz only

        # ---- PointNet++ set abstraction ----
        l1_xyz, l1_points = self.sa1(xyz, feats)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # (B, 1024, 1)

        # Global geometric descriptor (flatten over the singleton point dimension)
        geom_global = l3_points.view(B, 1024)  # (B, 1024)

        # ---- Classification head ----
        x = self.fc1(geom_global)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        x = self.drop2(x)

        logits = self.fc3(x)
        return logits, l3_points




###PointNet
# class get_model(nn.Module):
#     """
#     Standard PointNet classification network (geometry branch only).
#
#     Args:
#         num_class:       Number of output classes.
#         normal_channel:  If True, the input is assumed to have 6 channels (xyz + normals),
#                          otherwise 3 channels (xyz only). The encoder input channel is
#                          configured accordingly.
#     """
#
#     def __init__(self, num_class: int, normal_channel: bool = False) -> None:
#         super().__init__()
#         self.normal_channel = normal_channel
#
#         # -----------------------------
#         # PointNet backbone (global_feat=True)
#         # -----------------------------
#         # If normals are used, the input channel is 6 (xyz + normals),
#         # otherwise it is 3 (xyz only).
#         in_ch = 6 if normal_channel else 3
#         self.pn = PointNetEncoder(
#             global_feat=True,
#             feature_transform=True,
#             channel=in_ch,
#         )
#
#         # -----------------------------
#         # Classification head
#         # -----------------------------
#         # The PointNet encoder outputs a 1024-dim global feature vector.
#         self.fc1 = nn.Linear(1024, 512)
#         self.bn1 = nn.BatchNorm1d(512)
#         self.drop1 = nn.Dropout(0.4)
#
#         self.fc2 = nn.Linear(512, 256)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.drop2 = nn.Dropout(0.4)
#
#         self.fc3 = nn.Linear(256, num_class)
#
#     def forward(self, points_3: torch.Tensor):
#         """
#         Forward pass.
#
#         Args:
#             points_3: Tensor of shape (B, C, N), where
#                       C = 3  (xyz) if normal_channel == False, or
#                       C = 6  (xyz + normals) if normal_channel == True.
#
#         Returns:
#             logits:   (B, num_class) classification scores.
#             feat_map: (B, 1024, 1) global feature map, kept for compatibility
#                       with the PointNet++ version (l3_points).
#         """
#         B = points_3.shape[0]
#
#         # PointNet encoder:
#         # x:          (B, 1024) global feature
#         # trans:      input transform matrix (not used here)
#         # trans_feat: feature transform matrix (for regularization in the loss)
#         x, trans, trans_feat = self.pn(points_3)
#
#         geom_global = x                            # (B, 1024)
#         feat_map = geom_global.unsqueeze(-1)       # (B, 1024, 1), for API consistency
#
#         # Classification head
#         x = self.fc1(geom_global)
#         x = self.bn1(x)
#         x = F.relu(x, inplace=True)
#         x = self.drop1(x)
#
#         x = self.fc2(x)
#         x = self.bn2(x)
#         x = F.relu(x, inplace=True)
#         x = self.drop2(x)
#
#         logits = self.fc3(x)                      # (B, num_class)
#         return logits, feat_map



# ---------------------------
# group by bridge_id
# ---------------------------
def _group_indices_by_bridge(bridge_ids):
    """Group sample indices within a batch by bridge_id."""
    table = {}
    for i, bid in enumerate(bridge_ids):
        table.setdefault(bid, []).append(i)
    groups = []
    for _, idxs in table.items():
        groups.append(torch.tensor(idxs, dtype=torch.long))
    return groups


# ---------------------------
class MaskedSoftmaxCELoss(nn.Module):
    def __init__(self,
                 num_classes=7,
                 class_weights=None,
                 class_name_to_id=None,
                 # # ----- Height band thresholds (based on relative height w.r.t. z_mid)
                 z_lo=0.3, z_mid1=0.7, z_hi=0.9,
                 # ----- others -----
                 eps=1e-6,
                 relax_gt=True,
                 inf_mask_value=-1e9,
                 Projection_overlap_priors: bool = False,
                 Elevation_based_hierarchical_priors: bool = False,
                 Spatial_adjacency_priors: bool = False
                 ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.eps = float(eps)
        self.relax_gt = bool(relax_gt)
        self.inf_mask_value = float(inf_mask_value)
        self.Projection_overlap_priors = bool(Projection_overlap_priors)
        self.Elevation_based_hierarchical_priors = bool(Elevation_based_hierarchical_priors)
        self.Spatial_adjacency_priors = bool(Spatial_adjacency_priors)


        if class_weights is None:
            self.register_buffer('class_weights', None)
        else:
            self.register_buffer('class_weights', class_weights.float())

        # Name -> id
        name2id = {k.lower(): int(v) for k, v in (class_name_to_id or {}).items()}
        alias = {
            'parapet': 'handrail',
            'pier cap': 'piercap',
            'diapharm': 'diaphragm',
            'grider': 'girder',
        }

        def _id(name):
            n = alias.get(name.lower(), name.lower())
            return name2id.get(n, None)

        self.id_abutment  = _id('abutment')
        self.id_pier      = _id('pier')
        self.id_piercap   = _id('piercap')
        self.id_girder    = _id('girder')
        self.id_deck      = _id('deck')
        self.id_parapet   = _id('parapet')   # handrail
        self.id_diaphragm = _id('diaphragm')

        # Height band partitioning
        self.z_lo, self.z_mid1,  self.z_hi = map(float, (z_lo, z_mid1,  z_hi))


        #
        self.id2name = {}
        for k, v in name2id.items():
            if v is not None:
                self.id2name[int(v)] = k

    @torch.no_grad()
    def apply_mask_at_inference_with_presence(self,
                                              logits,              # (B,C)
                                              coords_raw,          # (B,N,3)
                                              bridge_ids=None):  # list[int] or None
        feasible = self._build_feasible_mask(coords_raw, bridge_ids)   # (B,C) bool
        masked_logits = logits.masked_fill(~feasible, self.inf_mask_value)

        prob = F.softmax(masked_logits, dim=1)
        pred = prob.argmax(dim=1)
        return masked_logits, prob, pred


    #
    def _build_feasible_mask(self, coords_raw, bridge_ids=None):#

        device = coords_raw.device
        B = coords_raw.shape[0]
        C = self.num_classes
        mask = torch.zeros(B, C, dtype=torch.bool, device=device)
        idA, idP, idPC, idG, idD, idDeck, idH = (self.id_abutment, self.id_pier, self.id_piercap, self.id_girder,
                                                 self.id_diaphragm, self.id_deck,
                                                 self.id_parapet)

        ###########Elevation_based_hierarchical_priors（EHP）
        if self.Elevation_based_hierarchical_priors:

            z_vals = coords_raw[:, :, 2]  # (B,N)
            q_low, q_high = 0.0005, 0.9995
            z_bot = torch.quantile(z_vals, q_low, dim=1)  # (B,)
            z_top = torch.quantile(z_vals, q_high, dim=1)  # (B,)
            z_mid = 0.5 * (z_bot + z_top)### Compute the vertical bounding box midpoint (z_mid) of each component

            #
            if bridge_ids is not None:## 全 batch 统一 z_min/z_max
                uniq = set(map(str, bridge_ids))
                if len(uniq) != 1:
                    print("[WARN] mixed bridges in one batch:", uniq)

                ## # Group by bridge_id and adaptively compute z_min/z_max within each group
                groups = _group_indices_by_bridge(bridge_ids)
                z_mid_rel = torch.empty(B, dtype=z_mid.dtype, device=device)#relative height
                for idxs in groups:#
                    idxs = idxs.to(device)
                    z_g = z_vals[idxs]  # (B_g, N)
                    # Compute per-instance quantile bounds within the group
                    z_bot_g = torch.quantile(z_g, q_low, dim=1)  # (B_g,)
                    z_top_g = torch.quantile(z_g, q_high, dim=1)  # (B_g,)
                    # Take min/max of these "robust bounds" to get a group-level robust z-range
                    zmin_g = z_bot_g.min()
                    zmax_g = z_top_g.max()
                    zr_g = (zmax_g - zmin_g).clamp_min(self.eps)

                    z_mid_rel[idxs] = (z_mid[idxs] - zmin_g) / zr_g

            #（1）Elevation_based_hierarchical_priors（EHP）
            seg_sets = [
                [idP],
                [idA, idP, idPC],
                [idA, idPC, idG, idD, idDeck, idH],
                [idDeck, idH]
            ]

            s0 = z_mid_rel < self.z_lo
            s1 = (z_mid_rel >= self.z_lo) & (z_mid_rel < self.z_mid1)
            s3 = (z_mid_rel >= self.z_mid1) & (z_mid_rel < self.z_hi)
            s4 = z_mid_rel >= self.z_hi

            seg_inds = [s0, s1, s3, s4]
            ## For each height band, mark the "allowed candidate classes" as True
            for seg, allow_list in zip(seg_inds, seg_sets):
                if seg.any():##  # at least one sample falls into this band
                    for cls_id in allow_list:## # each class allowed in this band
                        if cls_id is not None and 0 <= cls_id < C:
                            mask[seg, cls_id] = True ## set True for samples in this band at the allowed class indices

        # Precompute pairwise geometric quantities (used for existence checks)
        offdiag = ~torch.eye(B, dtype=torch.bool, device=device)

        if bridge_ids is not None:
            bids = torch.tensor([hash(str(b)) for b in bridge_ids], device=coords_raw.device)
            same_bridge = (bids.view(-1, 1) == bids.view(1, -1))
        else:
            same_bridge = torch.ones(B, B, dtype=torch.bool, device=coords_raw.device)

        offdiag = offdiag & same_bridge

        def _forbid_where(cond, cls_id):
            if cls_id is not None and 0 <= cls_id < C:
                idx = cond & mask[:, cls_id]
                mask[idx, cls_id] = False


        if self.Spatial_adjacency_priors:
            q_low1, q_high1 = 0.0005, 0.9995
            x = coords_raw[:, :, 0]
            y = coords_raw[:, :, 1]
            x_lo = torch.quantile(x, q_low1, dim=1)
            x_hi = torch.quantile(x, q_high1, dim=1)
            y_lo = torch.quantile(y, q_low1, dim=1)
            y_hi = torch.quantile(y, q_high1, dim=1)
            #  --- Axial overlap width ---
            overlap_x = torch.minimum(x_hi.view(-1,1), x_hi.view(1,-1)) - torch.maximum(x_lo.view(-1,1), x_lo.view(1,-1))######################
            overlap_y = torch.minimum(y_hi.view(-1,1), y_hi.view(1,-1)) - torch.maximum(y_lo.view(-1,1), y_lo.view(1,-1))######################

            overl_x_ok = (overlap_x > 0)
            overl_y_ok = (overlap_y > 0)
            adj_xy_overlap = (overl_x_ok & overl_y_ok & offdiag)


            z_vals = coords_raw[:, :, 2]  # (B,N)
            z_bot_adj = torch.quantile(z_vals, q_low1, dim=1)
            z_top_adj = torch.quantile(z_vals, q_high1, dim=1)

            Z_ABS_NEAR_UP   = torch.as_tensor(1, device=coords_raw.device, dtype=coords_raw.dtype)
            Z_ABS_NEAR_DOWN = torch.as_tensor(1, device=coords_raw.device, dtype=coords_raw.dtype)

            # Upper neighbor j of i: gap_up_signed = z_bot_adj[j] - z_top_adj[i]
            gap_up_signed = z_bot_adj.view(1, -1) - z_top_adj.view(-1, 1)  # (B,B)
            # Lower neighbor j of i: gap_down_signed = z_bot_adj[i] - z_top_adj[j]
            gap_down_signed = z_bot_adj.view(-1, 1) - z_top_adj.view(1, -1)  # (B,B)

            #
            j_bot_gt_i_bot = (z_bot_adj.view(1, -1) > z_bot_adj.view(-1, 1))
            j_bot_lt_i_bot = (z_bot_adj.view(1, -1) < z_bot_adj.view(-1, 1))
            near_and_above = (gap_up_signed.abs() <= Z_ABS_NEAR_UP) & j_bot_gt_i_bot
            near_and_below = (gap_down_signed.abs() <= Z_ABS_NEAR_DOWN) & j_bot_lt_i_bot

            # --- Final XYZ adjacency: XY must overlap + Z-direction proximity + correct vertical direction ---
            def _exists_adjacent_upperXYZ_overlap():
                cond = adj_xy_overlap & near_and_above
                return cond.any(dim=1)  # (B,)

            def _exists_adjacent_lowerXYZ_overlap():
                cond = adj_xy_overlap & near_and_below
                # cond =  near_and_below
                return cond.any(dim=1)  # (B,)

            exist_adj_upper = _exists_adjacent_upperXYZ_overlap()
            exist_adj_lower = _exists_adjacent_lowerXYZ_overlap()


            def _need_upper(cls_id):
                if (cls_id is not None) and (0 <= cls_id < C):
                    _forbid_where(~exist_adj_upper, cls_id)

            def _need_lower(cls_id):
                if (cls_id is not None) and (0 <= cls_id < C):
                    _forbid_where(~exist_adj_lower, cls_id)

            def _forbid_lower(cls_id):
                if (cls_id is not None) and (0 <= cls_id < C):
                    _forbid_where(exist_adj_lower, cls_id)

            #  Abutment: has an upper neighbor, no lower neighbor ----
            _need_upper(self.id_abutment)
            _forbid_lower(self.id_abutment)
            # #
            # Pier: has an upper neighbor, no lower neighbor ----
            _need_upper(self.id_pier)  # 1
            _forbid_lower(self.id_pier)
            #
            # Pier cap: has an upper neighbor and a lower neighbor ----
            _need_upper(self.id_piercap)
            _need_lower(self.id_piercap)
            # #
            # # # Girder: has an upper neighbor and a lower neighbor ----
            _need_upper(self.id_girder)
            _need_lower(self.id_girder)
            # # # # # #
            # Deck: has a lower neighbor (upper neighbor not required) ----
            _need_lower(self.id_deck)
            # # # # # # # #
            # Handrail/Parapet: has a lower neighbor (upper neighbor not required) ----
            _need_lower(self.id_parapet)
            # # # #
            _need_upper(self.id_diaphragm)  ### Diaphragm with no component below middle region


        #
        if self.Projection_overlap_priors:

            q_low1, q_high1 = 0.0005, 0.9995

            x_raw = coords_raw[:, :, 0]  # (B, N)
            y_raw = coords_raw[:, :, 1]  # (B, N)
            z_raw = coords_raw[:, :, 2]  # (B, N)

            x_min_raw = torch.quantile(x_raw, q_low1, dim=1)  # (B,)
            x_max_raw = torch.quantile(x_raw, q_high1, dim=1)  # (B,)
            y_min_raw = torch.quantile(y_raw, q_low1, dim=1)  # (B,)
            y_max_raw = torch.quantile(y_raw, q_high1, dim=1)  # (B,)
            z_min_raw = torch.quantile(z_raw, q_low1, dim=1)  # (B,)
            z_max_raw = torch.quantile(z_raw, q_high1, dim=1)  # (B,)

            # --- Axial overlaps (X and Z, using raw/original bounding boxes) ---
            overlap_x_xz_raw = torch.minimum(x_max_raw.view(-1,1), x_max_raw.view(1,-1)) - \
                               torch.maximum(x_min_raw.view(-1,1), x_min_raw.view(1,-1))
            overlap_z_xz_raw = torch.minimum(z_max_raw.view(-1,1), z_max_raw.view(1,-1)) - \
                               torch.maximum(z_min_raw.view(-1,1), z_min_raw.view(1,-1))


            width_x_raw = (x_max_raw - x_min_raw).clamp_min(self.eps)  # (B,)
            width_z_raw = (z_max_raw - z_min_raw).clamp_min(self.eps)  # (B,)

            inter_area_xz_raw = overlap_x_xz_raw.clamp_min(0) * overlap_z_xz_raw.clamp_min(0)
            area_i_xz_raw = width_x_raw.view(-1,1) * width_z_raw.view(-1,1)

            # --- XZ projection "coverage ratio" ---
            cover_ratio_xz_raw = inter_area_xz_raw / area_i_xz_raw.clamp_min(self.eps)
            cover_ratio_xz_raw = cover_ratio_xz_raw * offdiag
            max_cover_i_xz_raw = cover_ratio_xz_raw.max(dim=1).values  # (B,)

            # Constraint: necessary condition based on diaphragm XZ projection coverage ratio ---
            xz_cover_min_for_D = 0.9
            if (self.id_diaphragm is not None) and (0 <= self.id_diaphragm < C):
                _forbid_where(max_cover_i_xz_raw < float(xz_cover_min_for_D), self.id_diaphragm)


            x_lo = x_min_raw; x_hi = x_max_raw
            y_lo = y_min_raw; y_hi = y_max_raw

            overlap_x = torch.minimum(x_hi.view(-1, 1), x_hi.view(1, -1)) - torch.maximum(x_lo.view(-1, 1), x_lo.view(1,-1))
            overlap_y = torch.minimum(y_hi.view(-1, 1), y_hi.view(1, -1)) - torch.maximum(y_lo.view(-1, 1), y_lo.view(1,-1))

            width_x = (x_hi - x_lo).clamp_min(self.eps)  # (B,)
            width_y = (y_hi - y_lo).clamp_min(self.eps)  # (B,)
            area_i = width_x.view(-1, 1) * width_y.view(-1, 1)

            inter_area = overlap_x.clamp_min(0) * overlap_y.clamp_min(0)  # (B,B)

            xy_cover_frac=0.9
            xy_cover_frac = float(xy_cover_frac)
            apply_cover_for_ids =[idP,idD,idPC]

            cover_ratio = inter_area / area_i.clamp_min(self.eps)  # (B,B)
            cover_ratio = cover_ratio * offdiag
            max_cover_i, _ = cover_ratio.max(dim=1)  # (B,)

            def _need_xy_cover_for(cls_id):
                if (cls_id is not None) and (0 <= cls_id < C):
                    _forbid_where(max_cover_i < xy_cover_frac, cls_id)

            for cls_id in apply_cover_for_ids:
                _need_xy_cover_for(cls_id)


            x_lo_b = x_lo.min()
            x_hi_b = x_hi.max()
            width_bridge_x = (x_hi_b - x_lo_b).clamp_min(self.eps)
            # Instance-wise length along the X direction
            width_x_i = (x_hi - x_lo).clamp_min(self.eps)  # (B,)
            # Fraction of the full-bridge X span covered by each instance
            frac_x = width_x_i / width_bridge_x  # (B,)
            thr_x = 0.50
            if (idDeck is not None) and (0 <= idDeck < C):
                _forbid_where(frac_x < thr_x, idDeck)  # samples with X-length < 50% cannot be classified as deck
            if (idG is not None) and (0 <= idG < C):
                _forbid_where(frac_x < thr_x, idG)


        empty_rows = ~mask.any(dim=1)
        if empty_rows.any():
            num_empty = int(empty_rows.sum())
            print(
                f"[WARN] feasible mask is all-False on {num_empty}/{B} samples "
                f"(all classes forbidden). bridge_ids_given={bridge_ids is not None}"
            )

        return mask

    # ---------- Main loss ----------
    def forward(self,
                pred, target, trans_feat=None,
                coords_raw=None,          # (B,N,3)
                bridge_ids=None,          # list[str] or None
                class_weights_override=None):

        assert coords_raw is not None, "Masked-Softmax requires coords_raw=(B,N,3) to build the feasible mask"
        device = pred.device
        B, C = pred.shape

        feasible = self._build_feasible_mask(coords_raw, bridge_ids)  # (B,C) bool


        if self.relax_gt:
            row = torch.arange(B, device=device, dtype=torch.long)
            feasible[row, target] = True

        masked_logits = pred.masked_fill(~feasible, self.inf_mask_value)

        cw = class_weights_override if class_weights_override is not None else self.class_weights
        if cw is not None and cw.device != pred.device:
            cw = cw.to(pred.device, non_blocking=True)

        return F.cross_entropy(masked_logits, target, weight=cw)



def get_loss(num_classes=7,
             class_weights=None,
             class_name_to_id=None,
             eps=1e-6,
             z_lo=0.3, z_mid1=0.7,  z_hi=0.9,
             relax_gt=True,
             Projection_overlap_priors: bool = False,
             Elevation_based_hierarchical_priors: bool = False,
             Spatial_adjacency_priors: bool = False
             ):

    assert num_classes is not None, "num_classes must be provided"
    assert class_name_to_id is not None, "class_name_to_id is required to build the class ID mapping"

    return MaskedSoftmaxCELoss(
        num_classes=num_classes,
        class_weights=class_weights,
        class_name_to_id=class_name_to_id,
        z_lo=z_lo, z_mid1=z_mid1,  z_hi=z_hi,
        eps=eps,
        relax_gt=relax_gt,
        Projection_overlap_priors=Projection_overlap_priors,
        Elevation_based_hierarchical_priors=Elevation_based_hierarchical_priors,
        Spatial_adjacency_priors=Spatial_adjacency_priors
    )