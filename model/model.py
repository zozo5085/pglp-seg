from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import clip
import math
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def focal_loss_multiclass(logits, targets, gamma=2.0, alpha=None, reduction="mean"):
    ce = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce)

    loss = ((1.0 - pt).clamp_min(1e-6) ** float(gamma)) * ce

    if alpha is not None:
        if torch.is_tensor(alpha):
            alpha_t = alpha.to(device=logits.device, dtype=logits.dtype)

            if alpha_t.numel() == targets.numel():
                alpha_t = alpha_t.view_as(loss)
            elif alpha_t.numel() == logits.size(1):
                alpha_t = alpha_t[targets]
            elif alpha_t.numel() == 1:
                alpha_t = alpha_t.expand_as(loss)
            else:
                raise ValueError(
                    f"Invalid alpha shape: alpha.numel()={alpha_t.numel()}, "
                    f"targets.numel()={targets.numel()}, num_classes={logits.size(1)}"
                )

            loss = alpha_t * loss

        elif isinstance(alpha, (list, tuple)):
            alpha_t = logits.new_tensor(alpha)

            if alpha_t.numel() == targets.numel():
                alpha_t = alpha_t.view_as(loss)
            elif alpha_t.numel() == logits.size(1):
                alpha_t = alpha_t[targets]
            elif alpha_t.numel() == 1:
                alpha_t = alpha_t.expand_as(loss)
            else:
                raise ValueError(
                    f"Invalid alpha shape: alpha.numel()={alpha_t.numel()}, "
                    f"targets.numel()={targets.numel()}, num_classes={logits.size(1)}"
                )

            loss = alpha_t * loss

        else:
            loss = float(alpha) * loss

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

def load_state_dict_flexible(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            ckpt = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        elif "model" in ckpt:
            ckpt = ckpt["model"]

    if not isinstance(ckpt, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(ckpt)}")

    return ckpt


def load_prompt_token_flexible(path):
    sd = load_state_dict_flexible(path)

    candidates = [
        "module.text_encoder.prompt_token",
        "text_encoder.prompt_token",
    ]

    for k in candidates:
        if k in sd:
            return sd[k]

    raise KeyError(
        "Cannot find prompt_token in checkpoint. "
        f"Tried {candidates}. "
        f"First keys: {list(sd.keys())[:20]}"
    )

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", nn.GELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=True, attn_mask=None)[0]

    def forward(self, x: torch.Tensor):
        y = self.ln_1(x)

        attn_out, attn_weight = self.attn(
            y, y, y,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True
        )

        self.last_attn_weight = attn_weight.detach()

        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))

        return x

    def _initialize_weights(self, clip_model, i):
        self.ln_1 = clip_model.visual.transformer.resblocks[i].ln_1
        self.ln_1.eps = 1e-06
        self.attn = clip_model.visual.transformer.resblocks[i].attn.to(torch.float32)
        self.attn.batch_first = True
        self.mlp = clip_model.visual.transformer.resblocks[i].mlp.to(torch.float32)
        # self.mlp[1] = nn.GELU()
        self.ln_2 = clip_model.visual.transformer.resblocks[i].ln_2
        self.ln_2.eps = 1e-06
        for p in self.parameters():
            p.requires_grad = False
        return


class LastResidualAttentionBlock(nn.Module):
    def __init__(self, clip_model: clip, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", nn.GELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.lrab_alpha = 0.02 #0.05 >> 0.02
        self.lrab_lambda_out = 1.0

        self._initialize_weights(clip_model)

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        y = self.ln_1(x)

        # Raw CLIP QKV before out_proj: [N, L, 3*C]
        qkv = F.linear(y, self.attn.in_proj_weight, self.attn.in_proj_bias)
        N, L, C3 = qkv.shape
        C = C3 // 3

        # [3, N, L, C]
        qkv = qkv.view(N, L, 3, C).permute(2, 0, 1, 3)
        q_raw, k_raw, v_raw = qkv[0], qkv[1], qkv[2]

        # qkv_out = qkv.reshape(3 * N, L, C)
        qkv_out = qkv.contiguous().reshape(3 * N, L, C)
        qkv_out = F.linear(qkv_out, self.attn.out_proj.weight, self.attn.out_proj.bias)
        q, k, v_ori = qkv_out.tensor_split(3, dim=0)

        v_ori = v_ori + x
        v_ori = v_ori + self.mlp(self.ln_2(v_ori))

        x_ori = x + self.attention(y)
        x_ori = x_ori + self.mlp(self.ln_2(x_ori))

        d_k = q_raw.size(-1)
        scores = torch.matmul(q_raw, k_raw.transpose(-2, -1)) / math.sqrt(d_k)
        attn_weights = F.softmax(scores, dim=-1)

        context = torch.matmul(attn_weights, v_raw)
        context = F.linear(
            context,
            self.attn.out_proj.weight,
            self.attn.out_proj.bias
        )

        v_ssa = x + context
        v_ssa = v_ssa + self.mlp(self.ln_2(v_ssa))

        alpha = float(self.lrab_alpha)
        v = (1.0 - alpha) * v_ori + alpha * v_ssa

        out = [x_ori, q, k, v]
        return out

    def _initialize_weights(self, clip_model):
        self.ln_1 = clip_model.visual.transformer.resblocks[11].ln_1
        self.ln_1.eps = 1e-06
        self.attn = clip_model.visual.transformer.resblocks[11].attn.to(torch.float32)
        self.attn.batch_first = True
        self.mlp = clip_model.visual.transformer.resblocks[11].mlp.to(torch.float32)
        # self.mlp[1] = nn.GELU()
        self.ln_2 = clip_model.visual.transformer.resblocks[11].ln_2
        self.ln_2.eps = 1e-06
        for p in self.parameters():
            p.requires_grad = False
        return


class Transformer(nn.Module):
    def __init__(self, clip_model: clip, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblock = []
        for i in range(self.layers - 1):
            self.resblock.append(ResidualAttentionBlock(width, heads, attn_mask))
        self.resblock.append(LastResidualAttentionBlock(clip_model, width, heads, attn_mask))
        self._initialize_weights(clip_model)
        self.resblocks = nn.Sequential(*self.resblock)

    def forward(self, x: torch.Tensor):
        z, q, k, v = self.resblocks(x)

        sfp_attn = getattr(self.resblock[-2], "last_attn_weight", None)

        return z, q, k, v, sfp_attn

    def _initialize_weights(self, clip_model):
        for i in range(self.layers - 1):
            self.resblock[i]._initialize_weights(clip_model, i)
        return


class VisionTransformer(nn.Module):
    def __init__(self, clip_model: clip, input_resolution: int, patch_size: int, width: int, layers: int, heads: int,
                 output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.dilation = [1, 1]
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=self.patch_size, stride=self.patch_size,
                               bias=False)

        self.cls_token = torch.load('utils/cls_token.pt')

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = nn.LayerNorm(width)

        self.transformer = Transformer(clip_model, width, layers, heads)

        self.ln_post = nn.LayerNorm(width)
        self.proj = clip_model.visual.proj.to(torch.float32)
        self._initialize_weights(clip_model)

    def compute_sfp_score(self, attn_weight, output_h, output_w):
        if attn_weight is None:
            return None

        if attn_weight.dim() == 4:
            attn_weight = attn_weight.mean(dim=1)

        # attn_weight: [B, L, L]
        cls_to_patch = attn_weight[:, 0, 1:]  # [B, HW]
        patch_self = torch.diagonal(
            attn_weight[:, 1:, 1:],
            dim1=-2,
            dim2=-1
        )  # [B, HW]

        sfp_score = cls_to_patch - patch_self
        sfp_score = sfp_score.reshape(attn_weight.shape[0], output_h, output_w)

        return sfp_score

    def forward(self, x, train=False, img_metas=None):
        B = x.shape[0]
        # PatchEmbedding
        # Padding
        input_h, input_w = x.size()[-2:]
        kernel_h, kernel_w = (self.patch_size, self.patch_size)
        stride_h, stride_w = (self.patch_size, self.patch_size)
        output_h = math.ceil(input_h / stride_h)
        output_w = math.ceil(input_w / stride_w)
        pad_h = max((output_h - 1) * stride_h +
                    (kernel_h - 1) * self.dilation[0] + 1 - input_h, 0)
        pad_w = max((output_w - 1) * stride_w +
                    (kernel_w - 1) * self.dilation[1] + 1 - input_w, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [0, pad_w, 0, pad_h])
        x = x.to(device)
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        # cls_tokens = self.class_embedding.reshape(1, 1, 1024).expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Positional Embedding
        positional_embedding = self.positional_embedding
        positional_embedding = positional_embedding.unsqueeze(dim=0)
        pos_h = self.input_resolution // self.patch_size
        pos_w = self.input_resolution // self.patch_size
        cls_token_weight = positional_embedding[:, 0]
        pos_embed_weight = positional_embedding[:, (-1 * pos_h * pos_w):]
        pos_embed_weight = pos_embed_weight.reshape(
            1, pos_h, pos_w, positional_embedding.shape[2]).permute(0, 3, 1, 2)
        pos_embed_weight = F.interpolate(pos_embed_weight, size=(output_h, output_w), mode='bicubic',
                                         align_corners=False)
        cls_token_weight = cls_token_weight.unsqueeze(1)
        pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
        positional_embedding = torch.cat((cls_token_weight, pos_embed_weight), dim=1)

        x = x + positional_embedding

        x = self.ln_pre(x)

        # x = x.permute(1, 0, 2)  # NLD -> LND
        # x, q, k, v = self.transformer(x)
        x, q, k, v, sft_attn = self.transformer(x)
        sfp_score = self.compute_sfp_score(sft_attn, output_h, output_w)

        x = self.ln_post(x)
        v = self.ln_post(v)

        out = x[:, 1:]
        B, _, C = out.shape
        out = out.reshape(B, output_h, output_w,
                          C).permute(0, 3, 1, 2).contiguous()
        q = q[:, 1:]
        k = k[:, 1:]
        v = v[:, 1:]
        v = v.reshape(B, output_h, output_w, -1).permute(0, 3, 1, 2).contiguous()

        out = [out, q, k, v]
        cls_token = x[:, 0]

        if self.proj is not None:
            z_global = cls_token @ self.proj

        # return [v, (output_h, output_w), z_global, k]
        return [v, (output_h, output_w), z_global, k, positional_embedding[:, 1:, :], sfp_score]

    def _initialize_weights(self, clip_model):
        self.conv1 = clip_model.visual.conv1.to(torch.float32)
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding
        self.ln_pre = clip_model.visual.ln_pre
        self.ln_post = clip_model.visual.ln_post
        # tune CLIP
        for p in self.parameters():
            p.requires_grad = False
        return


class TextEncoder(nn.Module):
    def __init__(self, clip_model, training=False, cfg=None, device=None):
        super().__init__()
        self.transformer = clip_model.transformer.to(torch.float32)
        self.token_embedding = clip_model.token_embedding.to(torch.float32)
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection.to(torch.float32)
        self.dtype = torch.float32
        self.device = device
        
        token = torch.zeros((1, 73), dtype=torch.int).to(self.device)
        prompt_token = self.token_embedding(token)
        for p in self.parameters():
            p.requires_grad = False
        self.prompt_token = nn.Parameter(prompt_token)
        self.weight = False

        if not training:
            # prompt_token = torch.load(cfg.LOAD_PATH)['text_encoder.prompt_token']
            # prompt_token = torch.load(cfg.LOAD_PATH)['module.text_encoder.prompt_token']
            prompt_token = load_prompt_token_flexible(cfg.LOAD_PATH)
            self.prompt_token = nn.Parameter(prompt_token.float(), requires_grad=False)
            # self.prompt_token = nn.Parameter(prompt_token, requires_grad=False)

    def forward(self, cls_name_token):
        device = self.device
        prompt_token = self.prompt_token.repeat(cls_name_token.shape[0], 1, 1).to(device)
        cls_name_token = cls_name_token.to(device)

        start_token = self.token_embedding(torch.tensor(49406, dtype=torch.int, device=device)).repeat(
            cls_name_token.shape[0], 1, 1).to(device)
        cls_token = self.token_embedding(cls_name_token).to(device)
        x = torch.cat([start_token, prompt_token, cls_token], dim=1)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), 74 + cls_name_token.argmax(dim=-1)] @ self.text_projection
        return x


class DomainTransformRecursiveFilter(nn.Module):
    def __init__(self, sigma_s=30.0, sigma_r=0.30, num_iterations=1):
        super().__init__()
        self.sigma_s = float(sigma_s)
        self.sigma_r = float(sigma_r)
        self.num_iterations = int(num_iterations)

    def forward(self, img, joint_image):
        B, C, H, W = img.shape
        device = img.device
        dtype = img.dtype

        guide = joint_image.to(device=device, dtype=dtype)

        if guide.shape[-2:] != (H, W):
            guide = F.interpolate(
                guide,
                size=(H, W),
                mode="bilinear",
                align_corners=False
            )

        guide_min = guide.amin(dim=(2, 3), keepdim=True)
        guide_max = guide.amax(dim=(2, 3), keepdim=True)
        guide = (guide - guide_min) / (guide_max - guide_min).clamp_min(1e-6)

        grad_x = torch.abs(guide[:, :, :, 1:] - guide[:, :, :, :-1])
        grad_y = torch.abs(guide[:, :, 1:, :] - guide[:, :, :-1, :])

        grad_x = grad_x.sum(dim=1, keepdim=True)
        grad_y = grad_y.sum(dim=1, keepdim=True)

        ct_x = 1.0 + (self.sigma_s / max(self.sigma_r, 1e-6)) * grad_x
        ct_y = 1.0 + (self.sigma_s / max(self.sigma_r, 1e-6)) * grad_y

        ct_x = F.pad(ct_x, (1, 0, 0, 0), value=1.0)
        ct_y = F.pad(ct_y, (0, 0, 1, 0), value=1.0)

        out = img.clone()
        num_iter = max(int(self.num_iterations), 1)

        for i in range(num_iter):
            sigma_h = self.sigma_s * (
                math.sqrt(3.0) * (2.0 ** (num_iter - i - 1))
            ) / math.sqrt((4.0 ** num_iter) - 1.0)
            sigma_h = max(sigma_h, 1e-6)

            feedback = math.exp(-math.sqrt(2.0) / sigma_h)
            log_feedback = math.log(max(feedback, 1e-12))

            a_x = torch.exp(ct_x * log_feedback)
            a_y = torch.exp(ct_y * log_feedback)

            out = self._filter_horizontal(out, a_x)
            out = self._filter_vertical(out, a_y)

        return out

    def _filter_horizontal(self, img, a):
        B, C, H, W = img.shape
        out = img.clone()

        for w in range(1, W):
            out[:, :, :, w] = img[:, :, :, w] + a[:, :, :, w] * (
                out[:, :, :, w - 1] - img[:, :, :, w]
            )

        for w in range(W - 2, -1, -1):
            out[:, :, :, w] = out[:, :, :, w] + a[:, :, :, w + 1] * (
                out[:, :, :, w + 1] - out[:, :, :, w]
            )

        return out

    def _filter_vertical(self, img, a):
        B, C, H, W = img.shape
        out = img.clone()

        for h in range(1, H):
            out[:, :, h, :] = img[:, :, h, :] + a[:, :, h, :] * (
                out[:, :, h - 1, :] - img[:, :, h, :]
            )

        for h in range(H - 2, -1, -1):
            out[:, :, h, :] = out[:, :, h, :] + a[:, :, h + 1, :] * (
                out[:, :, h + 1, :] - out[:, :, h, :]
            )

        return out


class PGLP_Seg(nn.Module):
    def __init__(self, cfg, clip_model, rank, zeroshot_weights=None):
        super(PGLP_Seg, self).__init__()
        self.vit = VisionTransformer(clip_model=clip_model,
                                     input_resolution=224,
                                     patch_size=16,
                                     width=768,
                                     layers=12,
                                     heads=12,
                                     output_dim=768)

        self.clip = clip_model
        self.k = cfg.DATASET.K
        visual_channel = cfg.MODEL.VISUAL_CHANNEL
        text_channel = cfg.MODEL.TEXT_CHANNEL
        self.proj = nn.Conv2d(visual_channel, text_channel, 1, bias=False)
        self._initialize_weights(clip_model)

        self.logit_scale = clip_model.logit_scale
        for p in self.parameters():
            p.requires_grad = False
        self.text_encoder = TextEncoder(clip_model, training=cfg.MODEL.TRAINING, cfg=cfg, device=rank)
        self.cnum = cfg.DATASET.NUM_CLASSES
        self.device = rank

        self.use_focal_loss = True
        self.use_mask_area_adaptive_focal = False
        self.focal_gamma = 2.0
        self.focal_alpha = None

        self.sfp_enable = True
        self.sfp_topk = 800
        self.sfp_min_score = -1e9
        self.sfp_logit_beta = 0.55
        self.sfp_conf_thd = 0.97
        self.sfp_conf_scale = 10.0
        
        self.sfp_margin_enable = False
        self.sfp_margin_lambda = 0.30
        self.sfp_margin_hard_enable = False
        self.sfp_margin_thd = 0.20

        self.sfp_debug = False
        self.sfp_debug_count = 0

        self.sfp_last_stats_batch = []
        self.sfp_last_outlier_mask = None

        self.sfp_debug_export = False
        self.sfp_debug_maps = {}

        self.sfp_proxy_enable = True
        self.sfp_proxy_lambda = 2.00
        self.sfp_proxy_conf_thd = 0.95
        self.sfp_proxy_kernel = 5

        self.sfp_fbls_enable = False
        self.sfp_fbls_kernel = 5
        self.sfp_fbls_beta = 0.20
        self.sfp_fbls_conf_thd = 0.95
        self.sfp_fbls_sigma_color = 0.15
        self.sfp_fbls_sigma_spatial = 2.0
        self.sfp_fbls_eps = 1e-6

        self.sfp_dtlr_enable = True
        self.sfp_dtlr_beta = 1.20
        self.sfp_dtlr_sigma_s = 70.0
        self.sfp_dtlr_sigma_r = 1.50
        self.sfp_dtlr_num_iter = 1
        self.sfp_dtlr_boundary_only = False

        self.sfp_dtlr_structure_protect_enable = True
        self.sfp_dtlr_structure_gain_thd = 0.00
        self.sfp_dtlr_structure_classes = (4, 8, 10)  # bottle, chair, diningtable

        self.sfp_dtlr_class_beta_enable = False
        self.sfp_dtlr_structure_beta_scale = 0.60
        self.sfp_dtlr_class_beta_classes = (4, 8, 10)
        self.sfp_dtlr_class_beta_scale = 0.75
        # Attribute residual correction.
        self.sfp_attr_enable = True
        self.sfp_attr_class_eta_enable = True
        self.sfp_attr_eta_chair = 40.00
        self.sfp_attr_eta_table = 32.00
        self.sfp_attr_eta = 36.00
        self.sfp_attr_topm = 5
        self.sfp_attr_apply_classes = (8, 10)  # chair, diningtable
        self.sfp_attr_conflict_only = False
        self.sfp_attr_conflict_mode = "both"  # "before", "both", or "after"

        self.sfp_attr_directed_enable = False
        self.sfp_attr_directed_positive_only = False

        self.sfp_attr_positive_only = True
        self.sfp_attr_negative_only = False

        self.sfp_attr_prompt_template = "a photo of a {}, which has {}"

        self.sfp_attr_class_names = {
            0: "aeroplane",
            1: "bicycle",
            2: "bird",
            3: "boat",
            4: "bottle",
            5: "bus",
            6: "car",
            7: "cat",
            8: "chair",
            9: "cow",
            10: "dining table",
            11: "dog",
            12: "horse",
            13: "motorbike",
            14: "person",
            15: "potted plant",
            16: "sheep",
            17: "sofa",
            18: "train",
            19: "tv monitor",
        }

        self.sfp_attr_default_attributes = {
            4: [
                "a narrow neck",
                "a cylindrical body",
                "a cap",
                "a transparent or plastic container",
                "used for holding liquid",
                "a vertical elongated shape",
            ],
            8: [
                "a seat",
                "a backrest",
                "legs",
                "armrests",
                "used for sitting",
                "made of wood metal or plastic",
            ],
            10: [
                "a flat tabletop",
                "legs",
                "used for placing food",
                "a horizontal tabletop",
            ],
        }
        self.sfp_attr_bank = None
        self.sfp_attr_bank_classes = None
        self.sfp_attr_bank_device = None

        self.sfp_dtlr_filter = DomainTransformRecursiveFilter(
            sigma_s=self.sfp_dtlr_sigma_s,
            sigma_r=self.sfp_dtlr_sigma_r,
            num_iterations=self.sfp_dtlr_num_iter
        )

        if cfg.MODEL.TRAINING:
            self.pe_proj = nn.Conv2d(768, 512, kernel_size=1)

            # decoder
            self.decoder_conv2 = nn.Conv2d(512 + self.cnum, self.cnum, kernel_size=5, padding=2, stride=1)
            nn.init.kaiming_normal_(self.decoder_conv2.weight, a=0, mode='fan_out', nonlinearity='relu')
            self.decoder_norm2 = nn.BatchNorm2d(self.cnum)
            nn.init.constant_(self.decoder_norm2.weight, 1)
            nn.init.constant_(self.decoder_norm2.bias, 0)

        else:
            self.pe_proj = nn.Conv2d(768, 512, kernel_size=1)
            self.decoder_conv2 = nn.Conv2d(self.cnum + 512, self.cnum, kernel_size=5, padding=2, stride=1)
            self.decoder_norm2 = nn.BatchNorm2d(self.cnum)

    def sfp_logit_purify(self, output, sfp_score):
        """
        Proxy-guided logit purification.

        output:    [B, C, H, W]
        sfp_score: [B, Hs, Ws]

        The selected unreliable-region mask is cached in self.sfp_last_outlier_mask
        for later local refinement.
        """
        self.sfp_last_outlier_mask = None
        self.sfp_last_stats_batch = []

        if (not getattr(self, "sfp_enable", False)) or sfp_score is None:
            return output

        B, C, H, W = output.shape
        device = output.device
        dtype = output.dtype

        sfp_score = sfp_score.to(device=device, dtype=dtype)
        if sfp_score.shape[-2:] != (H, W):
            sfp_score = F.interpolate(
                sfp_score.unsqueeze(1),
                size=(H, W),
                mode="nearest"
            ).squeeze(1)

        with torch.no_grad():
            prob = torch.softmax(output * float(self.sfp_conf_scale), dim=1)
            conf = prob.max(dim=1)[0]  # [B, H, W]

            # Class ambiguity: small top-1/top-2 margin means the token is
            top2_prob = torch.topk(prob, k=2, dim=1).values  # [B, 2, H, W]
            margin = top2_prob[:, 0] - top2_prob[:, 1]       # [B, H, W], in [0, 1]

            flat_score = sfp_score.reshape(B, -1)
            flat_conf = conf.reshape(B, -1)
            flat_margin = margin.reshape(B, -1)

            # Original valid region: score-valid and low-confidence.
            outlier_flat = torch.zeros_like(flat_score, dtype=torch.bool)

            margin_enable = bool(getattr(self, "sfp_margin_enable", False))
            margin_lambda = float(getattr(self, "sfp_margin_lambda", 0.30))
            margin_hard_enable = bool(getattr(self, "sfp_margin_hard_enable", False))
            margin_thd = float(getattr(self, "sfp_margin_thd", 0.20))

            if margin_enable:
                score_min = flat_score.amin(dim=1, keepdim=True)
                score_max = flat_score.amax(dim=1, keepdim=True)
                flat_score_rank = (flat_score - score_min) / (score_max - score_min).clamp_min(1e-6)
                rank_score = flat_score_rank + margin_lambda * (1.0 - flat_margin)
            else:
                rank_score = flat_score

            for b in range(B):
                valid = (
                    (flat_score[b] > float(self.sfp_min_score)) &
                    (flat_conf[b] < float(self.sfp_conf_thd))
                )

                if margin_enable and margin_hard_enable:
                    valid = valid & (flat_margin[b] < margin_thd)

                if not valid.any():
                    continue

                k = min(int(self.sfp_topk), int(valid.sum().item()))
                score_b = rank_score[b].masked_fill(~valid, float("-inf"))
                topk_idx = torch.topk(score_b, k=k, dim=0).indices
                outlier_flat[b, topk_idx] = True

            outlier_mask = outlier_flat.reshape(B, H, W)
            self.sfp_last_outlier_mask = outlier_mask.detach().float()

            if getattr(self, "sfp_debug_export", False):
                self.sfp_debug_maps["sfp_score"] = sfp_score.detach().float().cpu()
                self.sfp_debug_maps["sfp_outlier_mask"] = outlier_mask.detach().float().cpu()
                self.sfp_debug_maps["sfp_confidence"] = conf.detach().float().cpu()
                self.sfp_debug_maps["sfp_margin"] = margin.detach().float().cpu()

        update_mask = outlier_mask.to(device=device, dtype=dtype).unsqueeze(1)
        keep_mask = 1.0 - update_mask

        if update_mask.sum() < 1:
            return output

        # 8-neighbor local proxy target.
        kernel = torch.ones((1, 1, 3, 3), device=device, dtype=dtype)
        kernel[:, :, 1, 1] = 0.0

        neigh_sum = F.conv2d(
            output * keep_mask,
            kernel.expand(C, 1, 3, 3),
            padding=1,
            groups=C
        )
        neigh_count = F.conv2d(keep_mask, kernel, padding=1).clamp_min(1.0)
        neigh_mean = neigh_sum / neigh_count

        refined_target = neigh_mean
        proxy_available = torch.zeros((B, 1, H, W), device=device, dtype=dtype)

        # Optional local high-confidence proxy.
        if getattr(self, "sfp_proxy_enable", False):
            proxy_kernel_size = int(getattr(self, "sfp_proxy_kernel", 5))
            if proxy_kernel_size not in (3, 5):
                raise ValueError(
                    f"Unsupported sfp_proxy_kernel={proxy_kernel_size}. Use 3 or 5."
                )

            proxy_padding = proxy_kernel_size // 2
            proxy_kernel = torch.ones(
                (1, 1, proxy_kernel_size, proxy_kernel_size),
                device=device,
                dtype=dtype
            )

            high_conf = (conf > float(self.sfp_proxy_conf_thd)).to(dtype).unsqueeze(1)
            proxy_source_mask = high_conf * keep_mask

            proxy_sum = F.conv2d(
                output * proxy_source_mask,
                proxy_kernel.expand(C, 1, proxy_kernel_size, proxy_kernel_size),
                padding=proxy_padding,
                groups=C
            )
            proxy_count = F.conv2d(
                proxy_source_mask,
                proxy_kernel,
                padding=proxy_padding
            )

            proxy_available = (proxy_count > 0).to(dtype)
            proxy_mean = proxy_sum / proxy_count.clamp_min(1.0)

            proxy_lambda = float(getattr(self, "sfp_proxy_lambda", 2.0))
            refined_target = neigh_mean + proxy_lambda * proxy_available * (
                proxy_mean - neigh_mean
            )

        output_clean = output * (1.0 - update_mask) + refined_target * update_mask
        beta = float(self.sfp_logit_beta)
        output_new = (1.0 - beta) * output + beta * output_clean

        if getattr(self, "sfp_debug_export", False):
            self.sfp_debug_maps["cpsfp_delta"] = (output_new - output).abs().mean(dim=1).detach().float().cpu()
            self.sfp_debug_maps["pred_before_cpsfp"] = output.argmax(dim=1).detach().cpu()
            self.sfp_debug_maps["pred_after_cpsfp"] = output_new.argmax(dim=1).detach().cpu()

        with torch.no_grad():
            num_outliers = outlier_mask.float().sum(dim=(1, 2))
            ratio = num_outliers / float(H * W)
            diff = (output_new - output).abs()
            proxy_used = (proxy_available * update_mask).sum(dim=(1, 2, 3))
            update_count = update_mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
            proxy_available_ratio = proxy_used / update_count

            for b in range(B):
                self.sfp_last_stats_batch.append({
                    "H": int(H),
                    "W": int(W),
                    "num_tokens": int(H * W),
                    "outliers": float(num_outliers[b].detach().cpu()),
                    "ratio": float(ratio[b].detach().cpu()),
                    "score_min": float(sfp_score[b].min().detach().cpu()),
                    "score_max": float(sfp_score[b].max().detach().cpu()),
                    "score_mean": float(sfp_score[b].mean().detach().cpu()),
                    "conf_min": float(conf[b].min().detach().cpu()),
                    "conf_max": float(conf[b].max().detach().cpu()),
                    "diff_mean": float(diff[b].mean().detach().cpu()),
                    "diff_max": float(diff[b].max().detach().cpu()),
                    "proxy_enable": int(getattr(self, "sfp_proxy_enable", False)),
                    "proxy_lambda": float(getattr(self, "sfp_proxy_lambda", 0.0)),
                    "proxy_conf_thd": float(getattr(self, "sfp_proxy_conf_thd", 0.0)),
                    "proxy_kernel": int(getattr(self, "sfp_proxy_kernel", 0)),
                    "proxy_available_ratio": float(proxy_available_ratio[b].detach().cpu()),

                    "margin_enable": int(getattr(self, "sfp_margin_enable", False)),
                    "margin_lambda": float(getattr(self, "sfp_margin_lambda", 0.0)),
                    "margin_hard_enable": int(getattr(self, "sfp_margin_hard_enable", False)),
                    "margin_thd": float(getattr(self, "sfp_margin_thd", 0.0)),
                    "margin_min": float(margin[b].min().detach().cpu()),
                    "margin_max": float(margin[b].max().detach().cpu()),
                    "margin_mean": float(margin[b].mean().detach().cpu()),
                    "selected_margin_mean": float(
                        margin[b][outlier_mask[b]].mean().detach().cpu()
                    ) if outlier_mask[b].any() else 0.0,

                    "fbls_enable": int(getattr(self, "sfp_fbls_enable", False)),
                })

        return output_new

    def sfp_domain_transform_logit_refine(self, output, image):
        """
        Selected-region domain-transform logit refinement.

        The domain-transform filter is computed on the current logits, but only
        Selected unreliable tokens are updated. High-confidence tokens are
        preserved because the update mask is exactly the cached unreliable mask.
        """
        if not getattr(self, "sfp_dtlr_enable", False):
            return output

        outlier_mask = getattr(self, "sfp_last_outlier_mask", None)
        if outlier_mask is None:
            return output

        B, C, H, W = output.shape
        device = output.device
        dtype = output.dtype

        outlier_mask = outlier_mask.to(device=device, dtype=dtype)

        if outlier_mask.shape[-2:] != (H, W):
            outlier_mask = F.interpolate(
                outlier_mask.unsqueeze(1),
                size=(H, W),
                mode="nearest"
            ).squeeze(1)

        selected = outlier_mask.unsqueeze(1)

        if selected.sum() < 1:
            return output

        # Keep parameters editable from __init__ without rebuilding the module.
        self.sfp_dtlr_filter.sigma_s = float(getattr(self, "sfp_dtlr_sigma_s", 30.0))
        self.sfp_dtlr_filter.sigma_r = float(getattr(self, "sfp_dtlr_sigma_r", 0.30))
        self.sfp_dtlr_filter.num_iterations = int(getattr(self, "sfp_dtlr_num_iter", 1))

        filtered = self.sfp_dtlr_filter(output, image)

        update_mask = selected
        reject_mask = torch.zeros((B, 1, H, W), device=device, dtype=dtype)

        if getattr(self, "sfp_dtlr_boundary_only", False):
            pred = output.argmax(dim=1, keepdim=True)

            boundary = torch.zeros_like(pred, dtype=dtype)
            boundary[:, :, 1:, :] += (pred[:, :, 1:, :] != pred[:, :, :-1, :]).to(dtype)
            boundary[:, :, :-1, :] += (pred[:, :, :-1, :] != pred[:, :, 1:, :]).to(dtype)
            boundary[:, :, :, 1:] += (pred[:, :, :, 1:] != pred[:, :, :, :-1]).to(dtype)
            boundary[:, :, :, :-1] += (pred[:, :, :, :-1] != pred[:, :, :, 1:]).to(dtype)

            boundary = (boundary > 0).to(dtype)
            boundary = F.max_pool2d(boundary, kernel_size=3, stride=1, padding=1)
            update_mask = update_mask * boundary

        # Structure-preserving protection
        if getattr(self, "sfp_dtlr_structure_protect_enable", False):
            scale = float(getattr(self, "sfp_conf_scale", 10.0))

            orig_prob = torch.softmax(output * scale, dim=1)
            filt_prob = torch.softmax(filtered * scale, dim=1)

            orig_conf, orig_cls = orig_prob.max(dim=1, keepdim=True)  # [B,1,H,W]
            filt_conf, filt_cls = filt_prob.max(dim=1, keepdim=True)

            structure_classes = getattr(self, "sfp_dtlr_structure_classes", (1, 8, 10, 15))
            structure_mask = torch.zeros_like(orig_cls, dtype=torch.bool)
            for cls_id in structure_classes:
                structure_mask = structure_mask | (orig_cls == int(cls_id))

            class_flip = filt_cls != orig_cls
            conf_gain = filt_conf - orig_conf
            gain_thd = float(getattr(self, "sfp_dtlr_structure_gain_thd", 0.03))

            reject = structure_mask & class_flip & (conf_gain < gain_thd)
            reject_mask = reject.to(dtype)
            update_mask = update_mask * (1.0 - reject_mask)

        if update_mask.sum() < 1:
            return output

        beta = float(getattr(self, "sfp_dtlr_beta", 0.15))
        delta = filtered - output

        if bool(getattr(self, "sfp_dtlr_class_beta_enable", False)):
            with torch.no_grad():
                pred_cls = output.argmax(dim=1, keepdim=True)
                cls_mask = torch.zeros_like(pred_cls, dtype=torch.bool)
                for cls_idx in getattr(self, "sfp_dtlr_class_beta_classes", (4, 8, 10)):
                    cls_mask = cls_mask | (pred_cls == int(cls_idx))

            beta_map = output.new_full((output.shape[0], 1, output.shape[2], output.shape[3]), beta)
            beta_map = torch.where(
                cls_mask,
                beta_map * float(getattr(self, "sfp_dtlr_class_beta_scale", 0.75)),
                beta_map,
            )
        else:
            beta_map = beta

        output_new = output + beta_map * update_mask * delta

        if getattr(self, "sfp_debug_export", False):
            self.sfp_debug_maps["dtlr_update_mask"] = update_mask.detach().float().cpu()
            self.sfp_debug_maps["dtlr_reject_mask"] = reject_mask.detach().float().cpu()
            self.sfp_debug_maps["dtlr_delta"] = (output_new - output).abs().mean(dim=1).detach().float().cpu()
            self.sfp_debug_maps["pred_before_dtlr"] = output.argmax(dim=1).detach().cpu()
            self.sfp_debug_maps["pred_after_dtlr"] = output_new.argmax(dim=1).detach().cpu()

        # Append DTLR protection statistics to existing diagnostic CSV rows.
        with torch.no_grad():
            selected_count = selected.sum(dim=(1, 2, 3)).clamp_min(1.0)
            updated_count = update_mask.sum(dim=(1, 2, 3))
            rejected_count = (reject_mask * selected).sum(dim=(1, 2, 3))
            dtlr_diff = (output_new - output).abs()

            for b in range(B):
                if b < len(self.sfp_last_stats_batch):
                    self.sfp_last_stats_batch[b].update({
                        "dtlr_enable": int(getattr(self, "sfp_dtlr_enable", False)),
                        "dtlr_beta": float(getattr(self, "sfp_dtlr_beta", 0.0)),
                        "dtlr_sigma_s": float(getattr(self, "sfp_dtlr_sigma_s", 0.0)),
                        "dtlr_sigma_r": float(getattr(self, "sfp_dtlr_sigma_r", 0.0)),
                        "dtlr_num_iter": int(getattr(self, "sfp_dtlr_num_iter", 1)),
                        "dtlr_boundary_only": int(getattr(self, "sfp_dtlr_boundary_only", False)),
                        "dtlr_structure_protect": int(getattr(self, "sfp_dtlr_structure_protect_enable", False)),
                        "dtlr_structure_gain_thd": float(getattr(self, "sfp_dtlr_structure_gain_thd", 0.0)),
                        "dtlr_class_beta_enable": int(getattr(self, "sfp_dtlr_class_beta_enable", False)),
                        "dtlr_class_beta_scale": float(getattr(self, "sfp_dtlr_class_beta_scale", 1.0)),
                        "dtlr_selected_count": float(selected_count[b].detach().cpu()),
                        "dtlr_updated_count": float(updated_count[b].detach().cpu()),
                        "dtlr_rejected_count": float(rejected_count[b].detach().cpu()),
                        "dtlr_rejected_ratio": float((rejected_count[b] / selected_count[b]).detach().cpu()),
                        "dtlr_diff_mean": float(dtlr_diff[b].mean().detach().cpu()),
                        "dtlr_diff_max": float(dtlr_diff[b].max().detach().cpu()),
                    })

        return output_new

    def sfp_fast_bilateral_logit_solver(self, output, image):
        if not getattr(self, "sfp_fbls_enable", False):
            return output

        outlier_mask = getattr(self, "sfp_last_outlier_mask", None)
        if outlier_mask is None:
            return output

        B, C, H, W = output.shape
        device = output.device
        dtype = output.dtype

        outlier_mask = outlier_mask.to(device=device, dtype=dtype)
        if outlier_mask.shape[-2:] != (H, W):
            outlier_mask = F.interpolate(
                outlier_mask.unsqueeze(1),
                size=(H, W),
                mode="nearest"
            ).squeeze(1)

        selected = outlier_mask.unsqueeze(1)
        if selected.sum() < 1:
            return output

        k = int(getattr(self, "sfp_fbls_kernel", 5))
        if k not in (3, 5, 7):
            raise ValueError(f"sfp_fbls_kernel must be 3, 5, or 7, got {k}.")
        pad = k // 2
        num_neighbors = k * k

        guide = F.interpolate(
            image.to(device=device, dtype=dtype),
            size=(H, W),
            mode="bilinear",
            align_corners=False
        )
        guide_min = guide.amin(dim=(2, 3), keepdim=True)
        guide_max = guide.amax(dim=(2, 3), keepdim=True)
        guide = (guide - guide_min) / (guide_max - guide_min).clamp_min(1e-6)

        prob = torch.softmax(output * float(self.sfp_conf_scale), dim=1)
        conf = prob.max(dim=1, keepdim=True)[0]

        source_mask = (conf > float(self.sfp_fbls_conf_thd)).to(dtype) * (1.0 - selected)
        if source_mask.sum() < 1:
            return output

        out_win = F.unfold(output, kernel_size=k, padding=pad)
        out_win = out_win.view(B, C, num_neighbors, H * W)

        src_win = F.unfold(source_mask, kernel_size=k, padding=pad)
        src_win = src_win.view(B, 1, num_neighbors, H * W)

        conf_win = F.unfold(conf, kernel_size=k, padding=pad)
        conf_win = conf_win.view(B, 1, num_neighbors, H * W)

        guide_win = F.unfold(guide, kernel_size=k, padding=pad)
        guide_win = guide_win.view(B, 3, num_neighbors, H * W)
        guide_center = guide.view(B, 3, 1, H * W)

        color_dist2 = ((guide_win - guide_center) ** 2).sum(dim=1, keepdim=True)
        sigma_color = float(getattr(self, "sfp_fbls_sigma_color", 0.15))
        color_weight = torch.exp(
            -color_dist2 / (2.0 * sigma_color * sigma_color + 1e-12)
        )

        coords = torch.arange(k, device=device, dtype=dtype) - pad
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        spatial_dist2 = (xx.reshape(-1) ** 2 + yy.reshape(-1) ** 2)
        sigma_spatial = float(getattr(self, "sfp_fbls_sigma_spatial", 2.0))
        spatial_weight = torch.exp(
            -spatial_dist2 / (2.0 * sigma_spatial * sigma_spatial + 1e-12)
        ).view(1, 1, num_neighbors, 1)

        weight = src_win * conf_win * color_weight * spatial_weight
        weight_sum = weight.sum(dim=2).clamp_min(float(self.sfp_fbls_eps))

        refined = (out_win * weight).sum(dim=2) / weight_sum
        refined = refined.view(B, C, H, W)

        available = (
            weight_sum.view(B, 1, H, W) > float(self.sfp_fbls_eps)
        ).to(dtype)
        update_mask = selected * available
        if update_mask.sum() < 1:
            return output

        output_clean = output * (1.0 - update_mask) + refined * update_mask
        beta = float(getattr(self, "sfp_fbls_beta", 0.20))
        output_new = (1.0 - beta) * output + beta * output_clean

        with torch.no_grad():
            changed = (output_new - output).abs()
            for stat in getattr(self, "sfp_last_stats_batch", []):
                stat["fbls_kernel"] = int(k)
                stat["fbls_beta"] = float(beta)
                stat["fbls_conf_thd"] = float(getattr(self, "sfp_fbls_conf_thd", 0.0))
                stat["fbls_sigma_color"] = float(sigma_color)
                stat["fbls_update_ratio"] = float(update_mask.mean().detach().cpu())
                stat["fbls_diff_mean"] = float(changed.mean().detach().cpu())

        return output_new

    def _sfp_encode_text_prompts(self, prompts, device):
        try:
            tokens = clip.tokenize(prompts, truncate=True).to(device)
        except TypeError:
            tokens = clip.tokenize(prompts).to(device)

        with torch.no_grad():
            # Use the transformer/embedding dtype for the text pathway.
            try:
                text_dtype = next(self.clip.transformer.parameters()).dtype
            except StopIteration:
                text_dtype = self.clip.token_embedding.weight.dtype

            x = self.clip.token_embedding(tokens).to(dtype=text_dtype)

            pos = self.clip.positional_embedding.to(device=device, dtype=text_dtype)
            x = x + pos

            x = x.permute(1, 0, 2)  # [NLD] -> [LND]
            x = self.clip.transformer(x)
            x = x.permute(1, 0, 2)  # [LND] -> [NLD]

            x = self.clip.ln_final(x).to(dtype=text_dtype)

            eot_idx = tokens.argmax(dim=-1)
            x = x[torch.arange(x.shape[0], device=device), eot_idx]

            proj = self.clip.text_projection.to(device=device)
            x = x.to(dtype=proj.dtype)
            text_features = x @ proj

            text_features = text_features.float()
            text_features = F.normalize(text_features, dim=-1)

        return text_features

    def _sfp_build_attribute_bank(self, device, dtype):
        if (
            self.sfp_attr_bank is not None
            and self.sfp_attr_bank_device == device
            and tuple(self.sfp_attr_bank_classes.detach().cpu().tolist()) == tuple(getattr(self, "sfp_attr_apply_classes", (4, 8, 10)))
        ):
            return self.sfp_attr_bank.to(device=device, dtype=dtype), self.sfp_attr_bank_classes.to(device=device)

        class_ids = tuple(int(c) for c in getattr(self, "sfp_attr_apply_classes", (4, 8, 10)))
        banks = []
        valid_class_ids = []

        for cls_id in class_ids:
            attrs = self.sfp_attr_default_attributes.get(cls_id, [])
            cls_name = self.sfp_attr_class_names.get(cls_id, str(cls_id))

            if len(attrs) == 0:
                continue

            prompts = [
                self.sfp_attr_prompt_template.format(cls_name, attr)
                for attr in attrs
            ]

            text_features = self._sfp_encode_text_prompts(prompts, device=device)
            banks.append(text_features)
            valid_class_ids.append(cls_id)

        if len(banks) == 0:
            return None, None

        # All default classes have the same number of attributes. Pad just in case.
        max_k = max(b.shape[0] for b in banks)
        dim = banks[0].shape[-1]
        padded = []
        for b in banks:
            if b.shape[0] < max_k:
                pad = b.new_zeros(max_k - b.shape[0], dim)
                b = torch.cat([b, pad], dim=0)
            padded.append(b)

        attr_bank = torch.stack(padded, dim=0).to(device=device, dtype=dtype)  # [G,K,D]
        attr_classes = torch.tensor(valid_class_ids, device=device, dtype=torch.long)

        self.sfp_attr_bank = attr_bank.detach()
        self.sfp_attr_bank_classes = attr_classes.detach()
        self.sfp_attr_bank_device = device

        return attr_bank, attr_classes

    def sfp_attribute_residual_refine(
        self,
        output,
        feat,
        output_before_dtlr=None,
        output_after_dtlr=None,
    ):

        if not getattr(self, "sfp_attr_enable", False):
            return output

        outlier_mask = getattr(self, "sfp_last_outlier_mask", None)
        if outlier_mask is None:
            return output

        B, C, H, W = output.shape
        device = output.device
        dtype = output.dtype

        if feat.shape[-2:] != (H, W):
            feat = F.interpolate(feat, size=(H, W), mode="bilinear", align_corners=False)

        feat = feat.to(device=device, dtype=dtype)
        feat = F.normalize(feat, dim=1)

        attr_bank, attr_classes = self._sfp_build_attribute_bank(device=device, dtype=dtype)
        if attr_bank is None or attr_classes is None:
            return output

        if attr_bank.shape[-1] != feat.shape[1]:
            # Feature dimension mismatch; keep original logits unchanged.
            return output

        outlier_mask = outlier_mask.to(device=device, dtype=dtype)
        if outlier_mask.shape[-2:] != (H, W):
            outlier_mask = F.interpolate(
                outlier_mask.unsqueeze(1),
                size=(H, W),
                mode="nearest"
            ).squeeze(1)

        class_ids = [int(x) for x in attr_classes.detach().cpu().tolist()]

        def _class_mask(cls_map):
            mask = torch.zeros_like(cls_map, dtype=torch.bool)
            for cls_idx in class_ids:
                mask = mask | (cls_map == int(cls_idx))
            return mask

        conflict_only = bool(getattr(self, "sfp_attr_conflict_only", False))
        conflict_mode = str(getattr(self, "sfp_attr_conflict_mode", "before")).lower()

        flip_mask = torch.ones((B, 1, H, W), device=device, dtype=torch.bool)

        if (
            conflict_only
            and output_before_dtlr is not None
            and output_after_dtlr is not None
        ):
            before_logits = output_before_dtlr
            after_logits = output_after_dtlr

            if before_logits.shape[-2:] != (H, W):
                before_logits = F.interpolate(
                    before_logits,
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False
                )
            if after_logits.shape[-2:] != (H, W):
                after_logits = F.interpolate(
                    after_logits,
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False
                )

            before_cls = before_logits.argmax(dim=1, keepdim=True)
            after_cls = after_logits.argmax(dim=1, keepdim=True)
            flip_mask = before_cls != after_cls

            if conflict_mode == "both":
                struct_mask = _class_mask(before_cls) | _class_mask(after_cls)
            elif conflict_mode == "after":
                struct_mask = _class_mask(after_cls)
            else:
                # Default: protect structure-sensitive predictions before DTLR.
                struct_mask = _class_mask(before_cls)
        else:
            pred_cls = output.argmax(dim=1, keepdim=True)
            struct_mask = _class_mask(pred_cls)

        update_mask = outlier_mask.unsqueeze(1) * struct_mask.to(dtype) * flip_mask.to(dtype)
        if float(update_mask.sum().detach().cpu()) <= 0.0:
            if getattr(self, "sfp_debug_export", False):
                self.sfp_debug_maps["attr_update_mask"] = update_mask.detach().float().cpu()
                self.sfp_debug_maps["pred_before_attr"] = output.argmax(dim=1).detach().cpu()
                self.sfp_debug_maps["pred_after_attr"] = output.argmax(dim=1).detach().cpu()
            return output

        # sim_attr: [B,G,K,H,W]
        sim_attr = torch.einsum(
            "bdhw,gkd->bgkhw",
            feat.float(),
            attr_bank.float()
        )

        topm = min(int(getattr(self, "sfp_attr_topm", 3)), sim_attr.shape[2])
        attr_score = sim_attr.topk(k=topm, dim=2).values.mean(dim=2)  # [B,G,H,W]

        attr_score = attr_score - attr_score.mean(dim=1, keepdim=True)

        attr_logits = output.new_zeros(output.shape)
        for gi, cls_idx in enumerate(class_ids):
            if 0 <= int(cls_idx) < C:
                attr_logits[:, int(cls_idx), :, :] = attr_score[:, gi, :, :].to(dtype)

        directed_enable = bool(getattr(self, "sfp_attr_directed_enable", False))
        if (
            directed_enable
            and conflict_only
            and output_before_dtlr is not None
            and output_after_dtlr is not None
        ):
            before_logits = output_before_dtlr
            after_logits = output_after_dtlr

            if before_logits.shape[-2:] != (H, W):
                before_logits = F.interpolate(
                    before_logits,
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False
                )
            if after_logits.shape[-2:] != (H, W):
                after_logits = F.interpolate(
                    after_logits,
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False
                )

            before_cls = before_logits.argmax(dim=1, keepdim=True)
            after_cls = after_logits.argmax(dim=1, keepdim=True)

            directed_channel_mask = output.new_zeros(output.shape)
            for cls_idx in class_ids:
                cls_idx = int(cls_idx)
                if 0 <= cls_idx < C:
                    involved = (before_cls == cls_idx) | (after_cls == cls_idx)
                    directed_channel_mask[:, cls_idx:cls_idx + 1, :, :] = involved.to(dtype)

            attr_logits = attr_logits * directed_channel_mask

            if bool(getattr(self, "sfp_attr_directed_positive_only", False)):
                attr_logits = attr_logits.clamp_min(0.0)

        pos_only = bool(getattr(self, "sfp_attr_positive_only", False))
        neg_only = bool(getattr(self, "sfp_attr_negative_only", False))
        if pos_only and neg_only:
            
            neg_only = False

        if pos_only:
            attr_logits = attr_logits.clamp_min(0.0)
        elif neg_only:
            attr_logits = attr_logits.clamp_max(0.0)

        if bool(getattr(self, "sfp_attr_class_eta_enable", False)):
            eta_map = output.new_zeros(output.shape)

            eta_chair = float(getattr(
                self,
                "sfp_attr_eta_chair",
                getattr(self, "sfp_attr_eta", 36.0),
            ))
            eta_table = float(getattr(
                self,
                "sfp_attr_eta_table",
                getattr(self, "sfp_attr_eta", 36.0),
            ))

            eta_map[:, 8:9, :, :] = eta_chair
            eta_map[:, 10:11, :, :] = eta_table

            output_new = output + update_mask * eta_map * attr_logits
        else:
            eta = float(getattr(self, "sfp_attr_eta", 36.0))
            output_new = output + eta * update_mask * attr_logits

        if getattr(self, "sfp_debug_export", False):
            attr_raw_delta = output_new - output
            self.sfp_debug_maps["attr_update_mask"] = update_mask.detach().float().cpu()
            self.sfp_debug_maps["attr_delta"] = attr_raw_delta.abs().mean(dim=1).detach().float().cpu()
            self.sfp_debug_maps["attr_logit_chair"] = attr_logits[:, 8:9, :, :].detach().float().cpu() if C > 8 else output.new_zeros((B, 1, H, W)).cpu()
            self.sfp_debug_maps["attr_logit_table"] = attr_logits[:, 10:11, :, :].detach().float().cpu() if C > 10 else output.new_zeros((B, 1, H, W)).cpu()
            self.sfp_debug_maps["attr_delta_chair"] = attr_raw_delta[:, 8:9, :, :].detach().float().cpu() if C > 8 else output.new_zeros((B, 1, H, W)).cpu()
            self.sfp_debug_maps["attr_delta_table"] = attr_raw_delta[:, 10:11, :, :].detach().float().cpu() if C > 10 else output.new_zeros((B, 1, H, W)).cpu()
            self.sfp_debug_maps["pred_before_attr"] = output.argmax(dim=1).detach().cpu()
            self.sfp_debug_maps["pred_after_attr"] = output_new.argmax(dim=1).detach().cpu()

        with torch.no_grad():
            attr_delta = (output_new - output).abs()
            attr_update_ratio = update_mask.mean(dim=(1, 2, 3))
            attr_flip_ratio = flip_mask.to(dtype).mean(dim=(1, 2, 3))
            for b in range(B):
                if b < len(self.sfp_last_stats_batch):
                    self.sfp_last_stats_batch[b].update({
                        "attr_enable": int(getattr(self, "sfp_attr_enable", False)),
                        "attr_eta": float(getattr(self, "sfp_attr_eta", 0.0)),
                        "attr_class_eta_enable": int(getattr(self, "sfp_attr_class_eta_enable", False)),
                        "attr_eta_chair": float(getattr(self, "sfp_attr_eta_chair", 0.0)),
                        "attr_eta_table": float(getattr(self, "sfp_attr_eta_table", 0.0)),
                        "attr_apply_classes": "-".join(str(int(c)) for c in getattr(self, "sfp_attr_apply_classes", ())),
                        "attr_topm": int(getattr(self, "sfp_attr_topm", 3)),
                        "attr_conflict_only": int(getattr(self, "sfp_attr_conflict_only", False)),
                        "attr_conflict_mode": str(getattr(self, "sfp_attr_conflict_mode", "before")),
                        "attr_directed_enable": int(getattr(self, "sfp_attr_directed_enable", False)),
                        "attr_directed_positive_only": int(getattr(self, "sfp_attr_directed_positive_only", False)),
                        "attr_positive_only": int(getattr(self, "sfp_attr_positive_only", False)),
                        "attr_negative_only": int(getattr(self, "sfp_attr_negative_only", False)),
                        "attr_update_ratio": float(attr_update_ratio[b].detach().cpu()),
                        "attr_flip_ratio": float(attr_flip_ratio[b].detach().cpu()),
                        "attr_delta_mean": float(attr_delta[b].mean().detach().cpu()),
                        "attr_delta_max": float(attr_delta[b].max().detach().cpu()),
                    })

        return output_new


    def forward(self, image, gt_cls, zeroshot_weights, cls_name_token, training=False, img_metas=None,
                return_feat=False):
        if getattr(self, "sfp_debug_export", False):
            self.sfp_debug_maps = {}
        else:
            self.sfp_debug_maps = {}

        cnum = zeroshot_weights.shape[0]
        device = self.device
        gt_cls_text_embeddings = zeroshot_weights.to(device)

        batch_size = image.shape[0]
        image = image.to(device)
        v, shape, z_global, k, positional_embedding, sfp_score = self.vit(
            image, train=False, img_metas=img_metas
        )

        positional_embedding = positional_embedding.reshape(1, shape[0], shape[1], -1).permute(0, 3, 1, 2)

        feat = self.proj(v)
        feat = feat / feat.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()

        # ori
        output_q = F.conv2d(feat, gt_cls_text_embeddings[:, :, None, None]).permute(0, 2, 3, 1).reshape(batch_size, -1,
                                                                                                        cnum)

        # reference prompt
        prompt = self.text_encoder(cls_name_token)
        prompt = prompt / prompt.norm()

        pe = self.pe_proj(positional_embedding).permute(0, 2, 3, 1).reshape(1, shape[0] * shape[1], -1)
        bias_logits = pe @ prompt.t()
        output = torch.sub(output_q, bias_logits).permute(0, 2, 1).reshape(batch_size, -1, shape[0], shape[1])

        try:
            self._debug_tensors = {}

            self._debug_tensors["output_q"] = output_q.detach()
            self._debug_tensors["bias_logits"] = bias_logits.detach()
            self._debug_tensors["output_after_bias_logits"] = output.detach()
        except Exception:
            pass

        feature = torch.cat((feat, output), dim=1)
        feature = self.decoder_conv2(feature)
        feature = self.decoder_norm2(feature)
        output = feature
        output = self.sfp_logit_purify(output, sfp_score)

        output_before_dtlr = output

        if (not training) and getattr(self, "sfp_dtlr_enable", False):
            output = self.sfp_domain_transform_logit_refine(output, image)

        output_after_dtlr = output

        if getattr(self, "sfp_debug_export", False):
            self.sfp_debug_maps["output_before_dtlr_logits"] = output_before_dtlr.detach().float().cpu()
            self.sfp_debug_maps["output_after_dtlr_logits"] = output_after_dtlr.detach().float().cpu()

        if (not training) and getattr(self, "sfp_attr_enable", False):
            output = self.sfp_attribute_residual_refine(
                output,
                feat,
                output_before_dtlr=output_before_dtlr,
                output_after_dtlr=output_after_dtlr,
            )

        if getattr(self, "sfp_debug_export", False):
            self.sfp_debug_maps["output_after_attr_logits"] = output.detach().float().cpu()
            self.sfp_debug_maps["pred_final_lowres"] = output.argmax(dim=1).detach().cpu()

        if (not training) and getattr(self, "sfp_fbls_enable", False):
            output = self.sfp_fast_bilateral_logit_solver(output, image)

        if return_feat:
            return output[0], feat[0], shape

        if training:
            # Gumbel-Softmax pseudo masks.
            output_scale = torch.mul(
                output.reshape(batch_size, cnum, -1).permute(0, 2, 1),
                100
            )
            output_gumbel = F.gumbel_softmax(
                output_scale,
                tau=1,
                hard=True,
                dim=2
            ).reshape(batch_size, shape[0], shape[1], -1)

            loss = output.new_tensor(0.0)
            valid_batch_count = 0

            for j in range(batch_size):
                masked_image_features = []
                valid_labels = []

                if len(gt_cls[j]) == 0:
                    continue

                for i in gt_cls[j]:
                    cls_id = int(i)
                    mask = output_gumbel[j, :, :, cls_id].unsqueeze(dim=0)
                    if mask.sum() < 1.0:
                        continue

                    masked_image_feature = torch.mul(feat[j].unsqueeze(dim=0), mask)
                    feature_pool = nn.AdaptiveAvgPool2d((1, 1))(
                        masked_image_feature
                    ).reshape(1, 512)

                    masked_image_features.append(feature_pool)
                    valid_labels.append(cls_id)

                if len(masked_image_features) == 0:
                    continue

                masked_image_features = torch.stack(
                    masked_image_features,
                    dim=0
                ).squeeze(dim=1)

                similarity_img = logit_scale * masked_image_features @ gt_cls_text_embeddings.t()
                labels = torch.tensor(valid_labels, dtype=torch.long, device=device)

                if getattr(self, "use_focal_loss", False):
                    loss += focal_loss_multiclass(
                        similarity_img,
                        labels,
                        gamma=float(self.focal_gamma),
                        alpha=self.focal_alpha,
                        reduction="mean",
                    )
                else:
                    loss += F.cross_entropy(similarity_img, labels)

                valid_batch_count += 1

            if valid_batch_count == 0:
                final_loss = output.sum() * 0.0
            else:
                final_loss = loss / valid_batch_count

            return output, final_loss


        return output

    def _initialize_weights(self, clip_model):
        self.proj.weight = nn.Parameter(clip_model.visual.proj[:, :, None, None].permute(1, 0, 2, 3).to(torch.float32),
                                        requires_grad=False)

