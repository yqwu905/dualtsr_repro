import math
import torch
import torch.nn.functional as F
from torch import nn
from diffusers.utils.import_utils import is_torch_npu_available

if is_torch_npu_available():
    import torch_npu


class Bmm(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, attn_weights, value_states):
        ### attn_weights: b, num_head,    seqlen, head_dim
        ### value_states: b, num_kv_head, seqlen, head_dim
        output = torch.matmul(attn_weights, value_states)
        return output


class ScaledDotProductAttnAigc(torch.nn.Module):
    def __init__(self, dim_head, heads):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        self.bmm_key = Bmm()
        self.bmm_value = Bmm()

    def forward(self, query, key, value, attn_mask=None, dropout_p=0.0,
            is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)

        if query.ndim > attn_bias.ndim:
            for add_dim in range(query.ndim - attn_bias.ndim):
                attn_bias = attn_bias.unsqueeze(dim=0)
            attn_bias = attn_bias.repeat(query.shape[0], query.shape[1], 1, 1)

        if is_causal and attn_mask is None:
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias = attn_mask + attn_bias

        if enable_gqa:
            key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
            value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

        # attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight = self.bmm_key(query, key.transpose(-2, -1)) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=self.training)
        # out = attn_weight @ value
        out = self.bmm_value(attn_weight, value)
        return out


class SparseProcessAttnAigc(ScaledDotProductAttnAigc):

    def __init__(self, dim_head=None, heads=None):
        super().__init__(dim_head, heads)

    def split_and_squeeze(self, ori_tensor, block_lenth):
        B, n, L, C = ori_tensor.shape
        new_L = L // block_lenth
        assert L % block_lenth == 0, f'B, N, L, C: {B}, {n}, {L}, {C}'
        mean_tensor = ori_tensor.view(B, n, new_L, block_lenth, C).mean(dim=3)

        return mean_tensor

    def scale_dot(self, query, key, value, attn_mask=None, dropout_p=0.0,
            is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:

        return super().forward(
            query, key, value, attn_mask, \
            dropout_p, is_causal, scale, enable_gqa
        )


    def forward(self, query, key, value, img_len, batch_size, sparse_ratio): #### qkv after retory
        # add by yulei.
        ''' args '''
        block_lenth = 64  # num_block = 4096 // 32 = 128 (1024 * 1024)  256 // 32 = 8 (256, 256)
        H, W = img_len
        new_img_len = H * W

        query_image = query[:, :, -new_img_len:, :]
        key_image = key[:, :, -new_img_len:, :]
        N_text = query.shape[2] - query_image.shape[2]

        num_block = (query_image.shape[2]) // block_lenth
        topK = int(num_block * sparse_ratio)  # k=0.75 means mask ratio is 75%, equal to kv_compress_ratio=2
        topK = num_block - 8 # 通路上最大计算只能取8，因此反算topK有此限制

        query_image_block = self.split_and_squeeze(query_image, block_lenth=block_lenth)
        key_image_block = self.split_and_squeeze(key_image, block_lenth=block_lenth)

        similarity_matrix = torch.matmul(query_image_block, key_image_block.transpose(-1, -2))
        # print(similarity_matrix)

        # print(f"similarity_matrix: {similarity_matrix.shape}")

        _, top_index = (-similarity_matrix).topk(k=topK, dim=-1)  
        # print(f"top_index: {top_index.shape}")
        mask = torch.zeros(batch_size, query.shape[1], num_block, num_block).to(query_image.device)  
        # print(f"mask: {mask.shape}")
        mask = mask.scatter_(3, top_index, float('-inf')) 
        # mask = mask.repeat_interleave(block_lenth, dim=2).repeat_interleave(block_lenth, dim=3)
        mask = mask.repeat_interleave(block_lenth, dim=2)
        mask_shape = mask.shape
        mask = mask.unsqueeze(4).expand(mask_shape[0], mask_shape[1], mask_shape[2], mask_shape[3], block_lenth)
        mask = mask.reshape(mask_shape[0], mask_shape[1], mask_shape[2], mask_shape[3] * block_lenth)

        # if encoder_hidden_states is not None:
        mask = torch.nn.functional.pad(mask, (N_text, 0, N_text, 0))
        mask = mask.to(torch.bool)
        # print(f"final mask: {mask.shape}")

        # add by dkp.
        ori_hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=mask.logical_not(), dropout_p=0.0,
                                                           is_causal=False)  # add mask input,
        return ori_hidden_states