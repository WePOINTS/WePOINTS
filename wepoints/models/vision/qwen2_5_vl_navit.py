import torch
import torch.nn.functional as F
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import \
    Qwen2_5_VisionTransformerPretrainedModel


class Qwen2_5VisionTransformerForNavitPOINTS(
        Qwen2_5_VisionTransformerPretrainedModel):  # noqa
    """Rewrite the forward function of Qwen2VisionTransformerPretrainedModel
        to adapt to POINTS.

    To use this version, you need to install the main branch of
    huggingface/transformers
    """
    def __init__(self, config) -> None:
        super().__init__(config)

    def forward(self, hidden_states: torch.Tensor,
                grid_thw: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2],
                                             grid_thw[:, 0]).cumsum(
                                                 dim=0,
                                                 dtype=torch.int32,
                                             )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens
            hidden_states = blk(hidden_states,
                                cu_seqlens=cu_seqlens_now,
                                rotary_pos_emb=rotary_pos_emb)
        return hidden_states, window_index
