import torch
import torch.nn.functional as F
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel


class Qwen3VLVisionModelForNavitPOINTS(Qwen3VLVisionModel):
    """Rewrite the forward function of Qwen3VLVisionModel to
    adapt to POINTS.

    Do no apply patch merging to the hidden features output by the transformer.
    """
    def __init__(self, config) -> None:
        super().__init__(config)

    def forward(self, hidden_states: torch.Tensor,
                grid_thw: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states)

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2],
                                             grid_thw[:, 0]).cumsum(
                                                 dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for layer_id, blk in enumerate(self.blocks):
            hidden_states = blk(hidden_states,
                                cu_seqlens=cu_seqlens,
                                position_embeddings=position_embeddings)
        return hidden_states
