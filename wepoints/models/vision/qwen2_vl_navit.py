import torch
import torch.nn.functional as F
from transformers.models.qwen2_vl.modeling_qwen2_vl import \
    Qwen2VisionTransformerPretrainedModel  # noqa


class Qwen2VisionTransformerForNavitPOINTS(
        Qwen2VisionTransformerPretrainedModel):  # noqa
    """Rewrite the forward function of Qwen2VisionTransformerPretrainedModel to
    adapt to POINTS.  # noqa.

    Do no apply patch merging to the hidden features output by the transformer.
    """
    def __init__(self, config) -> None:
        super().__init__(config)

    def forward(self, hidden_states: torch.Tensor,
                grid_thw: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2],
                                             grid_thw[:, 0]).cumsum(
                                                 dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        for blk in self.blocks:
            hidden_states = blk(hidden_states,
                                cu_seqlens=cu_seqlens,
                                rotary_pos_emb=rotary_pos_emb)

        return hidden_states
