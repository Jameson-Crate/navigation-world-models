from typing import Dict

from pydantic import BaseModel


class CDiTModelConfig(BaseModel):
    # Block configuration
    num_blocks: int = 4
    block_config: Dict[str, int] = {}


class CDiTBlockConfig(BaseModel):
    # Encoding Dimensions
    action_dim: int = 176
    time_enc_dim: int = 528
    denoise_enc_dim: int = 528
    rope_dim: int = 32

    # Norm Dimensions

    # FFN Dimensions
