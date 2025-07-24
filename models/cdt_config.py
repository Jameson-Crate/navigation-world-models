from pydantic import BaseModel


class CDTModelConfig(BaseModel):
    # Encoding Dimensions
    x_enc_dim: int = 176
    y_enc_dim: int = 176
    yaw_encd_dim: int = 176
    time_enc_dim: int = 528
    denoise_enc_dim: int = 528
    rope_dim: int = 32

    # Norm Dimensions

    # FFN Dimensions
