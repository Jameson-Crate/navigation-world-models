from pydantic import BaseModel


class CDTModelConfig(BaseModel):
    input_dim: int = 128
    hidden_dim: int = 256
    output_dim: int = 10
    dropout: float = 0.3
