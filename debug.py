import hydra
import torch

from models.cdt_config import CDTModelConfig
from models.cdt_model import ConditionalDiffusionTransformer

device = "cuda"


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: CDTModelConfig) -> None:
    model_cfg = CDTModelConfig(**cfg.model)
    model = ConditionalDiffusionTransformer(model_cfg)
    model = model.to(device)

    s_t = torch.randn((1, 4096, 4), device=device)
    k = torch.randn((1, 1), device=device)
    t = torch.randn((1, 1), device=device)
    s_prev = torch.randn((1, 4096 * 4, 4), device=device)

    model(s_t, k, t, None, s_prev)


if __name__ == "__main__":
    main()
