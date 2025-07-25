import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from models.cdit_config import CDiTModelConfig
from models.cdit_model import CDiTModel

device = "cuda"


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg_dict = OmegaConf.to_container(cfg.model, resolve=True)
    model_cfg = CDiTModelConfig(**cfg_dict)
    model = CDiTModel(model_cfg)
    model = model.to(device)

    # Example Tensors
    B = 6
    s_t = torch.randn((B, 4096, 4), device=device)
    k = torch.randn((B, 1), device=device)
    t = torch.randn((B, 1), device=device)
    s_prev = torch.randn((B, 4096 * 4, 4), device=device)

    prediction = model(s_t, k, t, None, s_prev)
    print(prediction.shape)


if __name__ == "__main__":
    main()
