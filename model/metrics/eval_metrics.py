import torch
import torch.nn as nn
import torch.nn.functional as F

from model.metrics.fvd import FVDCalculator
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics import MetricCollection
from torchvision.transforms import v2
from einops import rearrange

class EvalMetrics(nn.Module):
    def __init__(self, config, eval_prefix='eval'):
        super().__init__()
        self.eval_prefix = eval_prefix
        self.eval_metrics = MetricCollection(
            {
                "psnr": PeakSignalNoiseRatio(data_range=2), # -1, 1
                "ssim": StructuralSimilarityIndexMeasure(data_range=2),
            },
            prefix=f"{eval_prefix}/",
        )

        self.fvd = None
        if config.training.eval.log_fvd:
            self.fvd = FVDCalculator().eval()

    def update(self, recon, target):
        recon = recon.clamp(-1, 1)
        self.eval_metrics.update(
            rearrange(recon, 'b c t h w -> (b t) c h w'),
            rearrange(target, 'b c t h w -> (b t) c h w'),
        )

        if self.fvd is not None:
            self.fvd.update(recon, target)

    def compute(self):
        out_dict = self.eval_metrics.compute()
        if self.fvd is not None:
            out_dict[f"{self.eval_prefix}/fvd"] = self.fvd.gather()
        return out_dict
    
    def reset(self):
        self.eval_metrics.reset()
        if self.fvd is not None:
            self.fvd.reset()