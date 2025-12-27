import torch
import pytorch_lightning as pl
from torchvision.transforms import v2, functional
from decord import VideoReader, cpu, bridge
import io
import numpy as np
import uuid
import webdataset as wds
from torch.utils.data import default_collate
from einops import rearrange
from torch.utils.data import DataLoader
import glob
import random
import math

# def custom_collate(batch):
#     # batch is a list of dicts, want list under a single dict header?
#     chunks = [item['video'] for item in batch]
#     keys = [item['__key__'] for item in batch]
#     fps = [item['fps'] for item in batch]
#     return {'video': chunks, 'fps': fps, '__key__': keys}

def _video_process(data, config=None, eval=False):
    cd = config.dataset
    cm = config.tokenizer.model
    trg_grid = cm.in_grid # THW
    trg_fps = cd.fps
    max_eval = config.training.eval.num_eval

    patch_size = cm.patch_size # eg. [4, 8, 8]
    assert all([dim % ps == 0 for dim, ps in zip(trg_grid, patch_size)]), "dimensions in grid must be evenly divisible by their respective patch sizes"

    seen = 0
    for sample in data:
        for video_key in sample.keys():
            if video_key == 'mp4' or video_key.endswith('.mp4'): # allow paths in mp4 key
                try:
                    bridge.set_bridge('torch')
                    if sample['__key__']:
                        out_key = sample['__key__']
                    else:
                        out_key = str(uuid.uuid4())

                    with io.BytesIO(sample[video_key]) as video_bytes:
                        vr = VideoReader(video_bytes, ctx=cpu(0), num_threads=0) # auto threading
                        in_fps = int(vr.get_avg_fps())
                        in_grid = [len(vr)] + list(vr[0].shape) # THW

                        if all([x >= y for x, y in zip(in_grid, trg_grid)]) and in_fps >= trg_fps:
                            start_idx = 0
                            end_idx = 0

                            while True:
                                end_idx = start_idx + int(trg_grid[0] * (in_fps / trg_fps))

                                if in_grid[0] < end_idx: # end condition
                                    break

                                chunk_indices = np.linspace(start_idx, end_idx - 1, trg_grid[0], dtype=int).tolist()
                                chunk = torch.Tensor(vr.get_batch(chunk_indices))
                                
                                if eval:
                                    transform = v2.Compose([
                                        v2.Resize(size=max(trg_grid[1:]), interpolation=functional.InterpolationMode.BICUBIC, antialias=True),
                                        v2.CenterCrop(size=trg_grid[1:]),
                                    ])
                                else:
                                    transform = v2.Compose([
                                        v2.RandomResizedCrop(size=trg_grid[1:], interpolation=functional.InterpolationMode.BICUBIC, antialias=True),
                                        v2.RandomHorizontalFlip(p=0.5),                    
                                    ])

                                # need separate resolutions for transforms.
                                chunk = chunk.permute(0, 3, 1, 2)
                                chunk = transform(chunk)
                                chunk = chunk.permute(1, 0, 2, 3)

                                chunk = chunk.to(torch.float32) / 255
                                chunk = (chunk * 2) - 1 # [-1, 1]
                                ###

                                yield {'video': chunk, '__key__': f'{out_key}_{start_idx}-{end_idx}'} 
                                start_idx = end_idx + 1 # setup for next chunk

                                seen += 1
                                if eval and seen >= max_eval:
                                    return                                    

                except Exception as error:
                    print(f'Decode fail: {error}')


video_process = wds.filters.pipelinefilter(_video_process)

class WebdatasetVideoDataModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        cd = config.dataset
        train_shard_path=cd.train_dataset
        eval_shard_path=cd.eval_dataset

        self.num_workers = cd.workers
        self.pin_memory = cd.pin_memory
        self.batch_size = config.training.main.batch_size

        train_pipeline = [
            wds.ResampledShards(train_shard_path),
            wds.split_by_worker, # no overlapping entries between workers
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(8, handler=wds.warn_and_continue),
            video_process(config, eval=False),
            wds.shuffle(64, handler=wds.warn_and_continue),
            wds.batched(self.batch_size, partial=False, collation_fn=default_collate),
        ]

        eval_pipeline = [
            wds.SimpleShardList(eval_shard_path),
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            video_process(config, eval=True),
            wds.batched(self.batch_size, partial=False, collation_fn=default_collate),
        ]
        
        self.train_dataset = wds.DataPipeline(*train_pipeline)
        self.eval_dataset = wds.DataPipeline(*eval_pipeline)

    
    def train_dataloader(self):
        return wds.WebLoader(self.train_dataset, batch_size=None, pin_memory=self.pin_memory, num_workers=self.num_workers, persistent_workers=True)
    
    def eval_dataloader(self):
        return wds.WebLoader(self.eval_dataset, batch_size=None, pin_memory=self.pin_memory, num_workers=1, persistent_workers=True)