from dataclasses import dataclass
from torch import Tensor
from typing import Dict, Optional, List, Tuple

import io
import os
import torch

from ..rig_package.info.asset import Asset
from ..model.tokenrig import TokenRig

PORT = 59875
SERVER = f"http://localhost:{PORT}"
TMP_CKPT_DIR = "./tmp_ckpt"

BPY_PORT = 59876
BPY_SERVER = f"http://localhost:{BPY_PORT}"

@dataclass
class TensorPacket:
    """make sure stays on cpu"""
    validate: bool=False
    know_skeleton: bool=False
    learned_mesh_cond: Optional[Tensor]=None
    cond_latents: Optional[Tensor]=None
    mesh_cond: Optional[Tensor]=None
    vertices: Optional[Tensor]=None
    assets: Optional[List[Asset]]=None
    output_ids: Optional[Tensor]=None
    start_embed_list: Optional[List[Tensor]]=None
    start_tokens_list: Optional[List[List[int]]]=None

    def to_device(self, device):
        if self.learned_mesh_cond is not None:
            self.learned_mesh_cond = self.learned_mesh_cond.to(device)
        if self.cond_latents is not None:
            self.cond_latents = self.cond_latents.to(device)
        if self.mesh_cond is not None:
            self.mesh_cond = self.mesh_cond.to(device)
        if self.vertices is not None:
            self.vertices = self.vertices.to(device)
        if self.output_ids is not None:
            self.output_ids = self.output_ids.to(device)
        if self.start_embed_list is not None:
            self.start_embed_list = [x.to(device) for x in self.start_embed_list]

    @property
    def B(self):
        assert self.learned_mesh_cond is not None
        return self.learned_mesh_cond.shape[0]

    def to_bytes(self):
        return object_to_bytes(self)

    @classmethod
    def from_bytes(cls, bytes) -> 'TensorPacket':
        return bytes_to_object(bytes)


def object_to_bytes(t):
    buffer = io.BytesIO()
    torch.save(t, buffer)
    return buffer.getvalue()

def bytes_to_object(b, map_location=None):
    return torch.load(io.BytesIO(b), weights_only=False, map_location=map_location)

def default_device() -> str:
    # ROCm builds of PyTorch (AMD GPUs) also report themselves as "cuda"
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def default_precision(device: str) -> str:
    # NVIDIA: bf16 weights as trained. AMD/ROCm and CPU: fp32 weights + bf16 autocast ("mixed"); on CPU the
    # bf16 weights fail (the cuda autocast decorators are inactive) and on ROCm they give noticeably worse skins.
    if device == 'cuda' and torch.version.hip is None:
        return 'bf16'
    return 'mixed'

def get_model(
    ckpt_path: str,
    hf_path: Optional[str]=None,
    device: Optional[str]=None,
    precision: Optional[str]=None,
) -> TokenRig:
    device = device or default_device()
    precision = precision or default_precision(device)
    model = TokenRig.load_from_system_checkpoint(checkpoint_path=ckpt_path)
    if hf_path is not None:
        from transformers import AutoModel
        a = AutoModel.from_pretrained(
            hf_path,
            local_files_only=True,
            _attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )
        model.transformer.model.load_state_dict(a.state_dict())

    if precision == 'mixed':
        model = model.float()
    model = model.to(device)
    return model
