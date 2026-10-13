"""Run after building, inside an Arrhenius GPU allocation with --nv."""
import json
import platform
import sys
from importlib.metadata import version

import torch

expected = {
    "torch": "2.1.2", "numpy": "1.25.1", "scipy": "1.11.1",
    "pandas": "2.0.3", "scikit-learn": "1.4.2", "einops": "0.7.0",
    "matplotlib": "3.7.2", "tqdm": "4.66.1",
}
actual = {name: version(name) for name in expected}
assert actual == expected, (actual, expected)
assert platform.machine() == "aarch64", platform.machine()
assert sys.version_info[:3] == (3, 11, 3), sys.version
assert torch.version.cuda == "12.1", torch.version.cuda
assert torch.cuda.is_available(), "CUDA unavailable: run in a GPU allocation with --nv"
assert torch.backends.cudnn.is_available(), "cuDNN unavailable"
assert torch.backends.cudnn.version() == 8902, torch.backends.cudnn.version()

# Exercise convolution, autograd, an optimizer, and FFT on the GPU.
model = torch.nn.Conv1d(4, 8, 3).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
x = torch.randn(8, 4, 64, device="cuda")
loss = model(x).square().mean()
loss.backward()
optimizer.step()
assert torch.isfinite(loss).item()
assert all(p.grad is not None and torch.isfinite(p.grad).all().item()
           for p in model.parameters())
spectrum = torch.fft.rfft(x)
assert torch.isfinite(spectrum).all().item()
torch.cuda.synchronize()

print(json.dumps({
    "packages": actual,
    "python": platform.python_version(),
    "architecture": platform.machine(),
    "cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "nccl": torch.cuda.nccl.version(),
    "gpu": torch.cuda.get_device_name(0),
    "gpu_capability": torch.cuda.get_device_capability(0),
    "loss": loss.item(),
    "status": "GPU smoke check passed",
}, indent=2))
