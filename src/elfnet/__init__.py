"""ELFNet: symmetry-aware SAD-to-ELF prediction."""

from .checkpoints import DEFAULT_CHECKPOINT, resolve_checkpoint

__all__ = [
    "DEFAULT_CHECKPOINT",
    "Sad2ElfLitModule",
    "UNet3DPeriodic",
    "resolve_checkpoint",
]

__version__ = "0.1.0"


def __getattr__(name: str):
    if name in {"Sad2ElfLitModule", "UNet3DPeriodic"}:
        from .model import Sad2ElfLitModule, UNet3DPeriodic

        return {
            "Sad2ElfLitModule": Sad2ElfLitModule,
            "UNet3DPeriodic": UNet3DPeriodic,
        }[name]
    raise AttributeError(name)
