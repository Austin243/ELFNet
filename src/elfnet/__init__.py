"""ELFNet: full-grid SAD-to-ELF prediction."""

from .checkpoints import DEFAULT_CHECKPOINT, resolve_checkpoint

__all__ = [
    "DEFAULT_CHECKPOINT",
    "ELFPredictor",
    "FlatResNet3D",
    "FlatResNet3DPeriodic",
    "ResidualUNet3D",
    "Sad2ElfLitModule",
    "UNet3DPeriodic",
    "resolve_checkpoint",
]

__version__ = "0.3.1"


def __getattr__(name: str):
    if name in {
        "ELFPredictor",
        "FlatResNet3D",
        "FlatResNet3DPeriodic",
        "ResidualUNet3D",
        "Sad2ElfLitModule",
        "UNet3DPeriodic",
    }:
        from .model import (
            ELFPredictor,
            FlatResNet3D,
            FlatResNet3DPeriodic,
            ResidualUNet3D,
            Sad2ElfLitModule,
            UNet3DPeriodic,
        )

        return {
            "ELFPredictor": ELFPredictor,
            "FlatResNet3D": FlatResNet3D,
            "FlatResNet3DPeriodic": FlatResNet3DPeriodic,
            "ResidualUNet3D": ResidualUNet3D,
            "Sad2ElfLitModule": Sad2ElfLitModule,
            "UNet3DPeriodic": UNet3DPeriodic,
        }[name]
    raise AttributeError(name)
