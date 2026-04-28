"""ELFNet: full-grid SAD-to-ELF prediction."""

from .checkpoints import DEFAULT_CHECKPOINT, resolve_checkpoint

__all__ = [
    "DEFAULT_CHECKPOINT",
    "ELFPredictor",
    "ResidualUNet3D",
    "Sad2ElfLitModule",
    "UNet3DPeriodic",
    "resolve_checkpoint",
]

__version__ = "0.3.0"


def __getattr__(name: str):
    if name in {"ELFPredictor", "ResidualUNet3D", "Sad2ElfLitModule", "UNet3DPeriodic"}:
        from .model import ELFPredictor, ResidualUNet3D, Sad2ElfLitModule, UNet3DPeriodic

        return {
            "ELFPredictor": ELFPredictor,
            "ResidualUNet3D": ResidualUNet3D,
            "Sad2ElfLitModule": Sad2ElfLitModule,
            "UNet3DPeriodic": UNet3DPeriodic,
        }[name]
    raise AttributeError(name)
