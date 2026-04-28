#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""POSCAR-to-ELFCAR inference for the full-grid ELFNet checkpoint.

The bundled checkpoint is the verified ChiNet ``epoch1000.ckpt`` model. The
pipeline mirrors the old ChiNet inference path that produced the current
``ChiNet/outputs`` files:

1. parse ``POSCAR_*`` files,
2. build a full superposed atomic density (SAD) grid,
3. run one full-grid forward pass,
4. write ``ELFCAR_<id>.vasp``.

No symmetry operations and no patch reconstruction are used by this model.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from importlib import resources
from math import ceil, pi, sqrt
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from .checkpoints import resolve_checkpoint
from .model import ELFPredictor


EV2HA = 1 / 27.211386
ANG2BOHR = 1 / 0.529177
BOHR_TO_ANG = 0.5291772108
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG
BOHR3_TO_ANG3 = BOHR_TO_ANG**3
RHO_THRESH = 1e-4
RCUT_MAX = 20.0

DEFAULT_VALENCE: Dict[str, float] = {
    "H": 1,
    "He": 2,
    "Li": 1,
    "Be": 2,
    "B": 3,
    "C": 4,
    "N": 5,
    "O": 6,
    "F": 7,
    "Ne": 8,
    "Na": 1,
    "Mg": 2,
    "Al": 3,
    "Si": 4,
    "P": 5,
    "S": 6,
    "Cl": 7,
    "Ar": 8,
    "K": 1,
    "Ca": 2,
    "Sc": 11,
    "Ti": 12,
    "V": 13,
    "Cr": 14,
    "Mn": 15,
    "Fe": 16,
    "Co": 17,
    "Ni": 18,
    "Cu": 11,
    "Zn": 12,
    "Ga": 13,
    "Ge": 14,
    "As": 15,
    "Se": 16,
    "Br": 17,
    "Kr": 18,
    "Rb": 9,
    "Sr": 10,
    "Y": 11,
    "Zr": 12,
    "Nb": 13,
    "Mo": 14,
    "Tc": 15,
    "Ru": 16,
    "Rh": 17,
    "Pd": 18,
    "Ag": 11,
    "Cd": 12,
    "In": 13,
    "Sn": 14,
    "Sb": 15,
    "Te": 16,
    "I": 17,
    "Xe": 18,
    "Cs": 9,
    "Ba": 10,
    "Hf": 12,
    "Ta": 13,
    "W": 14,
    "Re": 15,
    "Os": 16,
    "Ir": 17,
    "Pt": 18,
    "Au": 11,
    "Hg": 12,
    "Tl": 13,
    "Pb": 14,
    "Bi": 15,
    "La": 11,
    "Ce": 12,
    "Pr": 13,
    "Nd": 14,
    "Pm": 15,
    "Sm": 16,
    "Eu": 17,
    "Gd": 18,
    "Tb": 19,
    "Dy": 20,
    "Ho": 21,
    "Er": 22,
    "Tm": 23,
    "Yb": 24,
    "Lu": 25,
}


def _fft_len(n: int) -> int:
    def good(m: int) -> bool:
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        return m == 1

    m = n + (n % 2)
    while not good(m):
        m += 2
    return m


def get_grid_sizes(
    lattice_ang: np.ndarray,
    encut_ev: float = 680,
    prec: str = "Accurate",
) -> tuple[int, int, int]:
    """Old ChiNet VASP-like grid estimate, rounded to multiples of 16."""
    f_prec = {
        "accurate": 2.0,
        "high": 2.0,
        "normal": 1.5,
        "medium": 1.5,
        "low": 1.5,
        "single": 1.5,
    }.get(prec.lower(), 2.0)
    gcut = sqrt(2 * encut_ev * EV2HA)
    lengths_bohr = np.linalg.norm(lattice_ang, axis=1) * ANG2BOHR
    ngx, ngy, ngz = (_fft_len(ceil(length * gcut * f_prec / pi)) for length in lengths_bohr)
    nx = ceil(ngx / 16) * 16
    ny = ceil(ngy / 16) * 16
    nz = ceil(ngz / 16) * 16
    return int(nx), int(ny), int(nz)


def minimal_image(df: np.ndarray) -> np.ndarray:
    return (df + 0.5) % 1.0 - 0.5


def make_interp(r: np.ndarray, rho: np.ndarray):
    return lambda rq: np.interp(rq, r, rho, left=rho[0], right=0.0)


def choose_rcut(r: np.ndarray, rho: np.ndarray) -> float:
    mask = rho > RHO_THRESH
    return float(min(r[mask][-1], RCUT_MAX)) if np.any(mask) else 0.0


def add_atom(
    grid: np.ndarray,
    fpos: np.ndarray,
    lat: np.ndarray,
    rho_f,
    rcut: float,
) -> None:
    nx, ny, nz = grid.shape
    gx = np.arange(nx) / nx
    gy = np.arange(ny) / ny
    gz = np.arange(nz) / nz
    dx = minimal_image(gx[:, None, None] - fpos[0])
    dy = minimal_image(gy[None, :, None] - fpos[1])
    dz = minimal_image(gz[None, None, :] - fpos[2])
    cart = dx[..., None] * lat[0] + dy[..., None] * lat[1] + dz[..., None] * lat[2]
    r_ang = np.linalg.norm(cart, axis=-1)
    r_bohr = r_ang * ANG_TO_BOHR
    mask = r_bohr <= rcut
    if not np.any(mask):
        return
    vals = np.zeros_like(r_bohr, dtype=np.float64)
    vals[mask] = rho_f(r_bohr[mask])
    grid += vals / BOHR3_TO_ANG3


def build_sad(
    lat: np.ndarray,
    species: list[str],
    counts: list[int],
    frac: np.ndarray,
    shape: tuple[int, int, int],
    neutral_dir: Path,
    valence: Dict[str, float] | None = None,
) -> np.ndarray:
    """Build the project SAD grid in the same convention as old ChiNet."""
    valence = DEFAULT_VALENCE if valence is None else valence
    sad = np.zeros(shape, dtype=np.float64)
    idx = 0
    for sym, n_atoms in zip(species, counts):
        if sym not in valence:
            raise KeyError(f"No default valence configured for element {sym!r}")
        pk = neutral_dir / f"{sym}.pkl"
        if not pk.is_file():
            raise FileNotFoundError(f"Missing neutral density: {pk}")
        with pk.open("rb") as handle:
            data = pickle.load(handle)
        r_bohr = np.asarray(data["r_grid_bohr"])
        rho_neu = np.asarray(data["rho_neutral"]) * valence[sym]
        rho_f = make_interp(r_bohr, rho_neu)
        rcut = choose_rcut(r_bohr, rho_neu)
        for _ in range(n_atoms):
            add_atom(sad, frac[idx], lat, rho_f, rcut)
            idx += 1

    cell_vol = abs(float(np.linalg.det(lat)))
    voxel_vol = cell_vol / np.prod(shape)
    n_calc = sad.sum() * voxel_vol
    n_target = sum(valence[sym] * n_atoms for sym, n_atoms in zip(species, counts))
    if n_calc > 0:
        sad *= n_target / n_calc
    return (sad * cell_vol).astype(np.float32)


def parse_poscar(
    path: Path,
) -> tuple[np.ndarray, list[str], list[int], np.ndarray, list[str]]:
    """Parse a POSCAR and preserve header lines for ELFCAR output."""
    lines = path.read_text().splitlines()
    i = 0
    _comment = lines[i]
    i += 1
    scale = float(lines[i].strip())
    i += 1
    lattice = np.array([list(map(float, lines[i + j].split())) for j in range(3)]) * scale
    i += 3
    species = lines[i].split()
    i += 1
    counts = list(map(int, lines[i].split()))
    i += 1
    natoms = sum(counts)

    if lines[i].strip().lower().startswith("s"):
        i += 1

    mode = lines[i].strip().lower()
    i += 1
    if not mode.startswith(("d", "c")):
        raise ValueError(f"Expected Direct or Cartesian coordinates in {path}")
    cart = mode.startswith("c")

    frac = []
    for _ in range(natoms):
        while not lines[i].strip():
            i += 1
        coords = list(map(float, lines[i].split()[:3]))
        frac.append(coords)
        i += 1
    frac_arr = np.asarray(frac)
    if cart:
        frac_arr = frac_arr @ np.linalg.inv(lattice)

    header_lines = lines[:i]
    return lattice, species, counts, frac_arr, header_lines


def default_neutral_dir() -> Path:
    return Path(str(resources.files("elfnet").joinpath("neutral_densities")))


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_predictor(checkpoint: Path, device: torch.device) -> ELFPredictor:
    model = ELFPredictor.load_from_checkpoint(str(checkpoint), map_location=device)
    model.eval()
    model.to(device)
    return model


def predict_elf_grid(
    model: ELFPredictor,
    sad_grid: np.ndarray,
    device: torch.device,
    clip: bool = True,
) -> np.ndarray:
    sad_t = torch.from_numpy(sad_grid).unsqueeze(0).unsqueeze(0).to(device)
    with torch.inference_mode():
        pred, _ = model(sad_t)
    elf_grid = pred.squeeze(0).squeeze(0).cpu().numpy()
    if clip:
        elf_grid = np.clip(elf_grid, 0.0, 1.0)
    return elf_grid.astype(np.float32, copy=False)


def write_elfcar(
    out_file: Path,
    header_lines: Sequence[str],
    grid: np.ndarray,
) -> None:
    nx, ny, nz = grid.shape
    with out_file.open("w") as handle:
        handle.write("\n".join(header_lines) + "\n")
        handle.write("\n")
        handle.write(f"   {nx}   {ny}   {nz}\n")
        flat = grid.ravel(order="F")
        for i in range(0, len(flat), 5):
            chunk = flat[i : i + 5]
            handle.write(" ".join(f"{value:1.11E}" for value in chunk) + "\n")


def run_directory(
    checkpoint: str | Path | None,
    inputs: Path,
    outputs: Path,
    neutral_dir: Path | None = None,
    encut: float = 680.0,
    prec: str = "Accurate",
    device: str = "auto",
    clip: bool = True,
) -> list[Path]:
    ckpt = resolve_checkpoint(checkpoint)
    neutral = default_neutral_dir() if neutral_dir is None else Path(neutral_dir).expanduser()
    torch_device = resolve_device(device)
    model = load_predictor(ckpt, torch_device)
    outputs.mkdir(parents=True, exist_ok=True)

    poscars = sorted(inputs.glob("POSCAR_*"))
    if not poscars:
        raise FileNotFoundError(f"No files named POSCAR_* found in {inputs}")

    written: list[Path] = []
    for poscar in poscars:
        suffix = poscar.stem.split("_", 1)[1] if "_" in poscar.stem else poscar.stem
        out_file = outputs / f"ELFCAR_{suffix}.vasp"
        lat, species, counts, frac, header_lines = parse_poscar(poscar)
        shape = get_grid_sizes(lat, encut_ev=encut, prec=prec)
        sad_grid = build_sad(lat, species, counts, frac, shape, neutral)
        pred = predict_elf_grid(model, sad_grid, torch_device, clip=clip)
        write_elfcar(out_file, header_lines, pred)
        written.append(out_file)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict full-grid ELFCAR files from POSCAR inputs with ELFNet."
    )
    parser.add_argument(
        "paths",
        type=Path,
        nargs="+",
        help=(
            "Either CHECKPOINT INPUTS OUTPUTS or INPUTS OUTPUTS. "
            "When CHECKPOINT is omitted, ELFNET_CHECKPOINT or weights/elfnet_sad2elf.ckpt is used."
        ),
    )
    parser.add_argument("--neutral-dir", type=Path, default=None, help="Folder with neutral density .pkl files")
    parser.add_argument("--encut", type=float, default=680.0, help="ENCUT in eV for grid estimation")
    parser.add_argument("--prec", type=str, default="Accurate", help="VASP precision label for grid estimation")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--no-clip", action="store_true", help="Do not clip predictions to [0, 1]")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if len(args.paths) == 2:
        checkpoint = None
        inputs, outputs = args.paths
    elif len(args.paths) == 3:
        checkpoint, inputs, outputs = args.paths
    else:
        parser.error("expected either CHECKPOINT INPUTS OUTPUTS or INPUTS OUTPUTS")
        return 2
    try:
        written = run_directory(
            checkpoint=checkpoint,
            inputs=inputs,
            outputs=outputs,
            neutral_dir=args.neutral_dir,
            encut=args.encut,
            prec=args.prec,
            device=args.device,
            clip=not args.no_clip,
        )
    except Exception as exc:
        print(f"[elfnet-predict] {exc}", file=sys.stderr)
        return 1

    for path in written:
        print(f"[elfnet-predict] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
