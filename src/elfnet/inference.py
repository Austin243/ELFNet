#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference for the pretrained ELFNet SAD-to-ELF checkpoint.

Usage:
  elfnet-predict /path/to/checkpoint.ckpt ./inputs ./outputs \
      [--encut 520] [--patch-size 16] [--stride auto|INT] [--batch-size 64]
      [--neutral-dir /path/to/neutral_densities] [--device auto|cuda|cpu]
      [--window hann|ones] [--no-clip]

Inputs folder must contain files named 'POSCAR_*'. For each, we compute the VASP-like FFT
grid (NX, NY, NZ) from the lattice and ENCUT (default 520 eV), build a neutral-atom SAD
on that grid, construct Seitz symmetry operations {R|t} in fractional coords, run the
trained model on periodic patches, and write 'ELFCAR_*.vasp' into outputs.

The default packaged neutral densities match the data-generation convention
used by the pretrained model.
"""
from __future__ import annotations

import math, sys, argparse
from importlib import resources
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

# ------------------------------ constants ------------------------------

BOHR_ANG = 0.529177210903  # Angstrom
INV_BOHR_PER_A = 1.0 / BOHR_ANG  # 1/Angstrom per a.u.^-1
EV_PER_HA = 27.211386245988
DTYPE = np.float32
RHO_THRESH = 1e-4
RCUT_MAX = 20.0  # Bohr
BOHR3_TO_ANG3 = BOHR_ANG**3

# ------------------------------ helpers: FFT grid sizing ------------------------------

def _reciprocal_from_lattice(A: np.ndarray) -> np.ndarray:
    """
    Return reciprocal lattice vectors B with rows b1,b2,b3 (in 1/Angstrom)
    such that B @ A = 2*pi*I. A is (3,3) with rows a1,a2,a3 in Angstrom.
    """
    # Standard dual basis: B = 2*pi * (A^{-1})^T
    BinvT = np.linalg.inv(A).T
    B = 2.0 * np.pi * BinvT
    return B  # (3,3)

def _is_235_smooth(n: int) -> bool:
    x = n
    for p in (2, 3, 5):
        while x % p == 0 and x > 1:
            x //= p
    return x == 1

def _next_fft_friendly(n_min: int) -> int:
    """
    Round up to the next even, 2*3*5-smooth integer (VASP prefers such FFT dims).
    """
    n = max(1, int(math.ceil(n_min)))
    if n % 2 == 1:
        n += 1
    while not _is_235_smooth(n):
        n += 2  # keep even
    return n

def vasp_ng_from_poscar_lattice(A: np.ndarray, encut_ev: float = 520.0) -> Tuple[int, int, int]:
    """
    Compute (NX,NY,NZ) as VASP-like FFT grid for wavefunctions (coarse grid) from ENCUT and lattice.
    - A: (3,3) real-space lattice (rows are a1,a2,a3) in Angstrom.
    - ENCUT in eV (assume ENMAX is approximately ENCUT).
    Steps:
      1) Gmax[Angstrom^-1] = sqrt(2 * ENCUT[Ha]) * (1/Bohr)
      2) b_i = reciprocal vectors (1/Angstrom)
      3) N_i,min = ceil( 2 * Gmax / |b_i| )
      4) Round up to even 2*3*5-smooth integer
    """
    encut_ha = float(encut_ev) / EV_PER_HA
    gmax_au = math.sqrt(2.0 * encut_ha)                 # a.u.^-1
    gmax_ainv = gmax_au * INV_BOHR_PER_A               # Angstrom^-1

    B = _reciprocal_from_lattice(A)                    # (3,3), rows = b1,b2,b3
    b_norms = np.linalg.norm(B, axis=1)                # |b1|,|b2|,|b3| in 1/Angstrom
    n_min = 2.0 * gmax_ainv / b_norms                  # along each reciprocal axis

    NX = _next_fft_friendly(int(math.ceil(n_min[0])))
    NY = _next_fft_friendly(int(math.ceil(n_min[1])))
    NZ = _next_fft_friendly(int(math.ceil(n_min[2])))
    return int(NX), int(NY), int(NZ)

# ------------------------------ helpers: neutral SAD construction ------------------------------

def minimal_image(df: np.ndarray) -> np.ndarray:
    return (df + 0.5) % 1.0 - 0.5

def make_interp(r: np.ndarray, rho: np.ndarray):
    return lambda rq: np.interp(rq, r, rho, left=rho[0], right=0.0)

def choose_rcut(r: np.ndarray, rho: np.ndarray) -> float:
    mask = rho > RHO_THRESH
    return float(min(r[mask][-1], RCUT_MAX)) if np.any(mask) else 0.0

def add_atom(grid: np.ndarray, fpos: np.ndarray, lat: np.ndarray, rho_f, rcut_bohr: float) -> None:
    nx, ny, nz = grid.shape
    gx, gy, gz = np.arange(nx) / nx, np.arange(ny) / ny, np.arange(nz) / nz
    dx = minimal_image(gx[:, None, None] - fpos[0])
    dy = minimal_image(gy[None, :, None] - fpos[1])
    dz = minimal_image(gz[None, None, :] - fpos[2])
    cart = (dx[..., None] * lat[0] + dy[..., None] * lat[1] + dz[..., None] * lat[2])
    r_ang = np.linalg.norm(cart, axis=-1)
    r_bohr = r_ang / BOHR_ANG
    mask = r_bohr <= rcut_bohr
    if not np.any(mask):
        return
    vals = np.zeros_like(r_bohr, dtype=np.float64)
    vals[mask] = rho_f(r_bohr[mask])
    grid += vals / BOHR3_TO_ANG3

def build_sad_grid(lat: np.ndarray,
                   species: List[str],
                   counts: List[int],
                   frac: np.ndarray,
                   shape: Tuple[int, int, int],
                   neutral_dir: Path,
                   valence: Dict[str, float]) -> np.ndarray:
    """
    Build neutral-atom SAD grid on the requested (NX,NY,NZ) using supplied neutral densities.
    Scales to match total valence electron count and multiplies by cell volume (to match training).
    """
    sad = np.zeros(shape, dtype=np.float64)
    atom_idx = 0
    for sym, n_atoms in zip(species, counts):
        pk_path = neutral_dir / f"{sym}.pkl"
        if not pk_path.is_file():
            raise FileNotFoundError(f"Missing neutral density file: {pk_path}")
        with pk_path.open("rb") as f:
            import pickle
            data = pickle.load(f)
        r_bohr = np.asarray(data["r_grid_bohr"])
        rho_neu = np.asarray(data["rho_neutral"]) * valence[sym]
        rho_f = make_interp(r_bohr, rho_neu)
        rcut = choose_rcut(r_bohr, rho_neu)
        for _ in range(n_atoms):
            add_atom(sad, frac[atom_idx], lat, rho_f, rcut)
            atom_idx += 1

    cell_vol = float(np.linalg.det(lat))
    if cell_vol < 0:
        import warnings
        warnings.warn("Left-handed lattice detected; using abs(det(lat)) for SAD scaling.")
    cell_vol_abs = abs(cell_vol)
    voxel_vol = cell_vol_abs / float(np.prod(shape))
    n_calc = sad.sum() * voxel_vol
    n_target = sum(valence[s] * n for s, n in zip(species, counts))
    if n_calc > 1e-12:
        sad *= n_target / n_calc
    return (sad * cell_vol_abs).astype(np.float32)

# ------------------------------ symmetry ops ------------------------------

def get_symmetry_ops(lat: np.ndarray, species_list: List[str], counts: List[int], frac: np.ndarray) -> np.ndarray:
    """
    Return (N,4,4) Seitz matrices {R|t} in fractional coordinates (R int8, t in [0,1)).
    """
    try:
        from pymatgen.core import Lattice, Structure
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    except Exception as e:
        raise ImportError("pymatgen is required for symmetry analysis. Please install it.") from e

    full_species = [s for s, n in zip(species_list, counts) for _ in range(n)]
    struct = Structure(Lattice(lat), full_species, frac, validate_proximity=False)
    sga = SpacegroupAnalyzer(struct, symprec=1e-3)

    seitz_ops, seen = [], set()
    for op in sga.get_symmetry_operations(cartesian=False):
        R = op.rotation_matrix.astype(np.int8)
        t = (op.translation_vector % 1.0).astype(np.float32)
        S = np.eye(4, dtype=np.float32)
        S[:3, :3] = R.astype(np.float32)
        S[:3, 3] = t
        key = (R.tobytes(), np.round(t, 8).tobytes())
        if key not in seen:
            seitz_ops.append(S)
            seen.add(key)
    if not seitz_ops:
        S = np.eye(4, dtype=np.float32)
        seitz_ops = [S]
    return np.stack(seitz_ops, axis=0).astype(np.float32)

def sym_patchify(R_int: np.ndarray, t_frac: np.ndarray, origin_frac: np.ndarray) -> np.ndarray:
    """
    Map global {R|t} to patch-local {R|t'} with t' = (R o + t - o) mod 1, in fractional coordinates.
    """
    Ro = np.einsum("nij,j->ni", R_int.astype(np.float64), origin_frac.astype(np.float64))  # (N_ops, 3)
    t_prime = (Ro + t_frac.astype(np.float64) - origin_frac.astype(np.float64)) % 1.0
    t_prime = t_prime.astype(np.float32)
    N = R_int.shape[0]
    seitz = np.zeros((N, 4, 4), dtype=np.float32)
    seitz[:, :3, :3] = R_int.astype(np.float32)
    seitz[:, :3, 3] = t_prime
    seitz[:, 3, 3] = 1.0
    return seitz

# ------------------------------ POSCAR parsing ------------------------------

def parse_poscar(poscar_path: Path) -> Tuple[str, float, np.ndarray, List[str], List[int], np.ndarray]:
    """
    Minimal POSCAR parser that returns:
      title, scale, lattice(3,3), species(list[str]), counts(list[int]), frac_coords(N,3)
    """
    with poscar_path.open("r") as f:
        lines = [ln.rstrip("\n") for ln in f]
    if len(lines) < 8:
        raise ValueError(f"POSCAR seems too short: {poscar_path}")

    title = lines[0].strip()
    scale = float(lines[1].split()[0])
    lat = np.array([[float(x) for x in lines[2+i].split()] for i in range(3)], dtype=np.float64)
    lat = lat * scale  # Angstrom

    # Species and counts lines: handle both selective dynamics and symbol lines variants
    sp_line = lines[5].split()
    cnt_line_idx = 6
    try:
        counts = [int(x) for x in lines[cnt_line_idx].split()]
        species = sp_line if not all(s.isdigit() for s in sp_line) else [f"X{i+1}" for i in range(len(counts))]
    except Exception:
        raise ValueError("This POSCAR format is not supported by this minimalist parser. Use pymatgen.Structure.from_file instead.")

    # Coordinates start: look for 'Direct' or 'Cartesian' after counts (may have optional 'Selective dynamics' line)
    coord_start = cnt_line_idx + 1
    if lines[coord_start].lower().startswith("s"):
        coord_start += 1
    direct = lines[coord_start].lower().startswith("d")
    coord_start += 1
    natoms = sum(counts)
    coords = np.array([[float(x) for x in lines[coord_start + i].split()[:3]] for i in range(natoms)], dtype=np.float64)
    if not direct:  # Cartesian to fractional
        coords = np.linalg.solve(lat.T, coords.T).T  # frac = A^{-T} * cart
        coords %= 1.0
    return title, scale, lat, species, counts, coords

# ------------------------------ patching + blending utilities ------------------------------

def periodic_patch(arr: np.ndarray, ix: int, iy: int, iz: int, ps: int) -> np.ndarray:
    NX, NY, NZ = arr.shape
    xs = (ix + np.arange(ps))
    ys = (iy + np.arange(ps))
    zs = (iz + np.arange(ps))
    out = np.take(arr, xs, axis=0, mode="wrap")
    out = np.take(out, ys, axis=1, mode="wrap")
    out = np.take(out, zs, axis=2, mode="wrap")
    return out

def hann1(n: int) -> np.ndarray:
    """Periodic Hann window of length n: w[k] = 0.5*(1 - cos(2*pi*k/n))."""
    k = np.arange(n, dtype=np.float64)
    w = 0.5 * (1.0 - np.cos(2.0 * np.pi * k / float(n)))
    return w.astype(np.float32)

def make_window3(ps: int, kind: str = "hann") -> np.ndarray:
    if kind == "hann":
        wx = hann1(ps); wy = hann1(ps); wz = hann1(ps)
        W = wx[:, None, None] * wy[None, :, None] * wz[None, None, :]
        return W.astype(np.float32)
    elif kind == "ones":
        return np.ones((ps, ps, ps), dtype=np.float32)
    else:
        raise ValueError(f"Unknown window kind: {kind}")

def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b

# ------------------------------ writing ELFCAR ------------------------------

def write_elfcar(struct_title: str,
                 lat: np.ndarray,
                 species: List[str],
                 counts: List[int],
                 frac: np.ndarray,
                 grid: np.ndarray,
                 out_path: Path) -> None:
    """
    Write a VASP volumetric file (ELFCAR-like) with the given structure header and scalar field.
    We prefer pymatgen if available; else fall back to a minimal writer.
    """
    try:
        from pymatgen.core import Lattice, Structure
        from pymatgen.io.vasp.outputs import VolumetricData
        sps = [s for s, n in zip(species, counts) for _ in range(n)]
        struct = Structure(Lattice(lat), sps, frac, validate_proximity=False)
        vol = VolumetricData(struct.lattice, {"elf": grid.astype(float)}, structure=struct)
        vol.write_file(str(out_path))
        return
    except Exception:
        pass  # fallback to manual writer below

    # --- Minimal CHGCAR-like writer ---
    with out_path.open("w") as f:
        # Title and scale
        f.write(f"{struct_title}\n")
        f.write(f"   1.00000000000000\n")
        # Lattice vectors
        for i in range(3):
            f.write(f"   {lat[i,0]:20.16f} {lat[i,1]:20.16f} {lat[i,2]:20.16f}\n")
        # Species and counts
        f.write("  " + "  ".join(species) + "\n")
        f.write("  " + "  ".join(str(n) for n in counts) + "\n")
        f.write("Direct\n")
        for r in frac:
            f.write(f"  {r[0]:.16f}  {r[1]:.16f}  {r[2]:.16f}\n")
        # Grid header
        NX, NY, NZ = grid.shape
        f.write(f"  {NX:5d}  {NY:5d}  {NZ:5d}\n")
        # Data in 5-per-line as in VASP
        flat = grid.reshape(-1)
        for i in range(0, flat.size, 5):
            chunk = flat[i:i+5]
            f.write(" ".join(f"{x:16.11E}" for x in chunk) + "\n")

# ------------------------------ model inference on patches ------------------------------

def load_model_from_checkpoint(ckpt: Path, device_opt: str = "auto"):
    """
    Load the Sad2Elf model once so repeated patch calls reuse parameters.
    """
    import torch
    from .model import Sad2ElfLitModule

    if device_opt == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_opt)

    torch.set_grad_enabled(False)
    model = Sad2ElfLitModule.load_from_checkpoint(str(ckpt), map_location=device)
    model.eval()
    model.to(device)
    return model, device


def run_model_on_patches(ckpt: Path,
                         sad_grid: np.ndarray,
                         seitz_global: np.ndarray,
                         orig_shape: Tuple[int,int,int],
                         patch_size: int = 32,
                         stride: Optional[int] = None,
                         batch_size: int = 64,
                         device_opt: str = "auto",
                         clip01: bool = True,
                         window: str = "hann",
                         model: Optional["torch.nn.Module"] = None,
                         origin_offset: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
    """
    Tile the SAD grid with periodic patches, apply per-patch Seitz mapping, run the model, and
    reassemble the predicted ELF via overlap-and-blend (Hann window by default).
    """
    import torch
    torch.set_grad_enabled(False)

    # Device / model selection
    if model is None:
        model, device = load_model_from_checkpoint(ckpt, device_opt=device_opt)
    else:
        device = next(model.parameters()).device

    NX, NY, NZ = sad_grid.shape
    ps = int(patch_size)
    st = int(ps // 2 if (stride is None or stride == "auto") else stride)
    st = max(1, st)
    min_dim = min(NX, NY, NZ)
    if min_dim < ps:
        st = 1
    else:
        max_stride = max(1, min_dim // 2)
        st = min(st, max_stride)
    if origin_offset is None:
        origin_offset = (0, 0, 0)
    ox, oy, oz = (int(origin_offset[0]) % NX,
                  int(origin_offset[1]) % NY,
                  int(origin_offset[2]) % NZ)

    # Pre-extract integer rotation and fractional translation from global Seitz
    R_int = seitz_global[:, :3, :3].astype(np.int8)
    t_frac = seitz_global[:, :3, 3].astype(np.float32)

    # Accumulators for overlap-averaging
    pred_sum = np.zeros_like(sad_grid, dtype=np.float64)
    pred_w   = np.zeros_like(sad_grid, dtype=np.float64)

    # Weight window
    w_patch = make_window3(ps, window)

    # Start positions along each axis using the chosen stride
    xs0_base = list(range(0, NX, st)) or [0]
    ys0_base = list(range(0, NY, st)) or [0]
    zs0_base = list(range(0, NZ, st)) or [0]
    xs0 = [ (x + ox) % NX for x in xs0_base ]
    ys0 = [ (y + oy) % NY for y in ys0_base ]
    zs0 = [ (z + oz) % NZ for z in zs0_base ]

    # Build list of all patch origins
    origins = [(ix, iy, iz) for ix in xs0 for iy in ys0 for iz in zs0]

    # Batched inference
    b = int(batch_size)
    for start in range(0, len(origins), b):
        batch_origins = origins[start:start+b]
        sad_batch = []
        seitz_batch = []
        mask_batch = []
        shape_batch = []

        for (ix, iy, iz) in batch_origins:
            sad_patch = periodic_patch(sad_grid, ix, iy, iz, ps).astype(np.float32)  # (ps,ps,ps)
            sad_batch.append(sad_patch[None, None, ...])  # (1,1,ps,ps,ps)

            origin_frac = np.array([ix / NX, iy / NY, iz / NZ], dtype=np.float32)
            seitz_patch = sym_patchify(R_int, t_frac, origin_frac)  # (R,4,4)
            seitz_batch.append(seitz_patch[None, ...])              # (1,R,4,4)
            mask_batch.append(np.ones((seitz_patch.shape[0],), dtype=np.bool_)[None, ...])  # (1,R)
            shape_batch.append(np.array([NX, NY, NZ], dtype=np.int64)[None, ...])           # (1,3)

        import torch
        X = torch.from_numpy(np.concatenate(sad_batch, axis=0)).to(device)       # (B,1,ps,ps,ps)
        SE = torch.from_numpy(np.concatenate(seitz_batch, axis=0)).to(device)    # (B,R,4,4)
        MSK = torch.from_numpy(np.concatenate(mask_batch, axis=0)).to(device)    # (B,R)
        SHP = torch.from_numpy(np.concatenate(shape_batch, axis=0)).to(device)   # (B,3)

        # Forward
        Y = model.forward(X, SE, MSK, SHP)  # (B,1,ps,ps,ps)
        y = Y.detach().cpu().numpy()[:,0]   # (B,ps,ps,ps)

        # Place back into full grid with wrap-around and blend with window
        for idx, (ix, iy, iz) in enumerate(batch_origins):
            patch = y[idx]
            if clip01:
                patch = np.clip(patch, 0.0, 1.0)
            xs = (ix + np.arange(ps)) % NX
            ys = (iy + np.arange(ps)) % NY
            zs = (iz + np.arange(ps)) % NZ
            # Accumulate weighted patch
            pred_sum[np.ix_(xs, ys, zs)] += (patch * w_patch).astype(np.float64)
            pred_w  [np.ix_(xs, ys, zs)] += w_patch.astype(np.float64)

    # Normalize where covered
    pred_w[pred_w == 0] = 1.0
    out = (pred_sum / pred_w).astype(np.float32)
    return out

# ------------------------------ CLI ------------------------------

DEFAULT_VALENCE = {
    'H': 1.0, 'He': 2.0, 'Li': 3.0, 'Be': 4.0, 'B': 3.0, 'C': 4.0, 'N': 5.0,
    'O': 6.0, 'F': 7.0, 'Ne': 8.0, 'Na': 7.0, 'Mg': 8.0, 'Al': 3.0, 'Si': 4.0,
    'P': 5.0, 'S': 6.0, 'Cl': 7.0, 'Ar': 8.0, 'K': 7.0, 'Ca': 8.0, 'Sc': 11.0,
    'Ti': 10.0, 'V': 11.0, 'Cr': 12.0, 'Mn': 13.0, 'Fe': 14.0, 'Co': 15.0, 'Ni': 16.0,
    'Cu': 17.0, 'Zn': 12.0, 'Ga': 3.0, 'Ge': 4.0, 'As': 5.0, 'Se': 6.0, 'Br': 7.0,
    'Kr': 8.0, 'Rb': 7.0, 'Sr': 10.0, 'Y': 11.0, 'Zr': 12.0, 'Nb': 11.0, 'Mo': 12.0,
    'Tc': 13.0, 'Ru': 14.0, 'Rh': 15.0, 'Pd': 16.0, 'Ag': 17.0, 'Cd': 12.0, 'In': 3.0,
    'Sn': 4.0, 'Sb': 5.0, 'Te': 6.0, 'I': 7.0, 'Xe': 8.0, 'Cs': 9.0, 'Ba': 10.0,
    'La': 11.0, 'Ce': 12.0, 'Pr': 13.0, 'Nd': 14.0, 'Pm': 15.0, 'Sm': 16.0, 'Eu': 17.0,
    'Gd': 18.0, 'Tb': 19.0, 'Dy': 20.0, 'Ho': 21.0, 'Er': 22.0, 'Tm': 23.0, 'Yb': 24.0,
    'Lu': 25.0, 'Hf': 10.0, 'Ta': 11.0, 'W': 14.0, 'Re': 13.0, 'Os': 14.0, 'Ir': 9.0,
    'Pt': 16.0, 'Au': 11.0, 'Hg': 12.0, 'Tl': 3.0, 'Pb': 4.0, 'Bi': 5.0, 'Ac': 11.0,
    'Th': 12.0, 'Pa': 13.0, 'U': 14.0, 'Np': 15.0, 'Pu': 16.0
}

def default_neutral_dir() -> Path:
    """Return the packaged neutral density table directory."""
    return Path(str(resources.files("elfnet") / "neutral_densities"))

def _args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("checkpoint", type=Path, help="Path to Lightning checkpoint (.ckpt)")
    p.add_argument("inputs", type=Path, help="Folder containing files named 'POSCAR_*'")
    p.add_argument("outputs", type=Path, help="Folder to receive 'ELFCAR_*.vasp'")
    p.add_argument("--encut", type=float, default=520.0, help="ENCUT/ENMAX in eV used to size the FFT grid (NX,NY,NZ)")
    p.add_argument("--patch-size", type=int, default=32, help="Patch edge length used at train time")
    p.add_argument("--stride", type=str, default="auto", help="'auto' (=patch_size//2) or integer voxel stride")
    p.add_argument("--batch-size", type=int, default=64, help="Number of patches per forward pass")
    p.add_argument("--neutral-dir", type=Path, default=default_neutral_dir(),
                   help="Directory with {Element}.pkl neutral densities")
    p.add_argument("--identity-symmetry", action="store_true",
                   help="Skip symmetry detection and use only the identity operation")
    p.add_argument("--device", type=str, default="auto", choices=["auto","cuda","cpu"], help="Inference device")
    p.add_argument("--window", type=str, default="hann", choices=["hann","ones"], help="Blending window")
    p.add_argument("--no-clip", action="store_true", help="Disable output clamping to [0,1]")
    return p.parse_args()

def main():
    args = _args()
    args.outputs.mkdir(parents=True, exist_ok=True)

    # Parse stride
    stride_val: Optional[int]
    if isinstance(args.stride, str) and args.stride != "auto":
        try:
            stride_val = int(args.stride)
        except Exception:
            print(f"[inference] --stride must be 'auto' or an integer; got {args.stride}", file=sys.stderr)
            sys.exit(2)
    else:
        stride_val = None  # will be resolved to ps//2

    # Discover inputs
    poscars = sorted([p for p in args.inputs.iterdir() if p.name.startswith("POSCAR_") and p.is_file()])
    if not poscars:
        print(f"[inference] No files named 'POSCAR_*' found in {args.inputs}", file=sys.stderr)
        sys.exit(1)

    # Optional: import pymatgen early to validate availability (but we have a manual writer fallback).
    try:
        import pymatgen  # noqa: F401
    except Exception:
        print("[inference] pymatgen not found. Will use the minimal ELFCAR writer.", file=sys.stderr)

    for poscar in poscars:
        suffix = poscar.name[len("POSCAR_"):]
        elf_out = args.outputs / f"ELFCAR_{suffix}.vasp"

        # Parse POSCAR
        title, scale, lat, species, counts, frac = parse_poscar(poscar)

        # Compute VASP-like NX,NY,NZ from ENCUT and lattice
        NX, NY, NZ = vasp_ng_from_poscar_lattice(lat, encut_ev=args.encut)
        shape = (NX, NY, NZ)
        print(f"[inference] {poscar.name}: grid (NX,NY,NZ) = {shape}, ps={args.patch_size}, stride={'auto' if stride_val is None else stride_val}, window={args.window}")

        # Build SAD on this grid
        sad = build_sad_grid(
            lat=lat,
            species=species,
            counts=counts,
            frac=frac,
            shape=shape,
            neutral_dir=args.neutral_dir,
            valence=DEFAULT_VALENCE,
        )

        # Global symmetry ops in fractional coords. If pymatgen is unavailable,
        # fall back to the identity op so inference remains usable.
        if args.identity_symmetry:
            seitz_global = np.eye(4, dtype=np.float32)[None, ...]
        else:
            try:
                seitz_global = get_symmetry_ops(lat, species, counts, frac)
            except ImportError as exc:
                print(f"[inference] {exc}. Falling back to identity symmetry.", file=sys.stderr)
                seitz_global = np.eye(4, dtype=np.float32)[None, ...]

        # Run inference on patches (with overlap-and-blend)
        pred = run_model_on_patches(
            ckpt=args.checkpoint,
            sad_grid=sad,
            seitz_global=seitz_global,
            orig_shape=shape,
            patch_size=args.patch_size,
            stride=stride_val,
            batch_size=args.batch_size,
            device_opt=args.device,
            clip01=(not args.no_clip),
            window=args.window,
        )

        # Write ELFCAR
        write_elfcar(
            struct_title=title,
            lat=lat,
            species=species,
            counts=counts,
            frac=frac,
            grid=pred,
            out_path=elf_out,
        )
        print(f"[inference] wrote {elf_out}")

if __name__ == "__main__":
    main()
