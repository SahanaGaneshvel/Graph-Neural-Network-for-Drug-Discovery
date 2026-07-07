#!/usr/bin/env python
"""
Verify that all required packages are installed and compatible.
Run this before any model code to catch version mismatches early.
"""

import sys
from importlib.metadata import version, PackageNotFoundError


def check_package(name: str, import_name: str = None) -> tuple[bool, str]:
    """Check if a package is installed and get its version."""
    import_name = import_name or name
    try:
        module = __import__(import_name.split('.')[0])
        try:
            ver = version(name)
        except PackageNotFoundError:
            ver = getattr(module, '__version__', 'unknown')
        return True, ver
    except ImportError as e:
        return False, str(e)


def main():
    print("=" * 60)
    print("DTI-GNN Environment Verification")
    print("=" * 60)
    print(f"Python: {sys.version}")
    print("-" * 60)

    packages = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("torch-geometric", "torch_geometric"),
        ("torch-scatter", "torch_scatter"),
        ("torch-sparse", "torch_sparse"),
        ("rdkit", "rdkit"),
        ("fair-esm", "esm"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scipy", "scipy"),
        ("scikit-learn", "sklearn"),
        ("hydra-core", "hydra"),
        ("omegaconf", "omegaconf"),
        ("tqdm", "tqdm"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("wandb", "wandb"),
        ("pytest", "pytest"),
    ]

    all_ok = True
    for pkg_name, import_name in packages:
        ok, ver_or_err = check_package(pkg_name, import_name)
        status = "OK" if ok else "MISSING"
        if ok:
            print(f"  {pkg_name:20s} {ver_or_err:15s} [{status}]")
        else:
            print(f"  {pkg_name:20s} {'---':15s} [{status}] {ver_or_err}")
            all_ok = False

    print("-" * 60)

    # Check CUDA availability
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"CUDA available: Yes")
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA available: No (CPU only)")
    except Exception as e:
        print(f"CUDA check failed: {e}")

    print("-" * 60)

    # Verify PyG and torch compatibility
    try:
        import torch
        import torch_geometric
        import torch_scatter
        import torch_sparse

        # Quick functionality test
        from torch_geometric.data import Data
        import torch

        x = torch.tensor([[1.0], [2.0], [3.0]])
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        data = Data(x=x, edge_index=edge_index)

        print(f"PyG Data creation: OK (nodes={data.num_nodes}, edges={data.num_edges})")

        # Test scatter
        from torch_scatter import scatter_mean
        src = torch.tensor([1.0, 2.0, 3.0, 4.0])
        index = torch.tensor([0, 0, 1, 1])
        result = scatter_mean(src, index)
        assert result.tolist() == [1.5, 3.5], "scatter_mean failed"
        print("torch-scatter: OK")

    except Exception as e:
        print(f"PyG/torch compatibility issue: {e}")
        all_ok = False

    # Verify RDKit
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, Descriptors

        mol = Chem.MolFromSmiles("CCO")
        assert mol is not None, "Failed to parse SMILES"
        assert mol.GetNumAtoms() == 3, "Incorrect atom count"
        print("RDKit SMILES parsing: OK")

    except Exception as e:
        print(f"RDKit issue: {e}")
        all_ok = False

    # Verify ESM
    try:
        import esm
        print("ESM import: OK")
    except Exception as e:
        print(f"ESM issue: {e}")
        all_ok = False

    print("=" * 60)
    if all_ok:
        print("All checks passed!")
        return 0
    else:
        print("Some checks failed - fix before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
