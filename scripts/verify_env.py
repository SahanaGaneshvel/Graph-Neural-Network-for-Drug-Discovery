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

    # Required packages
    required_packages = [
        ("torch", "torch"),
        ("torch-geometric", "torch_geometric"),
        ("rdkit", "rdkit"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scipy", "scipy"),
        ("scikit-learn", "sklearn"),
        ("pyyaml", "yaml"),
        ("tqdm", "tqdm"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
    ]

    # Optional packages
    optional_packages = [
        ("torchvision", "torchvision"),
        ("torch-scatter", "torch_scatter"),
        ("torch-sparse", "torch_sparse"),
        ("fair-esm", "esm"),
        ("wandb", "wandb"),
        ("pytest", "pytest"),
    ]

    print("Required packages:")
    all_required_ok = True
    for pkg_name, import_name in required_packages:
        ok, ver_or_err = check_package(pkg_name, import_name)
        status = "OK" if ok else "MISSING"
        if ok:
            print(f"  {pkg_name:20s} {ver_or_err:15s} [{status}]")
        else:
            print(f"  {pkg_name:20s} {'---':15s} [{status}] {ver_or_err}")
            all_required_ok = False

    print("\nOptional packages:")
    for pkg_name, import_name in optional_packages:
        ok, ver_or_err = check_package(pkg_name, import_name)
        status = "OK" if ok else "MISSING"
        if ok:
            print(f"  {pkg_name:20s} {ver_or_err:15s} [{status}]")
        else:
            print(f"  {pkg_name:20s} {'---':15s} [{status}]")

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

    # Verify PyG functionality (works without scatter/sparse in PyG 2.5+)
    pyg_ok = False
    try:
        import torch
        import torch_geometric
        from torch_geometric.data import Data
        from torch_geometric.nn import GCNConv, GATConv, GINConv, global_mean_pool

        # Test Data creation
        x = torch.tensor([[1.0], [2.0], [3.0]])
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        data = Data(x=x, edge_index=edge_index)
        print(f"PyG Data creation: OK (nodes={data.num_nodes}, edges={data.num_edges})")

        # Test GNN layer
        conv = GCNConv(1, 4)
        out = conv(x, edge_index)
        print(f"PyG GCNConv: OK (output shape={tuple(out.shape)})")

        # Test pooling
        batch = torch.tensor([0, 0, 0])
        pooled = global_mean_pool(out, batch)
        print(f"PyG global_mean_pool: OK (output shape={tuple(pooled.shape)})")

        pyg_ok = True

    except Exception as e:
        print(f"PyG functionality issue: {e}")

    # Verify RDKit
    rdkit_ok = False
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, Descriptors

        mol = Chem.MolFromSmiles("CCO")
        assert mol is not None, "Failed to parse SMILES"
        assert mol.GetNumAtoms() == 3, "Incorrect atom count"
        print("RDKit SMILES parsing: OK")
        rdkit_ok = True

    except Exception as e:
        print(f"RDKit issue: {e}")

    # Verify ESM (optional)
    esm_ok = False
    try:
        import esm
        print("ESM import: OK")
        esm_ok = True
    except ImportError:
        print("ESM: Not installed (optional - needed for ESM-2 protein embeddings)")

    print("=" * 60)

    # Final verdict
    core_ok = all_required_ok and pyg_ok and rdkit_ok

    if core_ok:
        print("Core requirements satisfied - ready to run experiments!")
        if not esm_ok:
            print("Note: ESM not installed. Use protein_encoder_type='cnn' in configs.")
        return 0
    else:
        print("Core requirements not met - fix before proceeding.")
        if not all_required_ok:
            print("  -> Install missing required packages")
        if not pyg_ok:
            print("  -> Fix PyTorch Geometric installation")
        if not rdkit_ok:
            print("  -> Fix RDKit installation")
        return 1


if __name__ == "__main__":
    sys.exit(main())
