"""
Model-level invariants.

The training speed-ups (dynamic padding, per-batch protein de-duplication,
grouped attention) must not change what the models compute; these tests pin
that down.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.run_experiment import ExperimentConfig, create_model
from src.data import (
    DTIDataset,
    ProteinGroupedBatchSampler,
    collate_dti_batch,
    get_sequence_vocab_size,
    get_smiles_vocab_size,
    sequence_to_indices,
    smiles_to_graph,
    smiles_to_indices,
)

MODELS = ["deepdta", "graphdta_gin", "graphdta_gcn", "graphdta_gat", "proposed", "proposed_concat"]

SMILES = [
    "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",  # imatinib
    "COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC",                   # erlotinib
    "CC(=O)Nc1ccc(O)cc1",
]
SEQUENCES = ["MAAGHKLLLPPKAMEERWQ" * 7, "MKTAYIAKQRQISFVKSHFSRQ" * 4, "ACDEFGHIKLMNPQRSTVWY" * 3]


@pytest.fixture(scope="module")
def tiny_df():
    rows = [
        {"drug_id": f"D{i}", "protein_id": f"P{j}", "smiles": s, "sequence": q,
         "affinity": 5.0 + i + 0.5 * j}
        for i, s in enumerate(SMILES) for j, q in enumerate(SEQUENCES)
    ]
    return pd.DataFrame(rows)


def _model(model_type):
    torch.manual_seed(0)
    model, rep, kwargs = create_model(ExperimentConfig(model_type=model_type), torch.device("cpu"))
    model.eval()
    return model, rep


def _forward(model, rep, batch, **kw):
    if rep == "sequence":
        out = model(drug_seq=batch["drug_seq"], protein_seq=batch["protein_seq"])
    else:
        out = model(drug_x=batch["drug_x"], drug_edge_index=batch["drug_edge_index"],
                    drug_batch=batch["drug_batch"], protein_seq=batch["protein_seq"], **kw)
    return (out[0] if isinstance(out, tuple) else out).reshape(-1)


def test_padding_index_is_reserved():
    # Index 0 must never be a real token, otherwise e.g. alanine is masked as padding
    assert sequence_to_indices("A")[0] != 0
    assert (sequence_to_indices("ACDEFGHIKLMNPQRSTVWYX") > 0).all()
    assert (smiles_to_indices("C#N", max_length=3) > 0).all()
    assert sequence_to_indices("A", max_length=4).tolist()[1:] == [0, 0, 0]
    assert get_sequence_vocab_size() == 22
    assert get_smiles_vocab_size() == 59


@pytest.mark.parametrize("model_type", MODELS)
def test_prediction_independent_of_padding(model_type):
    model, rep = _model(model_type)
    graph = smiles_to_graph(SMILES[0])
    x = torch.tensor(graph["x"])
    ei = torch.tensor(graph["edge_index"])
    b = torch.zeros(len(x), dtype=torch.long)
    outs = []
    for pad in [None, 1000]:
        prot = torch.tensor(sequence_to_indices(SEQUENCES[0], pad)).unsqueeze(0)
        drug = torch.tensor(smiles_to_indices(SMILES[0], 100 if pad else len(SMILES[0]))).unsqueeze(0)
        with torch.no_grad():
            batch = {"drug_x": x, "drug_edge_index": ei, "drug_batch": b,
                     "protein_seq": prot, "drug_seq": drug}
            outs.append(_forward(model, rep, batch).item())
    assert outs[0] == pytest.approx(outs[1], abs=1e-5)


@pytest.mark.parametrize("model_type", MODELS)
def test_batched_equals_single(model_type, tiny_df):
    """Batching (with repeated proteins -> de-duplication) changes nothing."""
    model, rep = _model(model_type)
    ds = DTIDataset(tiny_df, drug_representation=rep)
    with torch.no_grad():
        batched = _forward(model, rep, collate_dti_batch([ds[i] for i in range(len(ds))]))
        single = torch.cat([_forward(model, rep, collate_dti_batch([ds[i]])) for i in range(len(ds))])
    assert torch.allclose(batched, single, atol=1e-5)


def test_grouped_attention_matches_explicit(tiny_df):
    """The fast per-protein attention path equals the explicit one (values and grads)."""
    model, rep = _model("proposed")
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    model.train()
    batch = collate_dti_batch([DTIDataset(tiny_df)[i] for i in range(len(tiny_df))])
    grads, outs = [], []
    for explicit in [False, True]:
        model.zero_grad()
        out = _forward(model, rep, batch, return_attention=explicit)
        out.sum().backward()
        outs.append(out.detach())
        grads.append(torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]))
    assert torch.allclose(outs[0], outs[1], atol=1e-5)
    assert torch.allclose(grads[0], grads[1], atol=1e-4)


def test_attention_weights_shape():
    model, _ = _model("proposed")
    graph = smiles_to_graph(SMILES[1])
    x = torch.tensor(graph["x"])
    with torch.no_grad():
        _, attn = model(drug_x=x, drug_edge_index=torch.tensor(graph["edge_index"]),
                        drug_batch=torch.zeros(len(x), dtype=torch.long),
                        protein_seq=torch.tensor(sequence_to_indices(SEQUENCES[1])).unsqueeze(0),
                        return_attention=True)
    assert attn.shape == (1, 4, graph["num_atoms"], len(SEQUENCES[1]))
    assert torch.allclose(attn.sum(-1), torch.ones(1, 4, graph["num_atoms"]), atol=1e-5)


def test_grouped_sampler_covers_every_pair_once():
    rng = np.random.default_rng(0)
    proteins = rng.integers(0, 20, 1000)
    lengths = rng.integers(50, 1000, 20)[proteins]
    sampler = ProteinGroupedBatchSampler(proteins, lengths, batch_size=64, group_size=8, seed=1)
    batches = list(sampler)
    flat = [i for b in batches for i in b]
    assert sorted(flat) == list(range(1000))
    assert all(len(b) <= 64 for b in batches)
    # Grouping really reduces distinct proteins per batch
    assert np.mean([len(set(proteins[b])) for b in batches]) <= 64 / 8 + 2
    # A different epoch gives a different order
    assert [i for b in sampler for i in b] != flat


def test_overfit_tiny_subset(tiny_df):
    """Sanity check from the brief: the model can drive train loss near zero."""
    torch.manual_seed(0)
    model, rep, _ = create_model(ExperimentConfig(model_type="proposed", dropout=0.0),
                                 torch.device("cpu"))
    batch = collate_dti_batch([DTIDataset(tiny_df)[i] for i in range(len(tiny_df))])
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    model.train()
    for _ in range(150):
        opt.zero_grad()
        loss = ((_forward(model, rep, batch) - batch["affinity"]) ** 2).mean()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05
