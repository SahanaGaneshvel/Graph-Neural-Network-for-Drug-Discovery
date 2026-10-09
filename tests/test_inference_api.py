"""Checkpoint -> AffinityPredictor -> HTTP API, end to end (untrained weights)."""

import json
import threading
import urllib.error
import urllib.request

import pytest
import torch

from scripts.run_experiment import ExperimentConfig, create_model
from src.inference import AffinityPredictor, top_regions

IMATINIB = "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5"
KINASE = "MENFQKVEKIGEGTYGVVYKARNKLTGEVVALKKIRLDTETEGVPSTAIREISLLKELNHPNIVKLL"


def _write_checkpoint(path, model_type="proposed", seed=0):
    torch.manual_seed(seed)
    config = ExperimentConfig(model_type=model_type)
    model, _, kwargs = create_model(config, torch.device("cpu"))
    torch.save({"model_type": model_type, "model_kwargs": kwargs, "state_dict": model.state_dict(),
                "max_protein_length": 1000, "dataset": "davis", "split": "warm", "seed": seed,
                "test_metrics": {"ci": 0.8 + seed / 100}}, path)
    return path


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory):
    d = tmp_path_factory.mktemp("ckpt")
    return [_write_checkpoint(d / f"m{s}.pt", seed=s) for s in range(2)]


def test_predictor_ensemble(checkpoints):
    pred = AffinityPredictor(checkpoints, device="cpu", reference_smiles=[IMATINIB],
                             reference_sequences=[KINASE])
    out = pred.predict(IMATINIB, KINASE)
    assert len(out["member_predictions"]) == 2
    assert out["affinity"] == pytest.approx(sum(out["member_predictions"]) / 2)
    assert out["affinity_std"] >= 0
    assert out["drug_similarity"] == pytest.approx(1.0)
    assert out["protein_in_training_set"] is True
    assert len(out["atom_importance"]) == len(out["atom_symbols"]) == out["num_atoms"]
    assert len(out["residue_importance"]) == len(KINASE)
    assert max(out["residue_importance"]) == pytest.approx(1.0)
    assert out["top_regions"]
    info = pred.describe()
    assert info["ensemble_size"] == 2
    assert info["test_metrics"]["ci"]["mean"] == pytest.approx(0.805)


def test_predictor_is_deterministic(checkpoints):
    pred = AffinityPredictor(checkpoints[:1], device="cpu")
    a = pred.predict(IMATINIB, KINASE)["affinity"]
    b = pred.predict(IMATINIB, " " + KINASE.lower() + "\n")["affinity"]  # whitespace / case
    assert a == pytest.approx(b)


@pytest.mark.parametrize("model_type", ["deepdta", "graphdta_gin", "proposed_concat"])
def test_predictor_baselines(tmp_path, model_type):
    pred = AffinityPredictor([_write_checkpoint(tmp_path / "m.pt", model_type)], device="cpu")
    out = pred.predict(IMATINIB, KINASE)
    assert isinstance(out["affinity"], float)
    assert "residue_importance" not in out  # no attention in these models


def test_predictor_rejects_bad_input(checkpoints):
    pred = AffinityPredictor(checkpoints[:1], device="cpu")
    with pytest.raises(ValueError):
        pred.predict("not-a-smiles((", KINASE)
    with pytest.raises(ValueError):
        pred.predict(IMATINIB, "   ")


def test_bare_state_dict_is_rejected(tmp_path):
    model, _, _ = create_model(ExperimentConfig(model_type="proposed"), torch.device("cpu"))
    torch.save(model.state_dict(), tmp_path / "bare.pt")
    with pytest.raises(ValueError, match="metadata"):
        AffinityPredictor([tmp_path / "bare.pt"], device="cpu")


def test_top_regions_non_overlapping():
    import numpy as np
    scores = np.zeros(100)
    scores[10:15] = 1.0
    scores[60:64] = 0.8
    regions = top_regions(scores, "A" * 100, window=5, k=3)
    assert regions[0]["start"] == 11 and regions[0]["end"] == 15
    spans = [(r["start"], r["end"]) for r in regions]
    for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
        assert a1 < b0


@pytest.fixture(scope="module")
def server(checkpoints):
    from scripts import serve_app
    glob = str(checkpoints[0].parent / "*.pt")
    serve_app.APIHandler.state = serve_app.AppState(glob)
    httpd = serve_app.ThreadingServer(("127.0.0.1", 0), serve_app.APIHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def _get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.status, json.loads(r.read())


def _post(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_api_health_and_model(server):
    status, body = _get(server + "/api/health")
    assert status == 200 and body["model_loaded"] is True
    status, body = _get(server + "/api/model")
    assert body["loaded"] and body["ensemble_size"] == 2


def test_api_examples_and_datasets(server):
    _, body = _get(server + "/api/examples")
    names = {d["name"] for d in body["drugs"]}
    assert "Imatinib" in names
    assert any(t["id"] == "ABL1" for t in body["targets"])
    _, body = _get(server + "/api/datasets")
    davis = next(d for d in body["datasets"] if d["name"] == "Davis")
    assert (davis["pairs"], davis["drugs"], davis["targets"]) == (30056, 68, 442)


def test_api_predict_known_pair(server):
    _, examples = _get(server + "/api/examples")
    drug = next(d for d in examples["drugs"] if d["name"] == "Imatinib")
    target = next(t for t in examples["targets"] if t["id"] == "ABL1")
    status, body = _post(server + "/api/predict",
                         {"smiles": drug["smiles"], "protein_sequence": target["sequence"],
                          "protein_id": "ABL1"})
    assert status == 200 and body["success"] and body["source"] == "model"
    p = body["prediction"]
    assert p["measured_affinity"] == pytest.approx(8.96, abs=0.01)  # Davis Imatinib x ABL1
    assert len(p["measured_values"]) > 1               # ABL1 mutants share the sequence
    # Without an id the ambiguous measurement is not reported as a single value
    _, body2 = _post(server + "/api/predict",
                     {"smiles": drug["smiles"], "protein_sequence": target["sequence"]})
    assert body2["prediction"]["measured_affinity"] is None
    assert p["kd_nm"] == pytest.approx(10 ** (9 - p["binding_affinity"]), rel=1e-2)
    assert 0 <= p["confidence"] <= 1
    assert body["explanation"]["top_regions"]
    assert body["drug_properties"]["molecular_weight"] == pytest.approx(493.6, abs=0.5)
    assert "absorption" in body["admet"]


def test_api_rejects_bad_requests(server):
    assert _post(server + "/api/predict", {"smiles": "CCO"})[0] == 400
    assert _post(server + "/api/predict", {"smiles": "C1CC(", "protein_sequence": KINASE})[0] == 422
    assert _post(server + "/api/unknown", {})[0] == 404


def test_static_frontend_is_served(server):
    with urllib.request.urlopen(server + "/", timeout=10) as r:
        assert b"AffiniGraph" in r.read()
