#!/usr/bin/env python3
"""
NeuroPharma AI - application server.

Serves the frontend (frontend/) and a small JSON API backed by the trained
cross-attention GNN.

Endpoints
    GET  /api/health     server status and whether a trained model is loaded
    GET  /api/model      description + test metrics of the loaded ensemble
    GET  /api/examples   example drugs / kinase targets from the Davis dataset
    GET  /api/datasets   statistics of the benchmark datasets on disk
    POST /api/predict    {"smiles": str, "protein_sequence": str}

Usage
    python scripts/serve_app.py                       # http://localhost:8000
    python scripts/serve_app.py --port 9000
    python scripts/serve_app.py --checkpoints "experiments/sweeps/*/proposed_davis_warm/*/model.pt"

If no checkpoint is found the server still runs; predictions then come from a
transparent property-based heuristic and are labelled source="heuristic".
"""

import argparse
import glob
import http.server
import json
import os
import socketserver
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation import InteractionSimulator  # noqa: E402

DEFAULT_CHECKPOINT_GLOB = "models/serving/*.pt"
MAX_BODY_BYTES = 200_000

# Well-known kinase inhibitors in Davis, keyed by PubChem CID (the Davis drug id)
EXAMPLE_DRUGS = {
    "5291": "Imatinib",
    "176870": "Erlotinib",
    "123631": "Gefitinib",
    "3062316": "Dasatinib",
    "208908": "Lapatinib",
    "216239": "Sorafenib",
    "5329102": "Sunitinib",
    "44259": "Staurosporine",
}
EXAMPLE_TARGETS = ["ABL1", "EGFR", "ERBB2", "KIT", "VEGFR2", "SRC", "BRAF", "CDK2"]


class AppState:
    """Model + dataset lookups shared by all request handlers."""

    def __init__(self, checkpoint_glob: str):
        self.lock = threading.Lock()
        self.predictor = None
        self.model_error: Optional[str] = None
        self.davis = None  # dict from load_davis
        self.measured: Dict[tuple, List[float]] = {}
        self.measured_by_id: Dict[tuple, float] = {}
        self.dataset_stats: List[Dict[str, Any]] = []

        self._load_datasets()
        self._load_model(checkpoint_glob)
        self.simulator = InteractionSimulator(self.predictor)

    def _load_datasets(self):
        try:
            from src.data import load_davis, load_kiba, create_interaction_df
        except Exception as exc:  # pragma: no cover - environment problem
            print(f"  ! Could not import data loaders: {exc}")
            return

        for name, loader, desc in [
            ("Davis", load_davis, "Kd of 68 kinase inhibitors against 442 kinases "
                                  "(pKd = 9 - log10(Kd[nM]))"),
            ("KIBA", load_kiba, "KIBA scores integrating Ki, Kd and IC50 for kinase inhibitors"),
        ]:
            try:
                data = loader()
                if name != "Davis":
                    # Stats straight from the affinity matrix: building the full
                    # KIBA DataFrame costs memory a small server does not have
                    import numpy as np
                    m = np.asarray(data["affinity"], dtype=float)
                    valid = ~np.isnan(m) & (m != 0)
                    self.dataset_stats.append({
                        "name": name,
                        "pairs": int(valid.sum()),
                        "drugs": int(valid.any(axis=1).sum()),
                        "targets": int(valid.any(axis=0).sum()),
                        "affinity_mean": float(m[valid].mean()),
                        "affinity_min": float(m[valid].min()),
                        "affinity_max": float(m[valid].max()),
                        "description": desc,
                    })
                    continue
                df = create_interaction_df(data)
                self.dataset_stats.append({
                    "name": name,
                    "pairs": int(len(df)),
                    "drugs": int(df["drug_id"].nunique()),
                    "targets": int(df["protein_id"].nunique()),
                    "affinity_mean": float(df["affinity"].mean()),
                    "affinity_min": float(df["affinity"].min()),
                    "affinity_max": float(df["affinity"].max()),
                    "description": desc,
                })
                if name == "Davis":
                    self.davis = data
                    # Some Davis targets share a sequence (e.g. ABL1 and phosphorylated
                    # ABL1p) but have different measurements, so keep them all
                    for r in df.itertuples(index=False):
                        self.measured.setdefault((r.smiles, r.sequence), []).append(float(r.affinity))
                        self.measured_by_id[(r.smiles, r.protein_id)] = float(r.affinity)
            except FileNotFoundError:
                print(f"  ! {name} not found under data/raw - run the download step")

    def _load_model(self, checkpoint_glob: str):
        paths = sorted(glob.glob(str(PROJECT_ROOT / checkpoint_glob))) or sorted(
            glob.glob(checkpoint_glob)
        )
        if not paths:
            self.model_error = (
                f"No checkpoints match '{checkpoint_glob}'. Train a model "
                "(see README > Training) or pass --checkpoints."
            )
            return
        try:
            from src.inference import AffinityPredictor

            ref_smiles = list(self.davis["drugs"].values()) if self.davis else None
            ref_seqs = list(self.davis["proteins"].values()) if self.davis else None
            self.predictor = AffinityPredictor(
                paths, device="cpu", reference_smiles=ref_smiles, reference_sequences=ref_seqs
            )
        except Exception as exc:
            self.model_error = f"Failed to load checkpoints: {exc}"

    # ---------------------------------------------------------------- views

    def health(self) -> Dict[str, Any]:
        return {
            "status": "healthy",
            "model_loaded": self.predictor is not None,
            "model_error": self.model_error,
        }

    def model_info(self) -> Dict[str, Any]:
        if self.predictor is None:
            return {"loaded": False, "error": self.model_error}
        return {"loaded": True, **self.predictor.describe()}

    def examples(self) -> Dict[str, Any]:
        if not self.davis:
            return {"drugs": [], "targets": []}
        drugs = [
            {"id": cid, "name": name, "smiles": self.davis["drugs"][cid]}
            for cid, name in EXAMPLE_DRUGS.items() if cid in self.davis["drugs"]
        ]
        targets = [
            {"id": pid, "name": pid, "sequence": self.davis["proteins"][pid]}
            for pid in EXAMPLE_TARGETS if pid in self.davis["proteins"]
        ]
        return {"drugs": drugs, "targets": targets}

    def predict(self, smiles: str, sequence: str, protein_id: Optional[str] = None) -> Dict[str, Any]:
        sequence = "".join(sequence.split()).upper()
        if not sequence.isalpha():
            raise ValueError("Protein sequence must contain only one-letter amino-acid codes")
        from rdkit import Chem
        if Chem.MolFromSmiles(smiles) is None:
            raise ValueError(f"Could not parse SMILES: {smiles!r}")
        with self.lock:  # autograd-based explanations are not thread-safe
            interaction = self.simulator.simulate_interaction(smiles, sequence)
            admet = self.simulator.predict_admet(smiles)
        props = self.simulator.property_calculator.calculate_drug_properties(smiles)
        details = interaction.model_details
        from_model = interaction.source == "model"

        return {
            "success": True,
            "source": interaction.source,
            "model": self.model_info() if interaction.source == "model" else None,
            "prediction": {
                "binding_affinity": interaction.binding_affinity,
                "affinity_std": details.get("affinity_std"),
                "member_predictions": details.get("member_predictions"),
                "kd_nm": interaction.binding_affinity_nm,
                "confidence": interaction.confidence,
                "binding_probability": interaction.binding_probability,
                "interaction_type": interaction.interaction_type,
                # Several Davis entries (e.g. ABL1 and its mutants) share one
                # sequence: use the named entry if the client sent its id,
                # otherwise only report a single unambiguous measurement
                "measured_affinity": self._measured(smiles, sequence, protein_id),
                "measured_values": self.measured.get((smiles, sequence), []),
                "drug_similarity": details.get("drug_similarity"),
                "protein_in_training_set": details.get("protein_in_training_set"),
                "sequence_truncated": details.get("truncated", False),
            },
            # Explanations only exist for the trained model; the heuristic
            # fallback's placeholder scores are not shown as "attention"
            "explanation": {
                "atom_symbols": details.get("atom_symbols", []),
                "atom_importance": interaction.atom_importance if from_model else [],
                "residue_importance": interaction.residue_importance if from_model else [],
                "top_regions": details.get("top_regions", []),
            },
            "interactions": interaction.key_interactions,
            "drug_properties": props.to_dict(),
            "admet": admet.to_dict(),
            "notes": interaction.simulation_notes,
        }


    def _measured(self, smiles: str, sequence: str, protein_id: Optional[str]) -> Optional[float]:
        if protein_id and (smiles, protein_id) in self.measured_by_id:
            return self.measured_by_id[(smiles, protein_id)]
        values = self.measured.get((smiles, sequence), [])
        return values[0] if len(values) == 1 else None


class APIHandler(http.server.SimpleHTTPRequestHandler):
    """Static file server for frontend/ plus the JSON API."""

    state: AppState = None  # set in main()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PROJECT_ROOT / "frontend"), **kwargs)

    def do_GET(self):
        path = urlparse(self.path).path
        routes = {
            "/api/health": self.state.health,
            "/api/model": self.state.model_info,
            "/api/examples": self.state.examples,
            "/api/datasets": lambda: {"datasets": self.state.dataset_stats},
        }
        if path in routes:
            self._send_json(routes[path]())
        else:
            super().do_GET()

    def do_POST(self):
        path = urlparse(self.path).path
        if path not in ("/api/predict", "/api/simulate"):
            self._send_json({"error": "Not found"}, 404)
            return

        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY_BYTES:
            self._send_json({"error": "Request body missing or too large"}, 400)
            return
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json({"error": "Invalid JSON"}, 400)
            return

        smiles = str(data.get("smiles", "")).strip()
        sequence = str(data.get("protein_sequence", "")).strip()
        if not smiles or not sequence:
            self._send_json({"error": "Both 'smiles' and 'protein_sequence' are required"}, 400)
            return

        try:
            protein_id = data.get("protein_id")
            self._send_json(self.state.predict(smiles, sequence, str(protein_id) if protein_id else None))
        except ValueError as exc:  # bad SMILES / sequence
            self._send_json({"error": str(exc)}, 422)
        except Exception as exc:  # pragma: no cover - unexpected
            self._send_json({"error": f"Prediction failed: {exc}"}, 500)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def end_headers(self):
        # Always fetch fresh dashboard data / scripts during development
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _send_json(self, data: Any, status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {format % args}")


class ThreadingServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    parser = argparse.ArgumentParser(description="Serve the NeuroPharma AI web app")
    # Hosting platforms (Render, etc.) pass the port in $PORT and need 0.0.0.0
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    parser.add_argument("--checkpoints", default=DEFAULT_CHECKPOINT_GLOB,
                        help="Glob of model.pt files to ensemble (relative to repo root)")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # show request logs immediately

    print("NeuroPharma AI - loading datasets and model...")
    APIHandler.state = AppState(args.checkpoints)
    if APIHandler.state.predictor is not None:
        info = APIHandler.state.predictor.describe()
        print(f"  Model: {info['model_type']} x{info['ensemble_size']} "
              f"({info['parameters']:,} params) trained on {info['dataset']}/{info['split']}")
    else:
        print(f"  ! {APIHandler.state.model_error}")
        print("  ! Predictions will use the property-based heuristic fallback")

    with ThreadingServer((args.host, args.port), APIHandler) as httpd:
        print(f"  Serving on http://{args.host}:{args.port}  (Ctrl+C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


if __name__ == "__main__":
    main()
