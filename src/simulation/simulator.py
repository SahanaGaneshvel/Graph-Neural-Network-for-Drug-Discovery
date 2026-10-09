"""
Drug-Target Interaction Simulation Model

This module provides a comprehensive simulation system for predicting and
visualizing drug-protein interactions. It includes:

- Binding affinity prediction
- Molecular property calculation
- Interaction site prediction
- Binding mode simulation
- ADMET property estimation
"""

import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from pathlib import Path
import math

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem, Draw
    from rdkit.Chem.Draw import rdMolDraw2D
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class MolecularProperties:
    """Calculated molecular properties of a drug."""
    molecular_weight: float
    logp: float  # Lipophilicity
    hbd: int     # Hydrogen bond donors
    hba: int     # Hydrogen bond acceptors
    tpsa: float  # Topological polar surface area
    rotatable_bonds: int
    aromatic_rings: int
    heavy_atoms: int
    qed: float   # Quantitative Estimate of Drug-likeness
    lipinski_violations: int
    drug_likeness_score: float

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ProteinProperties:
    """Calculated properties of a target protein."""
    length: int
    molecular_weight: float
    isoelectric_point: float
    hydrophobicity: float
    secondary_structure_pred: Dict[str, float]
    binding_site_prediction: List[int]
    conservation_score: float

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class InteractionPrediction:
    """Predicted drug-target interaction details."""
    binding_affinity: float           # pKd/pKi value
    binding_affinity_nm: float        # Kd in nanomolar
    confidence: float                 # Prediction confidence (0-1)
    uncertainty: float                # Uncertainty estimate
    interaction_type: str             # Strong/Moderate/Weak/None
    binding_probability: float        # Probability of binding
    key_interactions: List[Dict]      # Predicted interaction types
    atom_importance: List[float]      # Importance of each drug atom
    residue_importance: List[float]   # Importance of each protein residue
    predicted_binding_site: List[int] # Predicted binding residues
    simulation_notes: List[str]       # Explanatory notes
    source: str = "heuristic"         # "model" (trained GNN) or "heuristic"
    model_details: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ADMETProperties:
    """Predicted ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) properties."""
    absorption: Dict[str, Any]
    distribution: Dict[str, Any]
    metabolism: Dict[str, Any]
    excretion: Dict[str, Any]
    toxicity: Dict[str, Any]
    overall_score: float
    drug_warnings: List[str]

    def to_dict(self) -> Dict:
        return asdict(self)


class DrugPropertyCalculator:
    """Calculate molecular properties of drug compounds."""

    # Amino acid properties for protein analysis
    AA_MW = {
        'A': 89.09, 'R': 174.20, 'N': 132.12, 'D': 133.10, 'C': 121.15,
        'E': 147.13, 'Q': 146.15, 'G': 75.07, 'H': 155.16, 'I': 131.17,
        'L': 131.17, 'K': 146.19, 'M': 149.21, 'F': 165.19, 'P': 115.13,
        'S': 105.09, 'T': 119.12, 'W': 204.23, 'Y': 181.19, 'V': 117.15,
    }

    AA_HYDROPHOBICITY = {
        'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
        'E': -3.5, 'Q': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
        'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
        'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
    }

    AA_PK = {
        'D': 3.9, 'E': 4.1, 'H': 6.0, 'C': 8.3, 'Y': 10.1, 'K': 10.5, 'R': 12.5,
    }

    def calculate_drug_properties(self, smiles: str) -> MolecularProperties:
        """
        Calculate comprehensive molecular properties from SMILES.

        Args:
            smiles: SMILES string of the drug molecule

        Returns:
            MolecularProperties dataclass with calculated values
        """
        if RDKIT_AVAILABLE:
            return self._calculate_rdkit_properties(smiles)
        else:
            return self._calculate_estimated_properties(smiles)

    def _calculate_rdkit_properties(self, smiles: str) -> MolecularProperties:
        """Calculate properties using RDKit."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return self._calculate_estimated_properties(smiles)

        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        tpsa = Descriptors.TPSA(mol)
        rotatable = Descriptors.NumRotatableBonds(mol)
        aromatic = Descriptors.NumAromaticRings(mol)
        heavy = Descriptors.HeavyAtomCount(mol)

        try:
            from rdkit.Chem.QED import qed
            qed_score = qed(mol)
        except:
            qed_score = self._estimate_qed(mw, logp, hbd, hba, tpsa, rotatable)

        # Lipinski's Rule of Five violations
        violations = sum([
            mw > 500,
            logp > 5,
            hbd > 5,
            hba > 10,
        ])

        # Drug-likeness score (0-1)
        drug_likeness = self._calculate_drug_likeness(
            mw, logp, hbd, hba, tpsa, rotatable, violations
        )

        return MolecularProperties(
            molecular_weight=round(mw, 2),
            logp=round(logp, 2),
            hbd=hbd,
            hba=hba,
            tpsa=round(tpsa, 2),
            rotatable_bonds=rotatable,
            aromatic_rings=aromatic,
            heavy_atoms=heavy,
            qed=round(qed_score, 3),
            lipinski_violations=violations,
            drug_likeness_score=round(drug_likeness, 3),
        )

    def _calculate_estimated_properties(self, smiles: str) -> MolecularProperties:
        """Estimate properties from SMILES without RDKit."""
        # Basic estimations from SMILES structure
        length = len(smiles)

        # Count common features
        aromatic = smiles.count('c') + smiles.count('n') // 2
        heavy = sum(1 for c in smiles if c.isupper())
        hbd = smiles.count('N') + smiles.count('O') - smiles.count('=O')
        hba = smiles.count('N') + smiles.count('O')
        rotatable = smiles.count('-') + length // 20

        # Estimates
        mw = heavy * 12 + smiles.count('N') * 2 + smiles.count('O') * 4 + smiles.count('S') * 20
        logp = aromatic * 0.5 + smiles.count('C') * 0.2 - hba * 0.3
        tpsa = hbd * 20 + hba * 10

        violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
        qed = self._estimate_qed(mw, logp, hbd, hba, tpsa, rotatable)
        drug_likeness = self._calculate_drug_likeness(
            mw, logp, hbd, hba, tpsa, rotatable, violations
        )

        return MolecularProperties(
            molecular_weight=round(mw, 2),
            logp=round(logp, 2),
            hbd=max(0, hbd),
            hba=max(0, hba),
            tpsa=round(tpsa, 2),
            rotatable_bonds=max(0, rotatable),
            aromatic_rings=max(0, aromatic // 6),
            heavy_atoms=heavy,
            qed=round(qed, 3),
            lipinski_violations=violations,
            drug_likeness_score=round(drug_likeness, 3),
        )

    def _estimate_qed(self, mw, logp, hbd, hba, tpsa, rotatable) -> float:
        """Estimate QED from individual properties."""
        # Simplified QED estimation
        scores = []

        # MW desirability (peak around 350)
        scores.append(math.exp(-0.5 * ((mw - 350) / 100) ** 2))

        # LogP desirability (peak around 2.5)
        scores.append(math.exp(-0.5 * ((logp - 2.5) / 1.5) ** 2))

        # HBD desirability (peak around 1)
        scores.append(math.exp(-0.5 * ((hbd - 1) / 1.5) ** 2))

        # HBA desirability (peak around 4)
        scores.append(math.exp(-0.5 * ((hba - 4) / 2) ** 2))

        # Geometric mean
        return np.prod(scores) ** (1 / len(scores))

    def _calculate_drug_likeness(self, mw, logp, hbd, hba, tpsa, rotatable, violations) -> float:
        """Calculate overall drug-likeness score."""
        score = 1.0

        # Penalize Lipinski violations
        score -= violations * 0.15

        # Penalize extreme values
        if mw > 600 or mw < 150:
            score -= 0.1
        if logp > 6 or logp < -1:
            score -= 0.1
        if tpsa > 140:
            score -= 0.1
        if rotatable > 10:
            score -= 0.1

        return max(0, min(1, score))

    def calculate_protein_properties(self, sequence: str) -> ProteinProperties:
        """
        Calculate protein properties from amino acid sequence.

        Args:
            sequence: Amino acid sequence string

        Returns:
            ProteinProperties dataclass
        """
        length = len(sequence)

        # Molecular weight
        mw = sum(self.AA_MW.get(aa, 110) for aa in sequence) - (length - 1) * 18

        # Hydrophobicity (GRAVY score)
        hydrophobicity = sum(
            self.AA_HYDROPHOBICITY.get(aa, 0) for aa in sequence
        ) / max(length, 1)

        # Isoelectric point estimation
        pI = self._estimate_pi(sequence)

        # Secondary structure prediction (simplified)
        ss_pred = self._predict_secondary_structure(sequence)

        # Binding site prediction (simplified - predict conserved regions)
        binding_sites = self._predict_binding_sites(sequence)

        return ProteinProperties(
            length=length,
            molecular_weight=round(mw, 2),
            isoelectric_point=round(pI, 2),
            hydrophobicity=round(hydrophobicity, 3),
            secondary_structure_pred=ss_pred,
            binding_site_prediction=binding_sites,
            conservation_score=0.75,  # Placeholder
        )

    def _estimate_pi(self, sequence: str) -> float:
        """Estimate isoelectric point."""
        # Simplified pI estimation
        pos_charge = sequence.count('K') + sequence.count('R') + sequence.count('H')
        neg_charge = sequence.count('D') + sequence.count('E')

        if pos_charge + neg_charge == 0:
            return 7.0

        # Very simplified estimate
        pI = 7.0 + (pos_charge - neg_charge) / len(sequence) * 3
        return max(3, min(12, pI))

    def _predict_secondary_structure(self, sequence: str) -> Dict[str, float]:
        """Predict secondary structure composition."""
        helix_formers = set('AELM')
        sheet_formers = set('VIY')

        helix_count = sum(1 for aa in sequence if aa in helix_formers)
        sheet_count = sum(1 for aa in sequence if aa in sheet_formers)

        total = len(sequence)
        return {
            'helix': round(helix_count / total, 2) if total > 0 else 0,
            'sheet': round(sheet_count / total, 2) if total > 0 else 0,
            'coil': round(1 - (helix_count + sheet_count) / total, 2) if total > 0 else 1,
        }

    def _predict_binding_sites(self, sequence: str) -> List[int]:
        """Predict potential binding site residues."""
        binding_residues = []

        # Look for conserved motifs
        motifs = ['GXG', 'DFG', 'HRD', 'VAIK', 'APE']

        for motif in motifs:
            for i in range(len(sequence) - len(motif)):
                match = True
                for j, m in enumerate(motif):
                    if m != 'X' and sequence[i + j] != m:
                        match = False
                        break
                if match:
                    binding_residues.extend(range(i, i + len(motif)))

        # Also include aromatic and charged residues as potential binders
        for i, aa in enumerate(sequence):
            if aa in 'FYWHRKED':
                binding_residues.append(i)

        return sorted(set(binding_residues))[:20]  # Top 20


class InteractionSimulator:
    """
    Simulate drug-target interactions and predict binding outcomes.

    This class combines molecular property analysis with machine learning
    predictions to provide comprehensive interaction insights.
    """

    def __init__(self, predictor=None):
        """
        Initialize the simulator.

        Args:
            predictor: A src.inference.AffinityPredictor wrapping trained
                checkpoints. Without one, affinity falls back to a transparent
                property-based heuristic (clearly flagged in the output).
        """
        self.predictor = predictor
        self.property_calculator = DrugPropertyCalculator()

    def simulate_interaction(
        self,
        drug_smiles: str,
        protein_sequence: str,
        detailed: bool = True,
    ) -> InteractionPrediction:
        """
        Simulate the interaction between a drug and target protein.

        Args:
            drug_smiles: SMILES string of the drug
            protein_sequence: Amino acid sequence of the target
            detailed: Whether to compute detailed predictions

        Returns:
            InteractionPrediction with comprehensive results
        """
        # Calculate properties
        drug_props = self.property_calculator.calculate_drug_properties(drug_smiles)
        protein_props = self.property_calculator.calculate_protein_properties(protein_sequence)

        model_output = None
        if self.predictor is not None:
            model_output = self.predictor.predict(drug_smiles, protein_sequence, explain=detailed)
            binding_affinity = model_output["affinity"]
            confidence = self._model_confidence(model_output)
        else:
            binding_affinity = self._estimate_base_affinity(drug_props, protein_props)
            confidence = 0.3  # heuristic estimate, not a trained model

        # Convert pKd to Kd in nM
        kd_nm = 10 ** (9 - binding_affinity)

        # Determine interaction type
        if binding_affinity >= 8:
            interaction_type = "Strong Binder"
        elif binding_affinity >= 6:
            interaction_type = "Moderate Binder"
        elif binding_affinity >= 5.5:
            interaction_type = "Weak Binder"
        else:
            interaction_type = "Non-Binder"

        # Logistic mapping centred on pKd 6 (Kd = 1 uM), a common activity cut-off
        binding_prob = 1 / (1 + np.exp(-(binding_affinity - 6) * 1.5))

        # Predict key interactions
        key_interactions = self._predict_interactions(drug_props, protein_props)

        if model_output is not None and "atom_importance" in model_output:
            atom_importance = model_output["atom_importance"]
        else:
            atom_importance = self._generate_atom_importance(drug_smiles)

        if model_output is not None and "residue_importance" in model_output:
            residue_importance = model_output["residue_importance"]
            binding_site = [
                pos for region in model_output["top_regions"]
                for pos in range(region["start"] - 1, region["end"])
            ]
        else:
            residue_importance = self._generate_residue_importance(protein_sequence)
            binding_site = protein_props.binding_site_prediction

        # Generate explanatory notes
        notes = self._generate_simulation_notes(
            drug_props, protein_props, binding_affinity, interaction_type
        )

        return InteractionPrediction(
            binding_affinity=round(binding_affinity, 3),
            binding_affinity_nm=round(kd_nm, 2),
            confidence=round(confidence, 3),
            uncertainty=round(1 - confidence, 3),
            interaction_type=interaction_type,
            binding_probability=round(binding_prob, 3),
            key_interactions=key_interactions,
            atom_importance=atom_importance,
            residue_importance=residue_importance,
            predicted_binding_site=binding_site,
            simulation_notes=notes,
            source="model" if model_output is not None else "heuristic",
            model_details=model_output or {},
        )

    @staticmethod
    def _model_confidence(model_output: Dict) -> float:
        """
        Heuristic confidence in [0, 1] for a model prediction.

        Combines ensemble agreement (std of member predictions, in pKd units)
        with how similar the query drug is to the training drugs (max Tanimoto).
        It is a ranking aid for the UI, not a calibrated probability.
        """
        std = model_output.get("affinity_std")
        agreement = float(np.exp(-std / 0.5)) if std is not None else 0.7
        similarity = model_output.get("drug_similarity")
        domain = 0.5 + 0.5 * similarity if similarity is not None else 0.75
        return max(0.0, min(1.0, agreement * domain))

    def _estimate_base_affinity(
        self,
        drug_props: MolecularProperties,
        protein_props: ProteinProperties,
    ) -> float:
        """Estimate binding affinity from molecular properties."""
        # Heuristic-based estimation
        affinity = 6.0  # Base value

        # Drug-likeness contribution
        affinity += drug_props.drug_likeness_score * 1.5

        # Size compatibility
        if 300 < drug_props.molecular_weight < 500:
            affinity += 0.5
        if protein_props.length > 200:
            affinity += 0.3

        # Lipophilicity
        if 1 < drug_props.logp < 4:
            affinity += 0.4

        # H-bonding potential
        if 2 <= drug_props.hbd <= 5 and 3 <= drug_props.hba <= 7:
            affinity += 0.5

        return max(2, min(12, affinity))

    def _predict_interactions(
        self,
        drug_props: MolecularProperties,
        protein_props: ProteinProperties,
    ) -> List[Dict]:
        """Predict types of molecular interactions."""
        interactions = []

        # Hydrogen bonds
        if drug_props.hbd > 0 or drug_props.hba > 0:
            interactions.append({
                'type': 'Hydrogen Bonds',
                'description': 'Electrostatic attraction between H-bond donors and acceptors',
                'strength': 'moderate',
                'count': drug_props.hbd + drug_props.hba,
            })

        # Hydrophobic interactions
        if drug_props.logp > 1:
            interactions.append({
                'type': 'Hydrophobic Interactions',
                'description': 'Non-polar contacts between lipophilic groups',
                'strength': 'strong' if drug_props.logp > 3 else 'moderate',
                'count': drug_props.aromatic_rings + 2,
            })

        # Pi-stacking
        if drug_props.aromatic_rings > 0:
            interactions.append({
                'type': 'Pi-Stacking',
                'description': 'Aromatic ring interactions with protein residues (Phe, Tyr, Trp)',
                'strength': 'moderate',
                'count': drug_props.aromatic_rings,
            })

        # Salt bridges (if charged groups present)
        if drug_props.hba > 3 or drug_props.hbd > 2:
            interactions.append({
                'type': 'Electrostatic/Salt Bridge',
                'description': 'Ionic interactions between charged groups',
                'strength': 'strong',
                'count': 1,
            })

        return interactions

    def _generate_atom_importance(self, smiles: str) -> List[float]:
        """Generate synthetic atom importance scores."""
        # Simple heuristic: aromatic and heteroatoms are more important
        importance = []
        base_score = 0.3

        for i, char in enumerate(smiles):
            if char in 'NOS':
                score = 0.8
            elif char in 'noc':
                score = 0.6
            elif char.isupper():
                score = 0.4
            else:
                score = base_score

            importance.append(score)

        # Normalize
        total = sum(importance)
        if total > 0:
            importance = [s / total for s in importance]

        return importance[:50]  # Limit

    def _generate_residue_importance(self, sequence: str) -> List[float]:
        """Generate synthetic residue importance scores."""
        importance = []
        binding_residues = set('FYWHRKED')  # Important for binding

        for aa in sequence:
            importance.append(0.8 if aa in binding_residues else 0.25)

        # Normalize
        total = sum(importance)
        if total > 0:
            importance = [s / total for s in importance]

        return importance

    def _generate_simulation_notes(
        self,
        drug_props: MolecularProperties,
        protein_props: ProteinProperties,
        affinity: float,
        interaction_type: str,
    ) -> List[str]:
        """Generate explanatory notes about the simulation."""
        notes = []

        # Binding strength note
        notes.append(
            f"Predicted binding affinity: pKd = {affinity:.2f} "
            f"(Kd = {10**(9-affinity):.2f} nM)"
        )

        # Drug-likeness
        if drug_props.lipinski_violations == 0:
            notes.append(
                "The compound follows Lipinski's Rule of Five, suggesting good oral bioavailability."
            )
        else:
            notes.append(
                f"The compound has {drug_props.lipinski_violations} Lipinski violations, "
                "which may affect oral bioavailability."
            )

        # Interaction type
        if interaction_type == "Strong Binder":
            notes.append(
                "Strong predicted binding suggests this compound may be a potent inhibitor."
            )
        elif interaction_type == "Moderate Binder":
            notes.append(
                "Moderate binding affinity - optimization may improve potency."
            )
        elif interaction_type == "Weak Binder":
            notes.append(
                "Weak predicted binding - significant optimization likely needed."
            )
        else:
            notes.append(
                "No meaningful binding predicted (Davis reports Kd >= 10 uM, i.e. pKd 5, "
                "for pairs with no measurable binding)."
            )

        # Size considerations
        if drug_props.molecular_weight > 500:
            notes.append(
                "High molecular weight may limit cell permeability and oral absorption."
            )

        # Hydrophobicity
        if drug_props.logp > 5:
            notes.append(
                "High lipophilicity (LogP > 5) may cause solubility issues."
            )
        elif drug_props.logp < 0:
            notes.append(
                "Low lipophilicity may limit membrane permeability."
            )

        return notes

    def predict_admet(self, drug_smiles: str) -> ADMETProperties:
        """
        Predict ADMET properties for a drug compound.

        Args:
            drug_smiles: SMILES string of the drug

        Returns:
            ADMETProperties with predicted values
        """
        props = self.property_calculator.calculate_drug_properties(drug_smiles)
        warnings = []

        # Absorption predictions
        absorption = {
            'oral_bioavailability': 'High' if props.lipinski_violations == 0 else 'Moderate',
            'intestinal_absorption': 'Good' if props.tpsa < 140 else 'Poor',
            'caco2_permeability': 'High' if props.tpsa < 80 else 'Low',
            'pgp_substrate': 'Yes' if props.molecular_weight > 400 else 'No',
        }
        if absorption['intestinal_absorption'] == 'Poor':
            warnings.append("High TPSA may reduce intestinal absorption")

        # Distribution predictions
        distribution = {
            'vd': 'Moderate',  # Volume of distribution
            'bbb_penetration': 'Yes' if props.tpsa < 90 and props.molecular_weight < 450 else 'No',
            'plasma_protein_binding': 'High' if props.logp > 3 else 'Moderate',
        }

        # Metabolism predictions
        metabolism = {
            'cyp_inhibition_risk': 'High' if props.aromatic_rings > 2 else 'Low',
            'metabolic_stability': 'Stable' if props.rotatable_bonds < 8 else 'Unstable',
            'half_life_estimate': 'Long' if props.molecular_weight > 400 else 'Short',
        }
        if metabolism['cyp_inhibition_risk'] == 'High':
            warnings.append("Multiple aromatic rings may cause CYP450 inhibition")

        # Excretion predictions
        excretion = {
            'clearance': 'Moderate',
            'renal_excretion': 'High' if props.tpsa > 80 else 'Low',
        }

        # Toxicity predictions
        toxicity = {
            'herg_inhibition': 'High Risk' if props.logp > 4 else 'Low Risk',
            'hepatotoxicity': 'Low Risk' if props.qed > 0.5 else 'Moderate Risk',
            'mutagenicity': 'Low Risk',
            'carcinogenicity': 'Unknown',
        }
        if toxicity['herg_inhibition'] == 'High Risk':
            warnings.append("High LogP may cause hERG channel inhibition (cardiac risk)")

        # Overall score
        score = props.qed * 0.4 + (1 - props.lipinski_violations / 4) * 0.3
        if absorption['oral_bioavailability'] == 'High':
            score += 0.2
        if toxicity['herg_inhibition'] == 'Low Risk':
            score += 0.1

        return ADMETProperties(
            absorption=absorption,
            distribution=distribution,
            metabolism=metabolism,
            excretion=excretion,
            toxicity=toxicity,
            overall_score=round(score, 3),
            drug_warnings=warnings,
        )


class SimulationReport:
    """Generate comprehensive simulation reports."""

    def __init__(self, predictor=None):
        self.simulator = InteractionSimulator(predictor)

    def generate_full_report(
        self,
        drug_smiles: str,
        protein_sequence: str,
        drug_name: str = "Unknown Drug",
        protein_name: str = "Unknown Target",
    ) -> Dict:
        """
        Generate a complete simulation report.

        Args:
            drug_smiles: Drug SMILES string
            protein_sequence: Protein amino acid sequence
            drug_name: Name of the drug
            protein_name: Name of the target protein

        Returns:
            Complete report dictionary
        """
        # Calculate all properties
        drug_props = self.simulator.property_calculator.calculate_drug_properties(drug_smiles)
        protein_props = self.simulator.property_calculator.calculate_protein_properties(protein_sequence)
        interaction = self.simulator.simulate_interaction(drug_smiles, protein_sequence)
        admet = self.simulator.predict_admet(drug_smiles)

        report = {
            'summary': {
                'drug_name': drug_name,
                'protein_name': protein_name,
                'drug_smiles': drug_smiles,
                'protein_length': len(protein_sequence),
                'prediction_timestamp': str(np.datetime64('now')),
            },
            'drug_properties': drug_props.to_dict(),
            'protein_properties': protein_props.to_dict(),
            'interaction_prediction': interaction.to_dict(),
            'admet_properties': admet.to_dict(),
            'recommendation': self._generate_recommendation(interaction, admet),
        }

        return report

    def _generate_recommendation(
        self,
        interaction: InteractionPrediction,
        admet: ADMETProperties,
    ) -> Dict:
        """Generate drug development recommendations."""
        overall_score = (
            interaction.binding_probability * 0.4 +
            interaction.confidence * 0.2 +
            admet.overall_score * 0.4
        )

        if overall_score > 0.7:
            verdict = "Promising Lead"
            description = "This compound shows favorable binding and ADMET properties. Consider for further development."
        elif overall_score > 0.5:
            verdict = "Potential Lead"
            description = "Moderate potential. Optimization of binding or ADMET properties recommended."
        else:
            verdict = "Not Recommended"
            description = "Significant optimization needed before proceeding."

        return {
            'verdict': verdict,
            'overall_score': round(overall_score, 3),
            'description': description,
            'key_strengths': self._identify_strengths(interaction, admet),
            'areas_for_improvement': self._identify_weaknesses(interaction, admet),
        }

    def _identify_strengths(self, interaction, admet) -> List[str]:
        strengths = []
        if interaction.binding_probability > 0.7:
            strengths.append("Strong predicted binding affinity")
        if interaction.confidence > 0.8:
            strengths.append("High prediction confidence")
        if admet.overall_score > 0.7:
            strengths.append("Favorable ADMET profile")
        if len(admet.drug_warnings) == 0:
            strengths.append("No major drug safety warnings")
        return strengths or ["Further analysis needed"]

    def _identify_weaknesses(self, interaction, admet) -> List[str]:
        weaknesses = []
        if interaction.binding_probability < 0.5:
            weaknesses.append("Weak predicted binding")
        if admet.drug_warnings:
            weaknesses.extend(admet.drug_warnings)
        if admet.toxicity.get('herg_inhibition') == 'High Risk':
            weaknesses.append("Potential cardiac toxicity risk")
        return weaknesses or ["No major concerns identified"]


if __name__ == "__main__":
    # Test simulation
    print("Testing Drug-Target Interaction Simulator...")

    # Example drug (Imatinib-like)
    test_smiles = "Cc1ccc(cc1)C(=O)Nc2ccc(cc2)CN3CCN(CC3)C"

    # Example protein sequence (kinase-like)
    test_sequence = "MENFQKVEKIGEGTYGVVYKARNKLTGEVVALKKIRLDTETEGVPSTAIREISLLKELNHPNIVKLLDVIHTENKLYLVFEFLHQDLKKFMDASALTGIPLPLIKSYLFQLLQGLAFCHSHRVLHRDLKPQNLLINTEGAIKLADFGLARAFGVPVRTYTHEVVTLWYRAPEILLGCKYYSTAVDIWSLGCIFAEMVTRRALFPGDSEIDQLFRIFRTLGTPDEVVWPGVTSMPDYKPSFPKWARQD"

    simulator = InteractionSimulator()
    report_generator = SimulationReport()

    # Test basic simulation
    result = simulator.simulate_interaction(test_smiles, test_sequence)
    print(f"\nBinding Affinity: pKd = {result.binding_affinity}")
    print(f"Interaction Type: {result.interaction_type}")
    print(f"Confidence: {result.confidence}")

    # Test full report
    report = report_generator.generate_full_report(
        test_smiles, test_sequence,
        "Test Compound", "Test Kinase"
    )
    print(f"\nRecommendation: {report['recommendation']['verdict']}")
    print(f"Overall Score: {report['recommendation']['overall_score']}")

    print("\nSimulator test completed!")
