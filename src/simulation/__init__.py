"""
Drug-Target Interaction Simulation Module

This module provides tools for simulating and predicting drug-protein interactions,
including binding affinity estimation, molecular property calculation, and
ADMET predictions.
"""

from .simulator import (
    InteractionSimulator,
    DrugPropertyCalculator,
    SimulationReport,
    MolecularProperties,
    ProteinProperties,
    InteractionPrediction,
    ADMETProperties,
)

__all__ = [
    'InteractionSimulator',
    'DrugPropertyCalculator',
    'SimulationReport',
    'MolecularProperties',
    'ProteinProperties',
    'InteractionPrediction',
    'ADMETProperties',
]
