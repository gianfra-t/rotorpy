"""The powered-aero data file: JSON in and out of the typed records in :mod:`.types`."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from rotorpy.vehicles.models.aero import AeroCoefficients, RateDerivatives
from rotorpy.vehicles.models.powered_aero.rotor import PolynomialRotorModel, RotorEnvelope, RotorFit
from rotorpy.vehicles.models.powered_aero.types import (
    CleanData, PairData, PoweredAeroData, ReceiverData, SurfaceIncrements,
)


def _jsonable(value):
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def to_dict(data: PoweredAeroData) -> dict:
    return _jsonable(asdict(data))


def from_dict(payload: dict) -> PoweredAeroData:
    clean = payload["clean"]
    return PoweredAeroData(**{
        **payload,
        "rotors": [RotorFit(**{**rotor, "model": PolynomialRotorModel(**rotor["model"]),
                               "envelope": RotorEnvelope(**rotor["envelope"])})
                   for rotor in payload["rotors"]],
        "clean": CleanData(**{**clean, "coefficients": AeroCoefficients(**clean["coefficients"]),
                              "rate_derivatives": RateDerivatives(**clean["rate_derivatives"])}),
        "receivers": [ReceiverData(**receiver) for receiver in payload["receivers"]],
        "pairs": [PairData(**pair) for pair in payload["pairs"]],
        "surfaces": [SurfaceIncrements(**{**surface, "increments": AeroCoefficients(**surface["increments"])})
                     for surface in payload["surfaces"]],
    })


def save_data(data: PoweredAeroData, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_dict(data), indent=2) + "\n")


def load_data(path) -> PoweredAeroData:
    return from_dict(json.loads(Path(path).read_text()))
