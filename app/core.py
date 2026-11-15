"""Facies prediction for one well: input checks, features, ONNX model and scores.

This module does not depend on the web framework, so it can be tested on its own.
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

BASE = Path(__file__).parent
MODEL_PATH = BASE / "model" / "facies_model.onnx"
SETTINGS_PATH = BASE / "model" / "settings.json"
DEMO_DIR = BASE / "demo"

MAX_ROWS = 20_000

# Other spellings accepted for column names (compared without case)
ALIASES = {
    "Depth": ["depth", "dept", "md"],
    "PE": ["pe", "pef"],
    "Well Name": ["well name", "well_name", "well"],
}


class InputError(ValueError):
    """Problem with the uploaded file, reported to the user as is."""


@dataclass(frozen=True)
class Model:
    settings: dict
    session: ort.InferenceSession

    @property
    def required(self) -> list[str]:
        return ["Depth"] + self.settings["logs"] + self.settings["other_inputs"]


def load_model() -> Model:
    settings = json.loads(SETTINGS_PATH.read_text())
    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    return Model(settings, session)


def read_csv(content: bytes) -> pd.DataFrame:
    try:
        data = pd.read_csv(io.BytesIO(content))
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as err:
        raise InputError(f"The file could not be read as CSV ({err.__class__.__name__}).") from err
    return data


def standardise_columns(data: pd.DataFrame, model: Model) -> pd.DataFrame:
    """Rename columns to the expected names, ignoring case and accepting a few aliases."""
    expected = model.required + ["Facies", "Well Name"]
    lookup = {name.lower(): name for name in expected}
    for name, others in ALIASES.items():
        lookup.update({other: name for other in others})
    renames = {}
    for col in data.columns:
        target = lookup.get(str(col).strip().lower())
        if target and target not in renames.values():
            renames[col] = target
    return data.rename(columns=renames)


def validate(data: pd.DataFrame, model: Model) -> pd.DataFrame:
    """Check the columns and values; return one well sorted by depth."""
    found = [str(c) for c in data.columns]
    data = standardise_columns(data, model)
    missing = [c for c in model.required if c not in data.columns]
    if missing:
        shown = ", ".join(found[:15]) + (" …" if len(found) > 15 else "")
        raise InputError(f"Missing column(s): {', '.join(missing)}. "
                         f"Required: {', '.join(model.required)}. Columns found in the file: {shown}.")
    if data.empty:
        raise InputError("The file has no data rows.")
    if len(data) > MAX_ROWS:
        raise InputError(f"The file has {len(data):,} rows; the limit is {MAX_ROWS:,}.")
    if "Well Name" in data.columns and data["Well Name"].nunique() > 1:
        raise InputError("The file contains several wells; upload one well per file.")

    out = data.copy()
    numeric = model.required + (["Facies"] if "Facies" in out.columns else [])
    for col in numeric:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    empty = out[model.required].isna().sum()
    empty = empty[empty > 0]
    if len(empty):
        detail = ", ".join(f"{col} ({n})" for col, n in empty.items())
        raise InputError(f"Empty or non-numeric values in: {detail}.")
    if not out["NM_M"].isin([1, 2]).all():
        raise InputError("NM_M must be 1 (non-marine) or 2 (marine).")
    if not out["RELPOS"].between(0, 1).all():
        raise InputError("RELPOS must be between 0 and 1.")
    if out["Depth"].duplicated().any():
        raise InputError("Depth values must be unique within the well.")
    if "Facies" in out.columns:
        codes = [int(k) for k in model.settings["facies"]]
        if out["Facies"].isna().any() or not out["Facies"].isin(codes).all():
            raise InputError(f"Facies, when given, must be integer codes {codes[0]} to {codes[-1]}.")
        out["Facies"] = out["Facies"].astype(int)
    return out.sort_values("Depth").reset_index(drop=True)


def build_features(well: pd.DataFrame, settings: dict) -> np.ndarray:
    """Same steps as the notebook: capping, depth context, standardisation."""
    logs = settings["logs"]
    data = well.copy()
    low, high = pd.Series(settings["cap_low"]), pd.Series(settings["cap_high"])
    data[logs] = data[logs].clip(low[logs], high[logs], axis=1)
    for log in logs:
        data[f"{log}_grad"] = data[log].diff().fillna(0.0)
        data[f"{log}_mean"] = data[log].rolling(settings["context_window"], center=True,
                                                min_periods=1).mean()
    features = settings["features"]
    mean = np.array([settings["feature_mean"][f] for f in features])
    std = np.array([settings["feature_std"][f] for f in features])
    return ((data[features].to_numpy(dtype=float) - mean) / std).astype("float32")


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean F1 over the facies present in the true or predicted column."""
    scores = []
    for label in np.union1d(y_true, y_pred):
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        scores.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(scores))


def predict(well: pd.DataFrame, model: Model) -> dict:
    """Predict the facies of a validated well and package the result for the page."""
    X = build_features(well, model.settings)
    session = model.session
    proba = session.run(None, {session.get_inputs()[0].name: X})[0]
    codes = np.array([int(k) for k in model.settings["facies"]])
    predicted = codes[proba.argmax(axis=1)]

    result = {
        "well": str(well["Well Name"].iloc[0]) if "Well Name" in well.columns else None,
        "samples": len(well),
        "depth": well["Depth"].round(2).tolist(),
        "logs": {log: well[log].round(4).tolist() for log in model.settings["logs"]},
        "predicted": predicted.tolist(),
        "confidence": proba.max(axis=1).round(4).tolist(),
        "probabilities": proba.round(4).tolist(),
        "facies": model.settings["facies"],
        "colors": model.settings["facies_colors"],
        "true": None,
        "scores": None,
    }
    if "Facies" in well.columns:
        truth = well["Facies"].to_numpy()
        result["true"] = truth.tolist()
        result["scores"] = {"accuracy": round(float((truth == predicted).mean()), 4),
                            "macro_f1": round(macro_f1(truth, predicted), 4)}
    return result


def demo_wells() -> dict[str, Path]:
    return {p.stem: p for p in sorted(DEMO_DIR.glob("*.csv"))}
