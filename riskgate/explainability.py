from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import xgboost as xgb


def _strip_transform_prefix(name: str) -> str:
    return name.split("__", 1)[1] if "__" in name else name


def _map_to_base_feature(name: str, base_features: List[str]) -> str:
    raw = _strip_transform_prefix(name)

    for base in sorted(base_features, key=len, reverse=True):
        if raw == base or raw.startswith(base + "_"):
            return base
    return raw


def make_xgb_contrib_pack(pipeline, raw_df: pd.DataFrame) -> Dict:
    """
    Compute XGBoost native contribution values on the transformed matrix.

    Note:
    - contributions are on the model's raw score scale
    - grouped one-hot contributions are aggregated back to base features later
    """
    feature_builder = pipeline.named_steps["features"]
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    built_df = feature_builder.transform(raw_df)
    transformed = preprocessor.transform(built_df)

    transformed_feature_names = list(preprocessor.get_feature_names_out())
    base_feature_names = [
        _map_to_base_feature(name, list(built_df.columns))
        for name in transformed_feature_names
    ]

    booster = model.get_booster()
    dmatrix = xgb.DMatrix(transformed, feature_names=transformed_feature_names)

    contribs = booster.predict(dmatrix, pred_contribs=True)

    # Some versions / configs can return 3D for grouped outputs
    if getattr(contribs, "ndim", 2) == 3:
        contribs = contribs[:, 0, :]

    base_value = contribs[:, -1]
    values = contribs[:, :-1]

    return {
        "values": values,
        "base_value": base_value,
        "transformed_feature_names": transformed_feature_names,
        "base_feature_names": base_feature_names,
    }


def global_contrib_table(contrib_pack: Dict) -> pd.DataFrame:
    values = contrib_pack["values"]
    base_feature_names = contrib_pack["base_feature_names"]

    df = pd.DataFrame({
        "feature": base_feature_names,
        "mean_abs_contrib": np.abs(values).mean(axis=0),
    })

    out = (
        df.groupby("feature", as_index=False)["mean_abs_contrib"]
        .sum()
        .sort_values("mean_abs_contrib", ascending=False)
        .reset_index(drop=True)
    )
    return out


def local_contrib_table(contrib_pack: Dict, row_idx: int, top_n: int = 10) -> pd.DataFrame:
    values = contrib_pack["values"][row_idx]
    base_feature_names = contrib_pack["base_feature_names"]

    df = pd.DataFrame({
        "feature": base_feature_names,
        "contribution": values,
    })

    out = (
        df.groupby("feature", as_index=False)["contribution"]
        .sum()
        .assign(abs_contribution=lambda d: d["contribution"].abs())
        .assign(direction=lambda d: np.where(d["contribution"] >= 0, "increase risk", "decrease risk"))
        .sort_values("abs_contribution", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return out