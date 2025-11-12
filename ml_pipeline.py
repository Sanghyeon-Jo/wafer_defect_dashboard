"""Utilities for preparing wafer defect data and training the risk model.

The original exploratory notebook was exported as ``Wafet_Defect_ML.py`` and
contains every step inline.  For a production-style application (e.g. a
Streamlit dashboard) we expose the important pieces here so that they can be
reused without copy & pasting large code blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


DATA_PATH = Path(__file__).resolve().parent / "dataset.csv"
ONTOLOGY_PATH = Path(__file__).resolve().parent / "ontology.json"

SIZE_COLUMNS = ["SIZE_X", "SIZE_Y", "DEFECT_AREA"]

CLUSTER_FEATURES = [
    "ENERGY_PARAM",
    "MDAT_OFFSET",
    "RELATIVEMAGNITUDE",
    "PATCHDEFECTSIGNAL",
    "INTENSITY",
    "POLARITY",
    "MDAT_GL",
    "MDAT_NOISE",
    "PATCHNOISE",
    "SIZE_X",
    "SIZE_Y",
    "DEFECT_AREA",
    "SIZE_D",
    "RADIUS",
    "ANGLE",
    "ALIGNRATIO",
    "SPOTLIKENESS",
    "ACTIVERATIO",
]

NUMERICAL_FEATURES = [
    "SIZE_X",
    "SIZE_Y",
    "DEFECT_AREA",
    "SIZE_D",
    "INTENSITY",
    "POLARITY",
    "ENERGY_PARAM",
    "MDAT_OFFSET",
    "MDAT_GL",
    "MDAT_NOISE",
    "RADIUS",
    "ANGLE",
    "ALIGNRATIO",
    "SPOTLIKENESS",
    "PATCHNOISE",
    "RELATIVEMAGNITUDE",
    "ACTIVERATIO",
    "PATCHDEFECTSIGNAL",
    "SNR_OFFSET_GL",
    "SNR_INTENSITY_NOISE",
    "ASPECT_RATIO",
    "DENSITY_SIGNAL",
]

SEVERITY_FEATURES = [
    "ACTIVERATIO",
    "PATCHDEFECTSIGNAL",
]

SEVERITY_FEATURE_WEIGHTS = {
    "ACTIVERATIO": 0.5,
    "PATCHDEFECTSIGNAL": 0.5,
}

SEVERITY_DEFECT_WEIGHTS = {
    "killer": 0.7,
    "real": 0.3,
}

DEFAULT_STEPS_TO_ANALYSE = ["PC", "RMG", "CBCMP"]

KILLER_CLUSTER_MAPPING = {
    "PC": 1,
    "RMG": 1,
    "CBCMP": 0,
}

# ---------------------------------------------------------------------------
# Ontology loading helpers
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_ontology(path: Path = ONTOLOGY_PATH) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Ontology file not found: {path}")
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)



@dataclass
class ModelArtifacts:
    model: RandomForestRegressor
    feature_names: List[str]


def load_raw_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the wafer defect CSV."""
    return pd.read_csv(path)


def remove_outliers_by_class(
    df: pd.DataFrame, columns: Iterable[str] = SIZE_COLUMNS
) -> pd.DataFrame:
    """Remove high outliers (Q3 + 1.5 IQR) per Class for the given columns."""
    filtered_parts: List[pd.DataFrame] = []

    for _, group in df.groupby("Class"):
        filtered = group.copy()
        for col in columns:
            if col not in filtered or filtered[col].count() < 2:
                continue
            q1 = filtered[col].quantile(0.25)
            q3 = filtered[col].quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            upper = q3 + 1.5 * iqr
            filtered = filtered[filtered[col] <= upper]
        filtered_parts.append(filtered)

    return pd.concat(filtered_parts, ignore_index=True)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the engineered features used downstream."""
    out = df.copy()

    out["SNR_OFFSET_GL"] = out["MDAT_OFFSET"] / (out["MDAT_GL"] + 1e-6)
    out["SNR_INTENSITY_NOISE"] = out["INTENSITY"] / (out["PATCHNOISE"] + 1e-6)
    out["ASPECT_RATIO"] = out["SIZE_X"] / (out["SIZE_Y"] + 1e-6)
    out["ASPECT_RATIO"] = out["ASPECT_RATIO"].replace([np.inf, -np.inf], np.nan)
    out["DENSITY_SIGNAL"] = out["INTENSITY"] / (out["DEFECT_AREA"] + 1e-6)
    out["DENSITY_SIGNAL"] = out["DENSITY_SIGNAL"].replace([np.inf, -np.inf], np.nan)

    return out


def run_kmeans_by_step(
    df: pd.DataFrame,
    steps_to_analyse: Iterable[str] = DEFAULT_STEPS_TO_ANALYSE,
    features: Iterable[str] = CLUSTER_FEATURES,
    n_clusters: int = 2,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[StandardScaler, KMeans]]]:
    """Assign a KMeans cluster per process step.

    Returns a copy of the dataframe with ``KMeans_Cluster`` (nullable Int64)
    plus a mapping of fitted (scaler, kmeans) per step.
    """
    df_with_clusters = df.copy()
    df_with_clusters["KMeans_Cluster"] = pd.Series(pd.NA, index=df.index, dtype="Int64")

    fitted_models: Dict[str, Tuple[StandardScaler, KMeans]] = {}

    for step in steps_to_analyse:
        mask = (df_with_clusters["IS_DEFECT"] == "REAL") & (
            df_with_clusters["Step_desc"] == step
        )
        step_data = df_with_clusters.loc[mask, list(features)]
        step_data = step_data.dropna()
        if step_data.empty:
            continue

        scaler = StandardScaler()
        scaled = scaler.fit_transform(step_data)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(scaled)

        df_with_clusters.loc[step_data.index, "KMeans_Cluster"] = labels
        fitted_models[step] = (scaler, kmeans)

    df_with_clusters["KMeans_Cluster"] = df_with_clusters["KMeans_Cluster"].astype(
        "Int64"
    )
    return df_with_clusters, fitted_models


def label_killer_defects(
    df: pd.DataFrame,
    mapping: Dict[str, int] = KILLER_CLUSTER_MAPPING,
) -> pd.DataFrame:
    """Flag rows that belong to killer clusters for each process step."""
    out = df.copy()
    out["is_killer_defect"] = False

    for step, killer_cluster in mapping.items():
        mask = (
            (out["IS_DEFECT"] == "REAL")
            & (out["Step_desc"] == step)
            & (out["KMeans_Cluster"] == killer_cluster)
        )
        out.loc[mask, "is_killer_defect"] = True
    return out


def _min_max_normalize(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    min_val = series.min()
    max_val = series.max()
    if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
        return pd.Series(0.0, index=series.index)
    return (series - min_val) / (max_val - min_val)


def compute_severity_scores(
    df: pd.DataFrame,
    feature_weights: Dict[str, float] = SEVERITY_FEATURE_WEIGHTS,
    defect_weights: Dict[str, float] = SEVERITY_DEFECT_WEIGHTS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute defect-level severity and aggregate to lot-level scores."""
    available_features = [feat for feat in SEVERITY_FEATURES if feat in df.columns]
    if not available_features:
        lot_names = df["Lot Name"].unique()
        empty_df = pd.DataFrame({"Lot Name": lot_names, "Severity_Score": 0.0})
        return df.assign(Defect_Severity=0.0), empty_df

    df_local = df.copy()

    group_keys = ["Step_desc", "Class"]
    for feat in available_features:
        norm_col = f"{feat}_norm"

        def _group_norm(series: pd.Series) -> pd.Series:
            min_val = series.min()
            max_val = series.max()
            if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
                return pd.Series(0.0, index=series.index)
            return (series - min_val) / (max_val - min_val)

        df_local[norm_col] = (
            df_local.groupby(group_keys)[feat]
            .transform(_group_norm)
            .fillna(0.0)
        )

    defect_weight = np.where(
        df_local["is_killer_defect"],
        defect_weights["killer"],
        np.where(df_local["IS_DEFECT"] == "REAL", defect_weights["real"], 0.0),
    )

    component_columns = []
    weighted_sum = np.zeros(len(df_local), dtype=float)
    for feat in available_features:
        comp_col = f"Severity_Component_{feat}"
        weighted = feature_weights.get(feat, 0.0) * df_local[f"{feat}_norm"]
        df_local[comp_col] = defect_weight * weighted
        component_columns.append(comp_col)
        weighted_sum += df_local[comp_col].to_numpy()

    df_local["Defect_Severity"] = weighted_sum

    lot_aggregated = (
        df_local.groupby("Lot Name")
        .agg(
            Severity_Score=("Defect_Severity", "mean"),
            **{col: (col, "mean") for col in component_columns},
        )
        .reset_index()
    )

    return df_local, lot_aggregated


def compute_process_priority_scores(
    df: pd.DataFrame,
    *,
    inner_threshold: float = 0.33,
    middle_threshold: float = 0.66,
    include_hotspots: bool = False,
    window_size: float = 300.0,
    min_window_samples: int = 40,
) -> pd.DataFrame:
    """Calculate process priority scores based on REAL defect distribution.

    When include_hotspots=False (default) the result focuses on Inner/Middle/Outer
    영역 기반 요약을 제공하고, True로 설정하면 지정한 window_size로 세분화된
    Hotspot 행이 추가되어 상세 분석에 활용할 수 있습니다.
    """

    columns = [
        "Final_Rank",
        "Step_desc",
        "Process_Id",
        "Process_Name",
        "Zone_Id",
        "Zone_Name",
        "IssueType_Id",
        "IssueType_Name",
        "Problem_Item",
        "Real_Ratio",
        "Rank_Score",
        "P_Score",
        "Sample_Size",
        "Category",
    ]
    if df.empty or "RADIUS" not in df.columns:
        return pd.DataFrame(columns=columns)

    df_local = df.copy()

    ontology = load_ontology()
    process_lookup = {proc["id"]: proc for proc in ontology.get("processes", [])}
    zone_lookup = {zone["id"]: zone for zone in ontology.get("spatial_zones", [])}
    zone_name_to_id = {
        zone["name"]: zone_id for zone_id, zone in zone_lookup.items()
    }
    issue_lookup = {it["id"]: it for it in ontology.get("issue_types", [])}
    zones_sorted = sorted(
        zone_lookup.values(), key=lambda z: z.get("radius_norm_min", 0.0)
    )

    def _get_process_info(step_desc: str) -> Tuple[str, str]:
        if step_desc in process_lookup:
            info = process_lookup[step_desc]
            return info["id"], info.get("name", step_desc)
        return step_desc, step_desc

    def _resolve_zone(zone_label: str) -> Tuple[str, str]:
        zone_id = zone_name_to_id.get(zone_label)
        if not zone_id:
            zone_id = zone_label.lower().replace(" ", "_")
        zone_info = zone_lookup.get(zone_id)
        zone_name = zone_info.get("name", zone_label) if zone_info else zone_label
        return zone_id, zone_name

    process_issue_map = {
        "CBCMP": "cmp_process_issue",
        "RMG": "deposition_issue",
        "PC": "etch_litho_issue",
    }

    def _resolve_issue_type(
        step_desc: str,
        zone_id: Optional[str],
        category: str,
    ) -> str:
        if category == "Hotspot":
            return "equipment_issue"
        if zone_id == "outer":
            return "wafer_handling_issue"
        return process_issue_map.get(step_desc, "cleaning_oxidation_issue")

    def _issue_rank(issue_id: str) -> int:
        return issue_lookup.get(issue_id, {}).get("rank_score_default", 3)

    def _zone_from_norm(norm_value: float) -> Tuple[str, str]:
        for zone in zones_sorted:
            min_norm = zone.get("radius_norm_min", 0.0)
            max_norm = zone.get("radius_norm_max", 1.0)
            if norm_value >= min_norm and norm_value < max_norm:
                return zone["id"], zone.get("name", zone["id"])
        if zones_sorted:
            last_zone = zones_sorted[-1]
            return last_zone["id"], last_zone.get("name", last_zone["id"])
        return "unknown", "Unknown"

    target_radius = 150000.0
    df_local["Radius_Norm"] = (
        df_local["RADIUS"].astype(float) / target_radius
    ).clip(lower=0.0, upper=1.0)
    df_local["Is_Real"] = (df_local["IS_DEFECT"] == "REAL").astype(int)

    bins = [0.0, inner_threshold, middle_threshold, float("inf")]
    area_labels = ["Inner Area", "Middle Area", "Outer Area"]
    df_local["Process_Area"] = pd.cut(
        df_local["Radius_Norm"],
        bins=bins,
        labels=area_labels,
        include_lowest=True,
    )

    total_area = df_local.groupby(
        ["Step_desc", "Process_Area"], observed=False
    ).size()
    real_area = (
        df_local[df_local["IS_DEFECT"] == "REAL"]
        .groupby(["Step_desc", "Process_Area"], observed=False)
        .size()
    )
    area_stats = (
        pd.DataFrame(
            {
                "Real_Ratio": real_area.div(total_area).fillna(0.0),
                "Sample_Size": total_area,
            }
        )
        .reset_index()
        .dropna(subset=["Process_Area"])
    )
    area_stats["Process_Area_Label"] = area_stats["Process_Area"].astype(str)
    if not area_stats.empty:
        zone_pairs = list(area_stats["Process_Area_Label"].map(_resolve_zone))
        area_stats["Zone_Id"], area_stats["Zone_Name"] = zip(*zone_pairs)
    else:
        area_stats["Zone_Id"] = []
        area_stats["Zone_Name"] = []
    area_stats["IssueType_Id"] = area_stats.apply(
        lambda row: _resolve_issue_type(
            row["Step_desc"], row.get("Zone_Id"), "Area"
        ),
        axis=1,
    )
    area_stats["IssueType_Name"] = area_stats["IssueType_Id"].map(
        lambda iid: issue_lookup.get(iid, {}).get("name", iid)
    )
    area_stats["Rank_Score"] = area_stats["IssueType_Id"].map(_issue_rank)
    if not area_stats.empty:
        process_pairs_area = list(area_stats["Step_desc"].map(_get_process_info))
        area_stats["Process_Id"], area_stats["Process_Name"] = zip(*process_pairs_area)
    else:
        area_stats["Process_Id"] = []
        area_stats["Process_Name"] = []
    area_stats["Problem_Item"] = area_stats.apply(
        lambda row: f"{row['Step_desc']} {row['Process_Area_Label']}", axis=1
    )
    area_stats["Category"] = "Area"

    priority_frames = [
        area_stats[
            [
                "Step_desc",
                "Process_Id",
                "Process_Name",
                "Zone_Id",
                "Zone_Name",
                "IssueType_Id",
                "IssueType_Name",
                "Problem_Item",
                "Real_Ratio",
                "Rank_Score",
                "Sample_Size",
                "Category",
            ]
        ]
    ]

    if include_hotspots and window_size and window_size > 0:
        df_local["Radius_Window_Start"] = (
            np.floor(df_local["RADIUS"] / window_size) * window_size
        )
        df_local["Radius_Window_End"] = df_local["Radius_Window_Start"] + window_size

        window_group_keys = [
            "Step_desc",
            "Radius_Window_Start",
            "Radius_Window_End",
        ]
        window_aggs = (
            df_local.groupby(window_group_keys, observed=False)
            .agg(
                Sample_Size=("Is_Real", "size"),
                Real_Count=("Is_Real", "sum"),
                Radius_Norm_Mean=("Radius_Norm", "mean"),
            )
            .reset_index()
        )
        window_aggs = window_aggs[window_aggs["Sample_Size"] >= min_window_samples]
        if not window_aggs.empty:
            window_aggs["Real_Ratio"] = window_aggs["Real_Count"] / (
                window_aggs["Sample_Size"] + 1e-9
            )

            start_vals = window_aggs["Radius_Window_Start"].fillna(0).round().astype(int)
            end_vals = window_aggs["Radius_Window_End"].fillna(0).round().astype(int)
            window_aggs["Problem_Item"] = (
                window_aggs["Step_desc"].astype(str)
                + " 웨이퍼 내 "
                + start_vals.map(lambda v: f"{v:,}")
                + "~"
                + end_vals.map(lambda v: f"{v:,}")
                + " 영역"
            )

            zone_pairs = list(window_aggs["Radius_Norm_Mean"].map(_zone_from_norm))
            process_pairs = list(window_aggs["Step_desc"].map(_get_process_info))
            window_aggs["Zone_Id"], window_aggs["Zone_Name"] = zip(*zone_pairs)
            window_aggs["Process_Id"], window_aggs["Process_Name"] = zip(*process_pairs)

            window_aggs["IssueType_Id"] = window_aggs.apply(
                lambda row: _resolve_issue_type(
                    row["Step_desc"], row.get("Zone_Id"), "Hotspot"
                ),
                axis=1,
            )
            window_aggs["IssueType_Name"] = window_aggs["IssueType_Id"].map(
                lambda iid: issue_lookup.get(iid, {}).get("name", iid)
            )
            window_aggs["Rank_Score"] = window_aggs["IssueType_Id"].map(_issue_rank)
            window_aggs["Sample_Size"] = window_aggs["Sample_Size"].astype(int)
            window_aggs["Category"] = "Hotspot"

            priority_frames.append(
                window_aggs[
                    [
                        "Step_desc",
                        "Process_Id",
                        "Process_Name",
                        "Zone_Id",
                        "Zone_Name",
                        "IssueType_Id",
                        "IssueType_Name",
                        "Problem_Item",
                        "Real_Ratio",
                        "Rank_Score",
                        "Sample_Size",
                        "Category",
                    ]
                ]
            )

    priority_df = pd.concat(priority_frames, ignore_index=True, sort=False)
    if priority_df.empty:
        return pd.DataFrame(columns=columns)

    priority_df["P_Score"] = priority_df["Real_Ratio"] * priority_df["Rank_Score"]
    priority_df = priority_df.sort_values("P_Score", ascending=False).reset_index(
        drop=True
    )
    priority_df["Final_Rank"] = priority_df.index + 1

    return priority_df[columns]


def killer_class_distribution(
    df: pd.DataFrame,
    *,
    killer_classes: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    """Return lot-level killer counts and proportions per class."""
    killer_df = df[(df["IS_DEFECT"] == "REAL") & (df["is_killer_defect"])]
    if killer_classes is not None:
        killer_df = killer_df[killer_df["Class"].isin(killer_classes)]

    if killer_df.empty:
        return pd.DataFrame(
            columns=["Class", "Lot Name", "Killer_Count", "Class_Total", "Killer_Defect_Proportion"]
        )

    grouped = (
        killer_df.groupby(["Class", "Lot Name"])
        .size()
        .reset_index(name="Killer_Count")
    )
    grouped["Class_Total"] = grouped.groupby("Class")["Killer_Count"].transform("sum")
    grouped["Killer_Defect_Proportion"] = grouped["Killer_Count"] / grouped["Class_Total"]
    return grouped


def compute_lot_level_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the aggregated lot-level metrics and Total_Risk_Score."""
    lot_analysis = (
        df[df["IS_DEFECT"] == "REAL"]
        .groupby("Lot Name")
        .size()
        .reset_index(name="Total_Count")
    )

    killer_counts = (
        df[df["is_killer_defect"]]
        .groupby("Lot Name")
        .size()
        .reset_index(name="Killer_Defect_Count")
    )
    lot_analysis = lot_analysis.merge(killer_counts, on="Lot Name", how="left")
    lot_analysis["Killer_Defect_Count"] = lot_analysis["Killer_Defect_Count"].fillna(0)

    false_counts = (
        df[df["IS_DEFECT"] == "FALSE"]
        .groupby("Lot Name")
        .size()
        .reset_index(name="False_Defect_Count")
    )
    lot_analysis = lot_analysis.merge(false_counts, on="Lot Name", how="left")
    lot_analysis["False_Defect_Count"] = lot_analysis["False_Defect_Count"].fillna(0)

    slot_counts = (
        df.groupby("Lot Name")["Slot No"]
        .nunique()
        .reset_index(name="Slot_No_nunique")
    )
    lot_analysis = lot_analysis.merge(slot_counts, on="Lot Name", how="left")
    lot_analysis["Slot_No_nunique"] = lot_analysis["Slot_No_nunique"].fillna(1)

    lot_analysis["Nuisance_Count"] = (
        lot_analysis["Total_Count"] - lot_analysis["Killer_Defect_Count"]
    )

    lot_analysis["Killer_Defect_Count_per_slot"] = lot_analysis[
        "Killer_Defect_Count"
    ] / (lot_analysis["Slot_No_nunique"] + 1e-6)
    lot_analysis["Nuisance_Count_per_slot"] = lot_analysis["Nuisance_Count"] / (
        lot_analysis["Slot_No_nunique"] + 1e-6
    )
    lot_analysis["False_Defect_Count_per_slot"] = lot_analysis[
        "False_Defect_Count"
    ] / (lot_analysis["Slot_No_nunique"] + 1e-6)

    scaler_killer = MinMaxScaler()
    scaler_nuisance = MinMaxScaler()
    scaler_false = MinMaxScaler()

    lot_analysis["Score_Killer"] = scaler_killer.fit_transform(
        lot_analysis[["Killer_Defect_Count_per_slot"]]
    )
    lot_analysis["Score_Nuisance"] = scaler_nuisance.fit_transform(
        lot_analysis[["Nuisance_Count_per_slot"]]
    )
    lot_analysis["Score_False"] = scaler_false.fit_transform(
        lot_analysis[["False_Defect_Count_per_slot"]]
    )

    w_killer, w_nuisance, w_false = 0.50, 0.30, 0.20
    lot_analysis["Total_Risk_Score"] = (
        w_killer * lot_analysis["Score_Killer"]
        + w_nuisance * lot_analysis["Score_Nuisance"]
        + w_false * lot_analysis["Score_False"]
    )
    lot_analysis["Killer_Defect_Proportion"] = (
        lot_analysis["Killer_Defect_Count"] / lot_analysis["Total_Count"]
    )

    return lot_analysis


def _build_step_class_cluster_features(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["Lot Name", "Step_desc", "Class", "KMeans_Cluster"])
        .size()
        .unstack(
            level=["Step_desc", "Class", "KMeans_Cluster"],
            fill_value=0,
        )
    )
    grouped.columns = [
        f"Count_Step_{step}_Class_{cls}_Cluster_{cluster}"
        for step, cls, cluster in grouped.columns
    ]
    return grouped


def _build_step_class_cluster_props(
    df: pd.DataFrame, totals: pd.Series
) -> pd.DataFrame:
    grouped = (
        df.groupby(["Lot Name", "Step_desc", "Class", "KMeans_Cluster"])
        .size()
        .unstack(
            level=["Step_desc", "Class", "KMeans_Cluster"],
            fill_value=0,
        )
    )
    grouped = grouped.div(totals, axis=0)
    grouped.columns = [
        f"Prop_Step_{step}_Class_{cls}_Cluster_{cluster}"
        for step, cls, cluster in grouped.columns
    ]
    return grouped


def _killer_stats(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    means = (
        df.groupby(["Lot Name", "is_killer_defect"])[NUMERICAL_FEATURES]
        .mean()
        .unstack(fill_value=0)
    )
    means.columns = [
        f"{feature}_{'killer' if killer else 'non_killer'}_mean"
        for feature, killer in means.columns
    ]

    medians = (
        df.groupby(["Lot Name", "is_killer_defect"])[NUMERICAL_FEATURES]
        .median()
        .unstack(fill_value=0)
    )
    medians.columns = [
        f"{feature}_{'killer' if killer else 'non_killer'}_median"
        for feature, killer in medians.columns
    ]

    return means, medians


def build_prediction_dataset(df: pd.DataFrame, lot_df: pd.DataFrame) -> pd.DataFrame:
    """Create the final lot-level dataset with engineered features."""
    total_per_lot = df.groupby("Lot Name").size()
    counts = _build_step_class_cluster_features(df)
    props = _build_step_class_cluster_props(df, total_per_lot)
    means, medians = _killer_stats(df)

    aggregated_numerical = (
        df.groupby("Lot Name")[NUMERICAL_FEATURES].agg(
            ["mean", "std", "min", "max", "median"]
        )
    )
    aggregated_numerical.columns = [
        "_".join(col).strip() for col in aggregated_numerical.columns.to_flat_index()
    ]

    categorical_unique = df.groupby("Lot Name").agg(
        Step_desc_nunique=("Step_desc", "nunique"),
        Class_nunique=("Class", "nunique"),
        Slot_No_nunique=("Slot No", "nunique"),
    )
    categorical_unique = categorical_unique.rename(
        columns={
            "Step_desc_nunique": "Distinct_Step_desc",
            "Class_nunique": "Distinct_Class",
            "Slot_No_nunique": "Distinct_Slot_No",
        }
    )

    dataset = (
        lot_df.set_index("Lot Name")
        .join([counts, props, means, medians, aggregated_numerical, categorical_unique])
        .reset_index()
    )
    numeric_columns = dataset.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        dataset[numeric_columns] = dataset[numeric_columns].fillna(0)
    bool_columns = dataset.select_dtypes(include=["bool"]).columns
    for col in bool_columns:
        dataset[col] = dataset[col].fillna(False).astype(bool)

    return dataset


def prepare_regression_xy(
    dataset: pd.DataFrame, target_column: str = "Total_Risk_Score"
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Split predictors and target, returning feature names for persistence."""
    feature_columns = [
        col for col in dataset.columns if col not in ("Lot Name", target_column)
    ]
    X = dataset[feature_columns]
    y = dataset[target_column]
    return X, y, feature_columns


def train_random_forest_model(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 100,
) -> Tuple[RandomForestRegressor, Dict[str, float]]:
    """Train the RandomForestRegressor and report metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    metrics = {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_test, y_pred)),
    }
    return model, metrics


def save_artifacts(
    artifacts: ModelArtifacts,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts.model, output_dir / "model.joblib")
    joblib.dump(artifacts.feature_names, output_dir / "feature_names.joblib")


def load_artifacts(path: Path) -> ModelArtifacts:
    model = joblib.load(path / "model.joblib")
    feature_names = joblib.load(path / "feature_names.joblib")
    return ModelArtifacts(model=model, feature_names=feature_names)


def build_full_pipeline(
    data_path: Path = DATA_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience helper that executes the full preprocessing pipeline.

    Returns:
        labelled_df: row-level dataframe with clusters, killer flag, severity
        lot_df: lot-level aggregated metrics (including Total_Risk_Score)
        prediction_dataset: final dataset used for regression
        priority_df: process priority scores for dashboard summaries
    """
    raw = load_raw_data(data_path)
    cleaned = remove_outliers_by_class(raw)
    engineered = add_engineered_features(cleaned)
    clustered, _ = run_kmeans_by_step(engineered)
    labelled = label_killer_defects(clustered)
    labelled_with_severity, severity_df = compute_severity_scores(labelled)
    priority_df = compute_process_priority_scores(labelled_with_severity)
    lot_df = compute_lot_level_risk(labelled_with_severity)
    lot_df = lot_df.merge(severity_df, on="Lot Name", how="left")
    prediction_dataset = build_prediction_dataset(labelled_with_severity, lot_df)
    return labelled_with_severity, lot_df, prediction_dataset, priority_df


def ensure_artifacts(
    output_dir: Path,
    data_path: Path = DATA_PATH,
) -> Tuple[ModelArtifacts, Dict[str, float], pd.DataFrame]:
    """Train and persist the model if artifacts are missing."""
    if (output_dir / "model.joblib").exists() and (
        output_dir / "feature_names.joblib"
    ).exists():
        artifacts = load_artifacts(output_dir)
        _, _, prediction_dataset, _ = build_full_pipeline(data_path)
        return artifacts, {}, prediction_dataset

    cleaned, lot_df, prediction_dataset, _ = build_full_pipeline(data_path)
    X, y, feature_names = prepare_regression_xy(prediction_dataset)
    model, metrics = train_random_forest_model(X, y)
    artifacts = ModelArtifacts(model=model, feature_names=feature_names)
    save_artifacts(artifacts, output_dir)
    return artifacts, metrics, prediction_dataset


