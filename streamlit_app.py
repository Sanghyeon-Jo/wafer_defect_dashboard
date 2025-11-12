"\"\"\"Streamlit dashboard for the wafer defect risk model.\"\"\""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from ml_pipeline import (
    ModelArtifacts,
    build_full_pipeline,
    compute_process_priority_scores,
    ensure_artifacts,
    load_artifacts,
    load_ontology,
)

ARTIFACT_DIR = Path("artifacts")


@st.cache_data(show_spinner=False)
def load_pipeline_outputs():
    """Run the preprocessing pipeline once and cache the results."""
    labelled_df, lot_df, prediction_dataset, priority_df = build_full_pipeline()
    return labelled_df, lot_df, prediction_dataset, priority_df


@st.cache_resource(show_spinner=False)
def load_ontology_data() -> Dict[str, Any]:
    """Load ontology metadata for process/zone/issue knowledge."""
    return load_ontology()


@st.cache_resource(show_spinner=False)
def load_trained_model() -> ModelArtifacts:
    """Load persisted model artifacts, training them if necessary."""
    try:
        return load_artifacts(ARTIFACT_DIR)
    except FileNotFoundError:
        artifacts, _, _ = ensure_artifacts(ARTIFACT_DIR)
        return artifacts


def _ensure_korean_font() -> None:
    """Register a font that supports Korean glyphs for Matplotlib."""
    if getattr(_ensure_korean_font, "_initialized", False):
        return

    font_candidates = [
        Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
        Path("/usr/share/fonts/truetype/nanum/NanumGothicCoding.ttf"),
        Path("/usr/share/fonts/truetype/nanum/NanumMyeongjo.ttf"),
        Path("C:/Windows/Fonts/NanumGothic.ttf"),
        Path("C:/Windows/Fonts/NanumSquareR.ttf"),
        Path("C:/Windows/Fonts/malgun.ttf"),
        Path("C:/Windows/Fonts/Malgun.ttf"),
        Path("C:/Windows/Fonts/malgunbd.ttf"),
    ]

    selected_font = None
    for font_path in font_candidates:
        if font_path.exists():
            try:
                fm.fontManager.addfont(str(font_path))
                selected_font = fm.FontProperties(fname=str(font_path)).get_name()
                break
            except Exception:
                continue

    if selected_font:
        plt.rcParams["font.family"] = selected_font
    else:
        plt.rcParams["font.family"] = [
            "NanumGothic",
            "Malgun Gothic",
            "AppleGothic",
            "DejaVu Sans",
        ]

    plt.rcParams["axes.unicode_minus"] = False
    _ensure_korean_font._initialized = True


_ensure_korean_font()


def _build_ontology_maps(ontology: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    processes = ontology.get("processes", [])
    zones = ontology.get("spatial_zones", [])
    issues = ontology.get("issue_types", [])

    process_by_id = {proc.get("id"): proc for proc in processes if proc.get("id")}
    process_by_name = {proc.get("name"): proc for proc in processes if proc.get("name")}

    zone_by_id = {zone.get("id"): zone for zone in zones if zone.get("id")}
    zone_by_name = {zone.get("name"): zone for zone in zones if zone.get("name")}

    issue_by_id = {issue.get("id"): issue for issue in issues if issue.get("id")}
    issue_by_name = {issue.get("name"): issue for issue in issues if issue.get("name")}

    return {
        "process_by_id": process_by_id,
        "process_by_name": process_by_name,
        "zone_by_id": zone_by_id,
        "zone_by_name": zone_by_name,
        "issue_by_id": issue_by_id,
        "issue_by_name": issue_by_name,
    }


def _lookup_ontology_entry(
    maps: Dict[str, Dict[str, Any]],
    *,
    entry_id: Optional[str],
    entry_name: Optional[str],
    entry_type: str,
) -> Optional[Dict[str, Any]]:
    if entry_id and entry_id in maps[f"{entry_type}_by_id"]:
        return maps[f"{entry_type}_by_id"][entry_id]
    if entry_name and entry_name in maps[f"{entry_type}_by_name"]:
        return maps[f"{entry_type}_by_name"][entry_name]
    return None


def render_summary(
    prediction_dataset: pd.DataFrame,
    warning_threshold: float,
    severity_threshold: float,
) -> None:
    total_lots = prediction_dataset["Lot Name"].nunique()
    warned_lots = (
        prediction_dataset["Predicted_Risk"] >= warning_threshold
    ).sum()
    severity_series = prediction_dataset.get("Severity_Score")
    severity_hot_lots = (
        (prediction_dataset["Predicted_Risk"] >= warning_threshold)
        & (severity_series >= severity_threshold)
    ).sum()
    avg_risk = prediction_dataset["Total_Risk_Score"].mean()
    avg_severity = prediction_dataset["Severity_Score"].mean()
    metric_cols = st.columns(5)
    col1, col2, col3, col4, col5 = metric_cols
    col1.metric("Lot 수", f"{total_lots:,}")
    col2.metric(
        "1차 경고 Lot 수",
        f"{warned_lots:,}",
        f"임계값 {warning_threshold:.2f}",
    )
    col3.metric(
        "2차 경고 Lot 수",
        f"{severity_hot_lots:,}",
        f"임계값 {severity_threshold:.2f}",
    )
    col4.metric(
        "평균 위험도",
        f"{avg_risk:.2f}",
    )
    col5.metric(
        "평균 심각도",
        f"{avg_severity:.2f}",
    )
    st.caption("임계값을 조정하면 경고 Lot 수와 평균 지표가 즉시 업데이트됩니다.")


def render_top_lots(
    top_df: pd.DataFrame,
    warning_threshold: float,
    severity_threshold: float,
) -> None:
    st.subheader("위험도 상위 Lot")
    if top_df.empty:
        st.info("상위 Lot 데이터가 없습니다.")
        return

    top_df = top_df.copy()
    top_df["Risk_Gap"] = top_df["Predicted_Risk"] - top_df["Total_Risk_Score"]
    sort_order = top_df.sort_values("Predicted_Risk", ascending=False)["Lot Name"].tolist()

    base = alt.Chart(top_df).encode(
        y=alt.Y("Lot Name:N", sort=sort_order, title="Lot Name"),
    )

    predicted_bars = base.mark_bar(color="#fb6a4a").encode(
        x=alt.X("Predicted_Risk:Q", title="위험도"),
        tooltip=[
            alt.Tooltip("Lot Name", title="Lot"),
            alt.Tooltip("Predicted_Risk", title="예측 위험도", format=".3f"),
            alt.Tooltip("Total_Risk_Score", title="실제 위험도", format=".3f"),
            alt.Tooltip("Risk_Gap", title="예측-실제 차이", format="+.3f"),
            alt.Tooltip("Severity_Score", title="심각도 점수", format=".3f"),
            alt.Tooltip("Killer_Defect_Count", title="킬러 결함 수"),
            alt.Tooltip("Total_Count", title="전체 결함 수"),
            alt.Tooltip("Killer_Defect_Count_per_slot", title="킬러/슬롯", format=".1f"),
            alt.Tooltip("Nuisance_Count_per_slot", title="일반/슬롯", format=".1f"),
            alt.Tooltip("False_Defect_Count_per_slot", title="거짓/슬롯", format=".1f"),
            alt.Tooltip(
                "Killer_Defect_Proportion",
                title="킬러 결함 비율",
                format=".1%",
            ),
        ],
        color=alt.condition(
            alt.datum.Risk_Gap > 0,
            alt.value("#fb6a4a"),
            alt.value("#9ecae1"),
        ),
    )

    actual_mark = base.mark_tick(color="#2171b5", thickness=2, size=30).encode(
        x=alt.X("Total_Risk_Score:Q"),
        tooltip=[
            alt.Tooltip("Lot Name", title="Lot"),
            alt.Tooltip("Total_Risk_Score", title="실제 위험도", format=".3f"),
        ],
    )

    gap_labels = base.mark_text(
        align="left",
        dx=6,
        color="#424242",
        fontSize=11,
    ).encode(
        x="Predicted_Risk:Q",
        text=alt.Text("Risk_Gap:Q", format="+.3f"),
    )

    chart = (predicted_bars + actual_mark + gap_labels).properties(height=400)
    st.altair_chart(chart, width="stretch")
    st.caption(
        "주황 막대=예측 위험도, 파란 표시=실제 위험도, 숫자=예측-실제 차이입니다. "
        "차이가 클수록 모델과 실제 간 격차가 크다는 뜻입니다."
    )


def render_risk_quadrant(
    prediction_dataset: pd.DataFrame,
    *,
    size_metric: str = "Killer_Defect_Proportion",
) -> None:
    if prediction_dataset.empty:
        st.info("데이터가 없습니다.")
        return

    mean_risk = prediction_dataset["Total_Risk_Score"].mean()
    mean_severity = prediction_dataset["Severity_Score"].mean()

    if size_metric not in prediction_dataset.columns:
        size_metric = "Killer_Defect_Proportion"

    size_titles = {
        "Killer_Defect_Proportion": "킬러 결함 비율",
        "Total_Count": "전체 결함 수",
        "Killer_Defect_Count": "킬러 결함 수",
        "Severity_Score": "심각도 점수",
    }
    size_title = size_titles.get(size_metric, size_metric)

    base_chart = (
        alt.Chart(prediction_dataset)
        .mark_circle()
        .encode(
            x=alt.X(
                "Total_Risk_Score:Q",
                title="위험도 (Total_Risk_Score)",
            ),
            y=alt.Y(
                "Severity_Score:Q",
                title="심각도 (Severity_Score)",
            ),
            size=alt.Size(
                f"{size_metric}:Q",
                title=size_title,
                scale=alt.Scale(
                    range=[60, 600] if size_metric != "Total_Count" else [80, 900]
                ),
                legend=None,
            ),
            color=alt.Color(
                "Predicted_Risk:Q",
                scale=alt.Scale(scheme="reds"),
                title="예측 위험도",
            ),
            tooltip=[
                alt.Tooltip("Lot Name", title="Lot"),
                alt.Tooltip("Predicted_Risk", title="예측 위험도", format=".3f"),
                alt.Tooltip("Total_Risk_Score", title="위험도", format=".3f"),
                alt.Tooltip("Severity_Score", title="심각도", format=".3f"),
                alt.Tooltip(size_metric, title=size_title, format=".3f"),
            ],
        )
        .properties(height=380)
    )

    mean_rules = (
        alt.Chart(pd.DataFrame({"x": [mean_risk], "y": [mean_severity]}))
        .mark_rule(strokeDash=[6, 6], color="gray")
        .encode(x="x:Q")
        + alt.Chart(pd.DataFrame({"x": [mean_risk], "y": [mean_severity]}))
        .mark_rule(strokeDash=[6, 6], color="gray")
        .encode(y="y:Q")
    )

    st.altair_chart((base_chart + mean_rules).interactive(), width="stretch")


def _render_process_warning_pie(
    prediction_df: pd.DataFrame,
    labelled_df: pd.DataFrame,
    *,
    lot_mask: pd.Series,
    title: str,
    caption: str,
    color_scheme: str = "category20c",
) -> None:
    warning_lots = prediction_df.loc[lot_mask, "Lot Name"]
    st.markdown(f"#### {title}")
    if warning_lots.empty:
        st.info("해당 경고 조건을 충족하는 Lot이 없습니다.")
        return

    step_df = (
        labelled_df[labelled_df["Lot Name"].isin(warning_lots)]
        .dropna(subset=["Step_desc"])
        .groupby("Step_desc")["Lot Name"]
        .nunique()
        .reset_index(name="Lot_Count")
    )
    if step_df.empty:
        st.info("공정 정보가 있는 경고 Lot이 없습니다.")
        return

    step_df["Percentage"] = step_df["Lot_Count"] / step_df["Lot_Count"].sum()
    chart = (
        alt.Chart(step_df)
        .mark_arc(innerRadius=60)
        .encode(
            theta=alt.Theta("Lot_Count:Q"),
            color=alt.Color(
                "Step_desc:N",
                legend=alt.Legend(title="공정"),
                scale=alt.Scale(scheme=color_scheme),
            ),
            tooltip=[
                alt.Tooltip("Step_desc", title="공정"),
                alt.Tooltip("Lot_Count", title="Lot 수", format="d"),
                alt.Tooltip("Percentage", title="비율", format=".1%"),
            ],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, width="stretch")
    st.caption(caption)


def render_process_warning_overview(
    prediction_df: pd.DataFrame,
    labelled_df: pd.DataFrame,
    *,
    warning_threshold: float,
    severity_threshold: float,
) -> None:
    severity_series = prediction_df.get("Severity_Score")
    if severity_series is None:
        severity_series = pd.Series(0, index=prediction_df.index, dtype=float)

    primary_mask = prediction_df["Predicted_Risk"] >= warning_threshold
    secondary_mask = primary_mask & (severity_series >= severity_threshold)

    if not primary_mask.any():
        st.markdown("#### 공정별 경고 Lot 분포")
        st.info("현재 설정된 임계값을 만족하는 1차 경고 Lot이 없습니다.")
        return

    if not secondary_mask.any():
        col_primary = st.container()
        with col_primary:
            _render_process_warning_pie(
                prediction_df,
                labelled_df,
                lot_mask=primary_mask,
                title="공정별 경고 Lot 분포 (1차)",
                caption="예측 위험도 임계값을 초과한 Lot을 공정 기준으로 집계했습니다.",
                color_scheme="reds",
            )
        return

    col_primary, col_secondary = st.columns(2, gap="large")
    with col_primary:
        _render_process_warning_pie(
            prediction_df,
            labelled_df,
            lot_mask=primary_mask,
            title="공정별 1차 경고 Lot",
            caption="예측 위험도 임계값을 초과한 Lot을 공정 기준으로 집계했습니다.",
            color_scheme="reds",
        )
    with col_secondary:
        _render_process_warning_pie(
            prediction_df,
            labelled_df,
            lot_mask=secondary_mask,
            title="공정별 2차 경고 Lot",
            caption="1차 경고 중에서 심각도 임계값까지 초과한 Lot입니다.",
            color_scheme="blues",
        )


def render_process_priority(
    priority_df: pd.DataFrame,
    ontology: Dict[str, Any],
    *,
    top_n: int = 12,
    hotspot_detail: Optional[pd.DataFrame] = None,
) -> None:
    st.markdown("#### 공정 문제 우선순위 (P-Score)")
    if priority_df.empty:
        st.info("우선순위 데이터를 계산할 수 없습니다.")
        return

    display_df = priority_df.head(top_n).copy()
    ontology_maps = _build_ontology_maps(ontology)
    problem_options = display_df["Problem_Item"].tolist()
    selected_problem = st.selectbox(
        "우선 해결할 문제 항목 선택",
        problem_options,
        index=0 if problem_options else None,
    )

    display_df["Is_Selected"] = np.where(
        display_df["Problem_Item"] == selected_problem, "선택", "기타"
    )
    display_df["선택"] = np.where(
        display_df["Problem_Item"] == selected_problem, "◎", ""
    )

    chart = (
        alt.Chart(display_df)
        .mark_bar(color="#74add1", stroke="#225ea8")
        .encode(
            x=alt.X("P_Score:Q", title="P-Score (우선순위 점수)"),
            y=alt.Y(
                "Problem_Item:N",
                sort=display_df.sort_values("P_Score", ascending=False)["Problem_Item"],
                title="문제 항목",
            ),
            color=alt.Color(
                "Is_Selected:N",
                scale=alt.Scale(domain=["기타", "선택"], range=["#74add1", "#fb6a4a"]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Final_Rank", title="순위"),
                alt.Tooltip("Process_Name", title="공정"),
                alt.Tooltip("Zone_Name", title="영역"),
                alt.Tooltip("IssueType_Name", title="이슈 유형"),
                alt.Tooltip("Problem_Item", title="항목"),
                alt.Tooltip("Real_Ratio", title="REAL 비율", format=".1%"),
                alt.Tooltip("Rank_Score", title="중요도(C)"),
                alt.Tooltip("P_Score", title="P-Score", format=".2f"),
                alt.Tooltip("Sample_Size", title="샘플 수"),
                alt.Tooltip("Category", title="분류"),
            ],
        )
        .properties(height=max(260, 24 * len(display_df)))
    )
    st.altair_chart(chart, width="stretch")

    if not selected_problem:
        return

    selected_row = display_df[display_df["Problem_Item"] == selected_problem].iloc[0]

    metrics_cols = st.columns(4)
    metrics_cols[0].metric("우선순위", f"{int(selected_row['Final_Rank'])}")
    metrics_cols[1].metric(
        "P-Score",
        f"{selected_row['P_Score']:.2f}",
        help="Real_Ratio × Rank_Score",
    )
    metrics_cols[2].metric(
        "REAL 비율",
        f"{selected_row['Real_Ratio']:.1%}",
    )
    metrics_cols[3].metric(
        "샘플 수",
        f"{int(selected_row['Sample_Size']):,}",
        help="해당 항목에 포함된 결함 개수",
    )

    process_info = _lookup_ontology_entry(
        ontology_maps,
        entry_id=selected_row.get("Process_Id"),
        entry_name=selected_row.get("Process_Name"),
        entry_type="process",
    )
    zone_info = _lookup_ontology_entry(
        ontology_maps,
        entry_id=selected_row.get("Zone_Id"),
        entry_name=selected_row.get("Zone_Name"),
        entry_type="zone",
    )
    issue_info = _lookup_ontology_entry(
        ontology_maps,
        entry_id=selected_row.get("IssueType_Id"),
        entry_name=selected_row.get("IssueType_Name"),
        entry_type="issue",
    )

    st.markdown("##### 온톨로지 권고 및 진단 포인트")
    description_lines: list[str] = []
    if issue_info:
        issue_name = issue_info.get("name", selected_row.get("IssueType_Name", ""))
        issue_description = issue_info.get("description")
        description_lines.append(f"- **이슈 유형:** {issue_name}")
        if issue_description:
            description_lines.append(f"  - {issue_description}")
    if zone_info:
        zone_name = zone_info.get("name", selected_row.get("Zone_Name", ""))
        zone_description = zone_info.get("description")
        description_lines.append(f"- **공간 영역:** {zone_name}")
        if zone_description:
            description_lines.append(f"  - {zone_description}")
        related_causes = zone_info.get("related_causes")
        if related_causes:
            description_lines.append("  - 가능한 원인: " + ", ".join(related_causes))
    if process_info:
        proc_name = process_info.get("name", selected_row.get("Process_Name", ""))
        proc_desc = process_info.get("description")
        description_lines.append(f"- **공정:** {proc_name}")
        if proc_desc:
            description_lines.append(f"  - {proc_desc}")

    if description_lines:
        st.markdown("\n".join(description_lines))
    else:
        st.info("해당 항목에 대한 추가 온톨로지 설명이 없습니다.")

    if issue_info and issue_info.get("recommended_initial_actions"):
        st.markdown("**추천 초기 조치:**")
        action_lines = "\n".join(
            f"- {action}" for action in issue_info["recommended_initial_actions"]
        )
        st.markdown(action_lines)

    if process_info and process_info.get("critical_parameters"):
        st.markdown("**관심 공정 파라미터:**")
        st.markdown(
            "\n".join(f"- {param}" for param in process_info["critical_parameters"])
        )

    if selected_row["Sample_Size"] < 50:
        st.warning(
            "샘플 수가 적은 항목입니다. 현장 데이터와 함께 추가 검증이 필요할 수 있습니다."
        )

    detail_candidates: Optional[pd.DataFrame] = None
    if hotspot_detail is not None and not hotspot_detail.empty:
        detail_candidates = hotspot_detail.copy()
        step_mask = detail_candidates["Step_desc"] == selected_row["Step_desc"]
        zone_id = selected_row.get("Zone_Id")
        zone_name = selected_row.get("Zone_Name")
        zone_mask = pd.Series(True, index=detail_candidates.index)
        if zone_id is not None and pd.notna(zone_id):
            zone_mask = detail_candidates["Zone_Id"] == zone_id
        elif zone_name is not None and pd.notna(zone_name):
            zone_mask = detail_candidates["Zone_Name"] == zone_name
        detail_candidates = detail_candidates[step_mask & zone_mask]
        if not detail_candidates.empty:
            detail_candidates = detail_candidates.sort_values(
                "P_Score", ascending=False
            )

    with st.expander("세부 Hotspot (1µm 단위)", expanded=False):
        if detail_candidates is None or detail_candidates.empty:
            st.info("선택한 공정에 대한 미세 Hotspot 데이터가 없습니다.")
        else:
            st.markdown(
                "선택 공정에서 반복 검출되는 미세 영역입니다. "
                "반경 구간이 좁을수록 특정 장비/패스에서의 오염 가능성이 높습니다."
            )
            st.dataframe(
                detail_candidates.head(30)[
                    [
                        "Problem_Item",
                        "Real_Ratio",
                        "Rank_Score",
                        "P_Score",
                        "Sample_Size",
                    ]
                ],
                hide_index=True,
                width="stretch",
            )

    st.dataframe(
        display_df[
            [
                "선택",
                "Final_Rank",
                "Process_Name",
                "Zone_Name",
                "IssueType_Name",
                "Problem_Item",
                "Real_Ratio",
                "Rank_Score",
                "P_Score",
                "Sample_Size",
                "Category",
            ]
        ],
        width="stretch",
        height=280,
        hide_index=True,
    )


def _prepare_wafer_map_data(lot_rows: pd.DataFrame) -> pd.DataFrame:
    data = lot_rows.copy()
    if data.empty or "RADIUS" not in data.columns or "ANGLE" not in data.columns:
        return data

    max_radius = data["RADIUS"].replace(0, np.nan).max()
    target_radius = 150000.0 if pd.notna(max_radius) and max_radius > 0 else 1.0
    if pd.notna(max_radius) and max_radius > 0:
        scale_factor = target_radius / max_radius
        data["radius_norm"] = (data["RADIUS"] * scale_factor) / target_radius
    else:
        data["radius_norm"] = data["RADIUS"] / target_radius
    data["theta"] = np.deg2rad(data["ANGLE"])
    data["x"] = data["radius_norm"] * np.cos(data["theta"])
    data["y"] = data["radius_norm"] * np.sin(data["theta"])

    def _categorize(row: pd.Series) -> str:
        if row.get("IS_DEFECT") == "FALSE":
            return "False Defect"
        if row.get("is_killer_defect", False):
            return "Killer Defect"
        return "Nuisance Defect"

    data["Defect_Category"] = data.apply(_categorize, axis=1)
    return data


def _render_wafer_map(
    lot_rows: pd.DataFrame,
    *,
    width: Optional[int] = None,
    height: int = 420,
) -> None:
    _ensure_korean_font()
    map_data = _prepare_wafer_map_data(lot_rows)
    if map_data.empty:
        st.info("웨이퍼맵을 생성할 데이터가 없습니다.")
        return

    colors = {
        "Killer Defect": "#d7191c",
        "Nuisance Defect": "#2b83ba",
        "False Defect": "#bdbdbd",
    }
    mapped_colors = map_data["Defect_Category"].map(colors).fillna("#999999")

    size_source = None
    for candidate in ["DEFECT_AREA", "SIZE_D", "SIZE_X"]:
        if candidate in map_data.columns:
            size_source = map_data[candidate].abs().replace(0, np.nan)
            break
    if size_source is None:
        sizes = np.full(len(map_data), 22.0)
    else:
        normalized = size_source / (size_source.max() + 1e-6)
        sizes = np.clip(24 + normalized * 96, 18, 110)

    distances = np.sqrt(map_data["x"] ** 2 + map_data["y"] ** 2)
    boundary_margin = np.clip(1.0 - distances, 0.05, 1.0)
    sizes = sizes * (boundary_margin ** 2)

    alpha_source = None
    for candidate in ["SNR_OFFSET_GL", "PATCHDEFECTSIGNAL"]:
        if candidate in map_data.columns:
            alpha_source = map_data[candidate]
            break
    if alpha_source is not None:
        alpha_norm = (alpha_source - alpha_source.min()) / (
            (alpha_source.max() - alpha_source.min()) + 1e-6
        )
        alphas = 0.3 + 0.7 * alpha_norm
    else:
        alphas = np.where(map_data["Defect_Category"] == "False Defect", 0.3, 0.8)

    figsize = (4.8, 4.8) if width is None else (width / 100, height / 100)
    fig, ax = plt.subplots(figsize=figsize, dpi=160)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8f9fb")

    scatter = ax.scatter(
        map_data["x"],
        map_data["y"],
        s=sizes,
        c=mapped_colors,
        alpha=np.clip(alphas, 0.1, 0.95),
        edgecolors="white",
        linewidths=0.3,
    )

    wafer_circle = plt.Circle((0, 0), 1.0, color="#757575", fill=False, linewidth=1.4)
    ax.add_patch(wafer_circle)

    zone_radii = [0.33, 0.66, 0.9]
    for radius in zone_radii:
        style = {"linestyle": "--", "linewidth": 0.6, "edgecolor": "#b0bec5"}
        if np.isclose(radius, 0.9):
            style.update({"linestyle": ":", "linewidth": 0.6})
        ring = plt.Circle((0, 0), radius, fill=False, **style)
        ax.add_patch(ring)

    radial_angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    for angle in radial_angles:
        ax.plot(
            [0, np.cos(angle)],
            [0, np.sin(angle)],
            color="#cfd8dc",
            linewidth=0.5,
            alpha=0.7,
        )

    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)

    ax.text(
        0,
        -1.08,
        "Inner / Middle / Outer 영역 기준선을 표시했습니다.",
        ha="center",
        va="top",
        fontsize=8.5,
        color="#607d8b",
    )

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=label,
            markerfacecolor=color,
            markersize=8,
        )
        for label, color in colors.items()
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(handles),
        title="결함 유형",
        frameon=False,
    )

    st.pyplot(fig, clear_figure=True)


def _pattern_summary(lot_rows: pd.DataFrame) -> pd.DataFrame:
    if lot_rows.empty:
        return pd.DataFrame()

    enriched = _prepare_wafer_map_data(lot_rows)
    grouped = (
        enriched.groupby(["Step_desc", "Class", "KMeans_Cluster", "Defect_Category"])
        .size()
        .reset_index(name="Count")
    )
    grouped["Proportion"] = grouped["Count"] / grouped["Count"].sum()
    grouped = grouped.sort_values(by="Count", ascending=False)
    return grouped


def render_lot_detail(
    prediction_dataset: pd.DataFrame,
    labelled_df: pd.DataFrame,
    warning_threshold: float,
    severity_threshold: float,
    selected_lot: str,
) -> None:
    st.subheader("Lot 상세")
    if selected_lot not in prediction_dataset["Lot Name"].values:
        st.info("선택한 Lot 데이터가 없습니다.")
        return

    lot_summary = prediction_dataset[
        prediction_dataset["Lot Name"] == selected_lot
    ].iloc[0]

    def _render_component_pie(
        data_map: dict[str, float],
        *,
        legend_title: str,
        color_scheme: str,
    ) -> None:
        filtered = [
            (label, float(value))
            for label, value in data_map.items()
            if pd.notna(value) and float(value) > 0
        ]
        total = sum(value for _, value in filtered)
        if total <= 0:
            st.info(f"{legend_title} 정보를 계산할 수 없습니다.")
            return
        pie_df = pd.DataFrame(filtered, columns=["Component", "Value"])
        pie_df["Percentage"] = pie_df["Value"] / total
        chart = (
            alt.Chart(pie_df)
            .mark_arc(innerRadius=40)
            .encode(
                theta=alt.Theta("Value:Q"),
                color=alt.Color(
                    "Component:N",
                    scale=alt.Scale(scheme=color_scheme),
                    legend=alt.Legend(title=legend_title),
                ),
                tooltip=[
                    alt.Tooltip("Component", title="구성 요소"),
                    alt.Tooltip("Value", title="가중치", format=".3f"),
                    alt.Tooltip("Percentage", title="비율", format=".1%"),
                ],
            )
            .properties(height=300)
        )
        st.altair_chart(chart, width="stretch")

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "예측 위험도",
        f"{lot_summary['Predicted_Risk']:.3f}",
        f"{lot_summary['Prediction_Error']:+.3f}",
    )
    col2.metric(
        "실제 위험도",
        f"{lot_summary['Total_Risk_Score']:.3f}",
    )
    col3.metric(
        "심각도 점수",
        f"{lot_summary.get('Severity_Score', 0):.3f}",
        f"{lot_summary['Killer_Defect_Proportion']:.1%}",
    )

    primary_warning = lot_summary["Predicted_Risk"] >= warning_threshold
    secondary_warning = lot_summary.get("Severity_Score", 0) >= severity_threshold
    warning_state: list[str] = []
    if primary_warning:
        warning_state.append("⚠️ 1차 경고 (위험도)")
    if secondary_warning:
        warning_state.append("🔁 2차 경고 (심각도)")
    if warning_state:
        st.warning(" · ".join(warning_state))
    else:
        st.success("경고 없음")

    severity_components = {
        col.replace("Severity_Component_", "").replace("_", " "): lot_summary[col]
        for col in lot_summary.index
        if str(col).startswith("Severity_Component_")
    }
    risk_component_weights = {
        "Score_Killer": ("킬러 결함 기여", 0.50),
        "Score_Nuisance": ("일반 결함 기여", 0.30),
        "Score_False": ("거짓 결함 기여", 0.20),
    }
    risk_components = {
        label: lot_summary.get(col, 0) * weight
        for col, (label, weight) in risk_component_weights.items()
    }

    chart_left, chart_right = st.columns(2)
    with chart_right:
        st.markdown("### Lot 심각도 구성")
        _render_component_pie(
            severity_components,
            legend_title="심각도 구성",
            color_scheme="blues",
        )
    with chart_left:
        st.markdown("### Lot 위험도 구성")
        _render_component_pie(
            risk_components,
            legend_title="위험도 구성",
            color_scheme="reds",
        )

    lot_rows = labelled_df[labelled_df["Lot Name"] == selected_lot].copy()

    with st.expander("경고 임계값 설정", expanded=False):
        st.write(
            f"현재 1차 경고 임계값: **{warning_threshold:.2f}**, "
            f"2차 경고 임계값: **{severity_threshold:.2f}**"
        )
        st.caption("사이드바에서 임계값을 조정할 수 있습니다.")

    st.markdown("### 결함 패턴 요약")
    pattern_df = _pattern_summary(lot_rows)
    if pattern_df.empty:
        st.info("패턴 요약을 계산할 데이터가 없습니다.")
    else:
        st.dataframe(
            pattern_df,
            width="stretch",
            height=320,
        )

    st.markdown("### 결함 상세 테이블")
    st.caption("필요 시 필터 후 CSV로 다운로드할 수 있습니다.")
    st.dataframe(lot_rows, width="stretch", height=420)

    csv = lot_rows.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="결함 데이터 CSV 다운로드",
        data=csv,
        file_name=f"{selected_lot}_defects.csv",
        mime="text/csv",
    )


def main() -> None:
    st.set_page_config(
        page_title="Wafer Defect Risk Dashboard",
        layout="wide",
    )
    st.title("Wafer Defect Risk Dashboard")
    labelled_df, lot_df, prediction_dataset, priority_df = load_pipeline_outputs()
    ontology = load_ontology_data()
    artifacts = load_trained_model()

    prediction_dataset = prediction_dataset.copy()
    prediction_dataset["Predicted_Risk"] = artifacts.model.predict(
        prediction_dataset[artifacts.feature_names]
    )
    prediction_dataset["Prediction_Error"] = (
        prediction_dataset["Predicted_Risk"] - prediction_dataset["Total_Risk_Score"]
    )

    st.sidebar.header("경고 · 필터")
    warning_threshold = st.sidebar.slider(
        "1차 경고 (예측 위험도)",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.05,
    )
    severity_threshold = st.sidebar.slider(
        "2차 경고 (심각도 점수)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
    )
    step_options = sorted(labelled_df["Step_desc"].dropna().unique().tolist())
    step_filter = st.sidebar.multiselect(
        "Step 필터",
        options=step_options,
        default=[],
    )

    filtered_prediction = prediction_dataset.copy()
    filtered_labelled = labelled_df.copy()
    filtered_priority = priority_df.copy()
    lot_mask = set(filtered_prediction["Lot Name"])
    if step_filter:
        step_lots = set(
            filtered_labelled[filtered_labelled["Step_desc"].isin(step_filter)][
                "Lot Name"
            ]
        )
        lot_mask = lot_mask.intersection(step_lots)
        filtered_labelled = filtered_labelled[
            filtered_labelled["Step_desc"].isin(step_filter)
        ]
        filtered_priority = filtered_priority[
            filtered_priority["Process_Id"].isin(step_filter)
        ]
    if step_filter:
        filtered_prediction = filtered_prediction[
            filtered_prediction["Lot Name"].isin(lot_mask)
        ]

    summary_priority = compute_process_priority_scores(
        filtered_labelled,
        include_hotspots=False,
    )
    hotspot_detail = compute_process_priority_scores(
        filtered_labelled,
        include_hotspots=True,
        window_size=1.0,
        min_window_samples=5,
    )
    hotspot_detail = hotspot_detail[hotspot_detail["Category"] == "Hotspot"]
    if not summary_priority.empty:
        filtered_priority = summary_priority

    render_summary(filtered_prediction, warning_threshold, severity_threshold)
    render_process_warning_overview(
        filtered_prediction,
        filtered_labelled,
        warning_threshold=warning_threshold,
        severity_threshold=severity_threshold,
    )

    tabs = st.tabs(["Lot 개요", "Lot 상세", "공정 온톨로지"])

    with tabs[0]:
        st.markdown("#### 위험도 상위 Lot")
        top_n = st.slider(
            "표시할 상위 Lot 수",
            min_value=5,
            max_value=30,
            value=10,
            step=5,
        )
        component_cols = [
            col
            for col in filtered_prediction.columns
            if col.startswith("Severity_Component_")
        ]
        per_slot_cols = [
            col
            for col in [
                "Killer_Defect_Count_per_slot",
                "Nuisance_Count_per_slot",
                "False_Defect_Count_per_slot",
            ]
            if col in filtered_prediction.columns
        ]
        base_cols = [
            "Lot Name",
            "Predicted_Risk",
            "Total_Risk_Score",
            "Severity_Score",
            "Killer_Defect_Count",
            "Total_Count",
            "Killer_Defect_Proportion",
            "Score_Killer",
            "Score_Nuisance",
            "Score_False",
        ]
        top_df = filtered_prediction.nlargest(
            top_n,
            "Predicted_Risk",
        )[base_cols + per_slot_cols + component_cols]
        render_top_lots(top_df, warning_threshold, severity_threshold)
        st.caption(
            "목록은 예측 위험도 순으로 정렬됩니다. 실제 값과 차이를 함께 확인한 뒤, 관심 Lot을 선택해 상세 정보를 확인하세요."
        )

        st.markdown("#### 위험도 vs 심각도")
        size_options = {
            "킬러 결함 비율": "Killer 결함 비율",
            "전체 결함 수": "Total_Count",
            "킬러 결함 수": "Killer_Defect_Count",
            "심각도 점수": "Severity_Score",
        }
        size_mapping = {
            "킬러 결함 비율": "Killer_Defect_Proportion",
            "전체 결함 수": "Total_Count",
            "킬러 결함 수": "Killer_Defect_Count",
            "심각도 점수": "Severity_Score",
        }
        selected_size_label = st.selectbox(
            "버블 크기 기준",
            list(size_options.keys()),
            index=0,
        )
        render_risk_quadrant(
            filtered_prediction,
            size_metric=size_mapping[selected_size_label],
        )
        st.caption(
            "X축=실제 위험도, Y축=심각도, 색상=예측 위험도, 버블 크기=선택한 기준입니다. "
            "크기 기준을 바꾸면 특정 공정 또는 Lot의 경향을 다른 관점에서 비교할 수 있습니다."
        )

    with tabs[1]:
        layout_left, layout_right = st.columns([3, 2], gap="large")
        with layout_left:
            lot_selection_df = filtered_prediction[
                ["Lot Name", "Predicted_Risk", "Total_Risk_Score", "Severity_Score"]
            ].copy()
            severity_series = filtered_prediction.get("Severity_Score")
            if severity_series is None:
                severity_series = pd.Series(0, index=filtered_prediction.index, dtype=float)
            primary_warning_mask = filtered_prediction["Predicted_Risk"] >= warning_threshold
            secondary_warning_mask = primary_warning_mask & (
                severity_series >= severity_threshold
            )
            lot_selection_df["Primary_Warning"] = primary_warning_mask.values
            lot_selection_df["Secondary_Warning"] = secondary_warning_mask.values

            warning_only_df = lot_selection_df[lot_selection_df["Primary_Warning"]].copy()
            if warning_only_df.empty:
                st.info("현재 경고 임계값을 만족하는 Lot이 없습니다. 전체 Lot 목록을 표시합니다.")
                warning_only_df = lot_selection_df.copy()
            lot_selection_df = warning_only_df

            if not lot_selection_df.empty:
                def _format_lot_label(row: pd.Series) -> str:
                    warning_badge = "2차 경고" if row["Secondary_Warning"] else "1차 경고"
                    return (
                        f"{row['Lot Name']} | 위험도 {row['Total_Risk_Score']:.2f} | "
                        f"심각도 {row['Severity_Score']:.2f} | {warning_badge}"
                    )

                lot_selection_df["Lot_Label"] = lot_selection_df.apply(
                    _format_lot_label,
                    axis=1,
                )
                lot_options = lot_selection_df.sort_values(
                    ["Secondary_Warning", "Predicted_Risk", "Severity_Score"],
                    ascending=[False, False, False],
                )
                lot_names = lot_options["Lot Name"].tolist()
                lot_labels = lot_options["Lot_Label"].tolist()
            else:
                lot_names = []
                lot_labels = []
            selected_lot = st.selectbox(
                "Lot 선택",
                lot_labels if lot_labels else lot_names,
                index=0 if lot_names else None,
            )
            if not selected_lot:
                st.info("표시할 Lot 데이터가 없습니다.")
                return
            selected_lot = (
                lot_names[lot_labels.index(selected_lot)]
                if lot_labels
                else selected_lot
            )
            render_lot_detail(
                filtered_prediction,
                filtered_labelled,
                warning_threshold,
                severity_threshold,
                selected_lot,
            )
        with layout_right:
            st.markdown("#### 웨이퍼맵")
            lot_rows = filtered_labelled[
                filtered_labelled["Lot Name"] == selected_lot
            ].copy()
            _render_wafer_map(lot_rows, width=380, height=420)

    with tabs[2]:
        st.markdown("### 온톨로지 기반 공정 우선순위")
        render_process_priority(
            filtered_priority,
            ontology,
            hotspot_detail=hotspot_detail,
        )


if __name__ == "__main__":
    main()


