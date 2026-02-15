"""
===================================================================
HTML総合レポート生成
===================================================================
02/03の分析結果(JSON)と既存PNGを読み込み、
自己完結型HTMLレポートを生成する。

- Jinja2テンプレート (Python文字列)
- matplotlib → base64 PNG変換 (CONSORT図, 視聴パターン)
- 既存PNG → base64読み込み
- @media print CSSで印刷/PDF最適化
===================================================================
"""

import base64
import io
import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
from jinja2 import Template

warnings.filterwarnings("ignore")

for _font in ["Yu Gothic", "MS Gothic", "Meiryo", "Hiragino Sans", "IPAexGothic"]:
    try:
        matplotlib.rcParams["font.family"] = _font
        break
    except Exception:
        pass
matplotlib.rcParams["axes.unicode_minus"] = False

# === データファイル・カラム設定 (02/03と同一) ===
ENT_PRODUCT_CODE = "00001"
CONTENT_TYPES = ["Webinar", "e-contents", "web講演会"]
ACTIVITY_CHANNEL_FILTER = "web講演会"

FILE_RW_LIST = "rw_list.csv"
FILE_SALES = "sales.csv"
FILE_DIGITAL = "デジタル視聴データ.csv"
FILE_ACTIVITY = "活動データ.csv"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
_required = [FILE_SALES, FILE_DIGITAL, FILE_ACTIVITY, FILE_RW_LIST]
_data_ok = all(os.path.exists(os.path.join(DATA_DIR, f)) for f in _required)
if not _data_ok:
    _alt = os.path.join(SCRIPT_DIR, "data2")
    if all(os.path.exists(os.path.join(_alt, f)) for f in _required):
        DATA_DIR = _alt

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
REPORTS_DIR = os.path.join(SCRIPT_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

START_DATE = "2023-04-01"
N_MONTHS = 33
WASHOUT_MONTHS = 2
LAST_ELIGIBLE_MONTH = 29


# ================================================================
# ユーティリティ関数
# ================================================================

def fig_to_base64(fig, dpi=120):
    """matplotlib figureをbase64 PNG文字列に変換"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close(fig)
    return b64


def png_to_base64(filepath):
    """既存PNGファイルをbase64文字列に変換"""
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ================================================================
# データ読み込み (本番形式)
# ================================================================
print("=" * 60)
print(" HTMLレポート生成")
print("=" * 60)

print("\n[データ読み込み]")

# 1. RW医師リスト
rw_list = pd.read_csv(os.path.join(DATA_DIR, FILE_RW_LIST))
n_rw_all = len(rw_list)
doctor_master = rw_list[rw_list["seg"].notna() & (rw_list["seg"] != "")].copy()
doctor_master = doctor_master.rename(columns={"fac_honin": "facility_id", "doc": "doctor_id"})

# 2. 売上データ (日付・実績・品目コードが文字列)
sales_raw = pd.read_csv(os.path.join(DATA_DIR, FILE_SALES), dtype=str)
sales_raw["実績"] = pd.to_numeric(sales_raw["実績"], errors="coerce").fillna(0)
sales_raw["日付"] = pd.to_datetime(sales_raw["日付"], format="mixed")
n_sales_all = len(sales_raw)
daily = sales_raw[sales_raw["品目コード"].str.strip() == ENT_PRODUCT_CODE].copy()
daily = daily.rename(columns={
    "日付": "delivery_date",
    "施設（本院に合算）コード": "facility_id",
    "実績": "amount",
})

# 3. デジタル視聴データ
digital = pd.read_csv(os.path.join(DATA_DIR, FILE_DIGITAL))
n_digital_all = len(digital)
digital["品目コード"] = digital["品目コード"].astype(str).str.strip().str.zfill(5)
digital = digital[digital["品目コード"] == ENT_PRODUCT_CODE].copy()

# 4. 活動データ → web講演会のみ抽出
activity = pd.read_csv(os.path.join(DATA_DIR, FILE_ACTIVITY))
n_activity_all = len(activity)
activity["品目コード"] = activity["品目コード"].astype(str).str.strip().str.zfill(5)
web_lecture = activity[
    (activity["品目コード"] == ENT_PRODUCT_CODE)
    & (activity["活動種別"] == ACTIVITY_CHANNEL_FILTER)
].copy()

# 5. 視聴データ結合 (デジタル + web講演会)
common_cols = ["活動日_dt", "品目コード", "活動種別", "活動種別コード", "fac_honin", "doc"]
viewing = pd.concat([digital[common_cols], web_lecture[common_cols]], ignore_index=True)
n_viewing_combined = len(viewing)
viewing = viewing.rename(columns={
    "活動日_dt": "view_date",
    "fac_honin": "facility_id",
    "doc": "doctor_id",
    "活動種別": "channel_category",
})
viewing["view_date"] = pd.to_datetime(viewing["view_date"], format="mixed")

daily["delivery_date"] = pd.to_datetime(daily["delivery_date"])
months = pd.date_range(start=START_DATE, periods=N_MONTHS, freq="MS")

print(f"  売上データ: {n_sales_all:,} 行 → ENT品目: {len(daily):,} 行")
print(f"  RW医師リスト: {n_rw_all} 行 → seg非空: {len(doctor_master)} 行")
print(f"  デジタル視聴: {n_digital_all:,} 行, 活動データ: {n_activity_all:,} 行")
print(f"  視聴データ結合 (ENT, web講演会抽出): {n_viewing_combined:,} 行")

# JSON結果読み込み
print("\n[JSON結果読み込み]")
with open(os.path.join(RESULTS_DIR, "did_results.json"), "r", encoding="utf-8") as f:
    did_results = json.load(f)
with open(os.path.join(RESULTS_DIR, "cate_results.json"), "r", encoding="utf-8") as f:
    cate_results = json.load(f)
print("  did_results.json, cate_results.json 読み込み完了")


# ================================================================
# 除外フロー再実行 (カウント取得 + 対象医師特定)
# ================================================================
print("\n[除外フロー再実行]")

docs_per_fac = doctor_master.groupby("facility_id")["doctor_id"].nunique()
single_doc_facs = set(docs_per_fac[docs_per_fac == 1].index)

facs_per_doc = doctor_master.groupby("doctor_id")["facility_id"].nunique()
single_fac_docs = set(facs_per_doc[facs_per_doc == 1].index)
multi_fac_docs = set(facs_per_doc[facs_per_doc > 1].index)

clean_pairs = doctor_master[
    (doctor_master["facility_id"].isin(single_doc_facs))
    & (doctor_master["doctor_id"].isin(single_fac_docs))
].copy()
fac_to_doc = dict(zip(clean_pairs["facility_id"], clean_pairs["doctor_id"]))
doc_to_fac = dict(zip(clean_pairs["doctor_id"], clean_pairs["facility_id"]))
clean_doc_ids = set(clean_pairs["doctor_id"])

washout_end = months[WASHOUT_MONTHS - 1] + pd.offsets.MonthEnd(0)
viewing_clean = viewing[viewing["doctor_id"].isin(clean_doc_ids)].copy()
washout_viewers = set(
    viewing_clean[viewing_clean["view_date"] <= washout_end]["doctor_id"].unique()
)
clean_doc_ids -= washout_viewers

viewing_after_washout = viewing_clean[
    (viewing_clean["doctor_id"].isin(clean_doc_ids))
    & (viewing_clean["view_date"] > washout_end)
]
first_view = viewing_after_washout.groupby("doctor_id")["view_date"].min().reset_index()
first_view.columns = ["doctor_id", "first_view_date"]
first_view["first_view_month"] = (
    (first_view["first_view_date"].dt.year - 2023) * 12
    + first_view["first_view_date"].dt.month - 4
)

late_adopters = set(
    first_view[first_view["first_view_month"] > LAST_ELIGIBLE_MONTH]["doctor_id"]
)
clean_doc_ids -= late_adopters

treated_doc_ids = set(
    first_view[first_view["first_view_month"] <= LAST_ELIGIBLE_MONTH]["doctor_id"]
) & clean_doc_ids
all_viewing_doc_ids = set(viewing["doctor_id"].unique())
control_doc_ids = clean_doc_ids - all_viewing_doc_ids
analysis_doc_ids = treated_doc_ids | control_doc_ids
analysis_fac_ids = {doc_to_fac[d] for d in analysis_doc_ids}

print(f"  処置群: {len(treated_doc_ids)}, 対照群: {len(control_doc_ids)}, 合計: {len(analysis_fac_ids)}")


# ================================================================
# CONSORT フロー図生成
# ================================================================
print("\n[CONSORT フロー図生成]")

def create_consort_diagram(flow):
    fig, ax = plt.subplots(figsize=(10, 18))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis("off")
    fig.suptitle("解析対象集団の選定", fontsize=14, fontweight="bold", y=0.98)

    def draw_box(ax, x, y, w, h, text, color="#E8F4FD", edge="#2196F3"):
        box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                             boxstyle="round,pad=0.15", facecolor=color,
                             edgecolor=edge, linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y, text, ha="center", va="center", fontsize=9, fontweight="bold")

    def draw_excluded(ax, x, y, w, h, text, color="#FFEBEE", edge="#E53935"):
        box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                             boxstyle="round,pad=0.1", facecolor=color,
                             edgecolor=edge, linewidth=1.2)
        ax.add_patch(box)
        ax.text(x, y, text, ha="center", va="center", fontsize=8, color="#B71C1C")

    def draw_arrow(ax, x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="-|>", color="#455A64", lw=1.5))

    def draw_arrow_right(ax, x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="-|>", color="#E53935", lw=1.2, ls="--"))

    # Data from flow
    n_sales_all = flow.get("total_delivery_rows", 0)
    n_ent_rows = flow.get("ent_delivery_rows", 0)
    n_rw_all = flow.get("total_rw_list", 0)
    n_ent_rw = flow.get("ent_rw_doctors", 0)
    n_view_all = flow.get("total_viewing_rows", 0)
    n_view_after = flow.get("viewing_after_filter", 0)
    total_docs = flow["total_doctors"]
    total_facs = flow["total_facilities"]
    n_single = flow["single_doc_facilities"]
    n_multi_doc = flow["multi_doc_facilities"]
    n_multi_fac = flow["multi_fac_doctors"]
    n_clean = flow["clean_pairs"]
    n_washout = flow["washout_excluded"]
    n_late = flow["late_excluded"]
    n_treated = flow["final_treated"]
    n_control = flow["final_control"]
    n_final = flow["final_total"]

    cx = 4.0  # center x for main boxes
    ex = 8.0  # x for exclusion boxes
    sp = 1.45  # vertical spacing

    # Step 0a: 全品目売上データ → ENT品目のみ
    y = 19.0
    draw_box(ax, cx, y, 4.2, 0.9,
             f"全品目売上データ\n{n_sales_all:,} 行")
    draw_arrow(ax, cx, y - 0.45, cx, y - sp + 0.45)
    y -= sp
    draw_box(ax, cx, y, 4.2, 0.9,
             f"ENT品目の売上データ\n{n_ent_rows:,} 行")
    draw_arrow_right(ax, cx + 2.1, y, ex - 1.5, y)
    draw_excluded(ax, ex, y, 2.8, 0.7,
                  f"他品目\n{n_sales_all - n_ent_rows:,} 行除外")

    # Step 0b: RW医師リスト → seg非空の医師
    draw_arrow(ax, cx, y - 0.45, cx, y - sp + 0.45)
    y -= sp
    draw_box(ax, cx, y, 4.2, 0.9,
             f"RW対象医師 (seg非空)\n{n_ent_rw} 行 / {total_facs}施設 / {total_docs}医師")
    draw_arrow_right(ax, cx + 2.1, y, ex - 1.5, y)
    draw_excluded(ax, ex, y, 2.8, 0.7,
                  f"seg空欄\n{n_rw_all - n_ent_rw} 行除外")

    # Step 0c: 視聴データ結合 (デジタル + web講演会)
    draw_arrow(ax, cx, y - 0.45, cx, y - sp + 0.45)
    y -= sp
    draw_box(ax, cx, y, 4.2, 0.9,
             f"視聴データ結合\n(デジタル+web講演会)\n{n_view_after:,} 行")
    draw_arrow_right(ax, cx + 2.1, y, ex - 1.5, y)
    draw_excluded(ax, ex, y, 2.8, 0.7,
                  f"他品目/その他活動\n{n_view_all - n_view_after:,} 行除外")

    # Step 1: 1施設複数医師除外
    draw_arrow(ax, cx, y - 0.45, cx, y - sp + 0.45)
    y -= sp
    draw_box(ax, cx, y, 4.2, 0.9,
             f"1施設1医師の施設\n{n_single}施設")
    draw_arrow_right(ax, cx + 2.1, y, ex - 1.5, y)
    draw_excluded(ax, ex, y, 2.8, 0.7,
                  f"複数医師施設\n{n_multi_doc}施設 除外")

    # Step 2: 1医師複数施設除外
    draw_arrow(ax, cx, y - 0.45, cx, y - sp + 0.45)
    y -= sp
    draw_box(ax, cx, y, 4.2, 0.9,
             f"1医師1施設ペア\n{n_clean}施設")
    draw_arrow_right(ax, cx + 2.1, y, ex - 1.5, y)
    draw_excluded(ax, ex, y, 2.8, 0.7,
                  f"複数施設所属\n{n_multi_fac}医師 除外")

    # Step 3: Wash-out除外
    draw_arrow(ax, cx, y - 0.45, cx, y - sp + 0.45)
    y -= sp
    after_washout = n_clean - n_washout
    draw_box(ax, cx, y, 4.2, 0.9,
             f"wash-out後\n{after_washout}施設")
    draw_arrow_right(ax, cx + 2.1, y, ex - 1.5, y)
    draw_excluded(ax, ex, y, 2.8, 0.7,
                  f"wash-out視聴\n{n_washout}医師 除外")

    # Step 4: 遅延除外
    draw_arrow(ax, cx, y - 0.45, cx, y - sp + 0.45)
    y -= sp
    after_late = after_washout - n_late
    draw_box(ax, cx, y, 4.2, 0.9,
             f"遅延除外後\n{after_late}施設")
    draw_arrow_right(ax, cx + 2.1, y, ex - 1.5, y)
    draw_excluded(ax, ex, y, 2.8, 0.7,
                  f"遅延視聴者\n{n_late}医師 除外")

    # Final: analysis sample
    draw_arrow(ax, cx, y - 0.45, cx, y - sp + 0.45)
    y -= sp
    draw_box(ax, cx, y, 4.2, 0.9,
             f"最終分析対象\n{n_final}施設", color="#E8F5E9", edge="#4CAF50")

    # Split to treated / control
    draw_arrow(ax, cx - 0.8, y - 0.45, cx - 1.8, y - sp + 0.45)
    draw_arrow(ax, cx + 0.8, y - 0.45, cx + 1.8, y - sp + 0.45)
    y -= sp

    draw_box(ax, cx - 2.0, y, 3.0, 0.9,
             f"処置群\n{n_treated}施設", color="#FFF3E0", edge="#FF9800")
    draw_box(ax, cx + 2.0, y, 3.0, 0.9,
             f"対照群\n{n_control}施設", color="#E3F2FD", edge="#1565C0")

    return fig


consort_fig = create_consort_diagram(did_results["exclusion_flow"])
consort_b64 = fig_to_base64(consort_fig, dpi=150)
print("  CONSORT図生成完了")


# ================================================================
# 視聴パターン可視化
# ================================================================
print("\n[視聴パターン可視化]")

def create_viewing_patterns_figure(viewing, doctor_master, months,
                                   washout_viewers, late_adopters, multi_fac_docs,
                                   treated_doc_ids, doc_to_fac):
    """除外例と解析対象の視聴パターンを可視化"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("視聴パターンの可視化: 除外例と解析対象", fontsize=13, fontweight="bold")

    # --- (a) Wash-out除外例 ---
    ax = axes[0, 0]
    wo_docs = sorted(list(washout_viewers))[:5]
    if wo_docs:
        for i, doc_id in enumerate(wo_docs):
            doc_views = viewing[viewing["doctor_id"] == doc_id].copy()
            doc_views["month_index"] = (
                (doc_views["view_date"].dt.year - 2023) * 12
                + doc_views["view_date"].dt.month - 4
            )
            view_months = sorted(doc_views["month_index"].unique())
            for vm in view_months:
                ax.barh(i, 1, left=vm - 0.5, height=0.6, color="#E53935", alpha=0.7)
            ax.text(-1.5, i, doc_id, ha="right", va="center", fontsize=7)
    ax.axvline(WASHOUT_MONTHS - 0.5, color="blue", ls="--", lw=1.5, label="wash-out境界")
    ax.set_xlabel("月 (0=2023/4)")
    ax.set_title("(a) 除外: wash-out期間視聴者", fontsize=10)
    ax.set_xlim(-2, N_MONTHS)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="x")

    # --- (b) 複数施設所属除外例 ---
    ax = axes[0, 1]
    mf_docs = sorted(list(multi_fac_docs))[:5]
    if mf_docs:
        for i, doc_id in enumerate(mf_docs):
            doc_facs = doctor_master[doctor_master["doctor_id"] == doc_id]["facility_id"].unique()
            colors = ["#1565C0", "#FF9800", "#4CAF50", "#9C27B0"]
            for j, fac_id in enumerate(doc_facs):
                ax.barh(i, 0.4, left=j, height=0.6, color=colors[j % len(colors)], alpha=0.8)
                ax.text(j + 0.2, i + 0.35, fac_id, ha="center", va="bottom", fontsize=6)
            ax.text(-0.5, i, doc_id, ha="right", va="center", fontsize=7)
    ax.set_xlabel("施設数")
    ax.set_title("(b) 除外: 複数施設所属医師", fontsize=10)
    ax.grid(True, alpha=0.2, axis="x")

    # --- (c) 遅延視聴者除外例 ---
    ax = axes[1, 0]
    late_docs = sorted(list(late_adopters))[:5]
    if late_docs:
        for i, doc_id in enumerate(late_docs):
            doc_views = viewing[viewing["doctor_id"] == doc_id].copy()
            doc_views["month_index"] = (
                (doc_views["view_date"].dt.year - 2023) * 12
                + doc_views["view_date"].dt.month - 4
            )
            view_months = sorted(doc_views["month_index"].unique())
            for vm in view_months:
                ax.barh(i, 1, left=vm - 0.5, height=0.6, color="#FF6F00", alpha=0.7)
            ax.text(-1.5, i, doc_id, ha="right", va="center", fontsize=7)
    ax.axvline(LAST_ELIGIBLE_MONTH + 0.5, color="red", ls="--", lw=1.5, label="遅延境界(2025/10)")
    ax.set_xlabel("月 (0=2023/4)")
    ax.set_title("(c) 除外: 遅延視聴者 (初回>=2025/10)", fontsize=10)
    ax.set_xlim(-2, N_MONTHS)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="x")

    # --- (d) 解析対象の視聴パターン (代表例) ---
    ax = axes[1, 1]
    treated_list = sorted(list(treated_doc_ids))
    sample_docs = treated_list[:min(15, len(treated_list))]
    ch_colors = {"Webinar": "#1f77b4", "e-contents": "#ff7f0e", "web講演会": "#2ca02c"}

    for i, doc_id in enumerate(sample_docs):
        doc_views = viewing[viewing["doctor_id"] == doc_id].copy()
        doc_views["month_index"] = (
            (doc_views["view_date"].dt.year - 2023) * 12
            + doc_views["view_date"].dt.month - 4
        )
        for ch in ch_colors:
            ch_views = doc_views[doc_views["channel_category"] == ch]
            if len(ch_views) == 0:
                continue
            view_months = sorted(ch_views["month_index"].unique())
            for vm in view_months:
                ax.barh(i, 1, left=vm - 0.5, height=0.5,
                        color=ch_colors[ch], alpha=0.7)

    ax.set_xlabel("月 (0=2023/4)")
    ax.set_title("(d) 解析対象: 処置群の視聴パターン (代表例)", fontsize=10)
    ax.set_xlim(-2, N_MONTHS)
    ax.grid(True, alpha=0.2, axis="x")

    # legend for channels
    legend_patches = [mpatches.Patch(color=c, label=ch, alpha=0.7)
                      for ch, c in ch_colors.items()]
    ax.legend(handles=legend_patches, fontsize=8, loc="upper left")

    plt.tight_layout()
    return fig


viewing_fig = create_viewing_patterns_figure(
    viewing, doctor_master, months,
    washout_viewers, late_adopters, multi_fac_docs,
    treated_doc_ids, doc_to_fac
)
viewing_b64 = fig_to_base64(viewing_fig, dpi=130)
print("  視聴パターン図生成完了")


# ================================================================
# コホート分布グラフ生成
# ================================================================
print("\n[コホート分布グラフ生成]")

def create_cohort_distribution_figure(cohort_dist):
    """コホート分布（初回視聴月ごとの施設数）をバーチャートで可視化"""
    labels = list(cohort_dist.keys())
    values = list(cohort_dist.values())

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(range(len(labels)), values, color="#1565C0", alpha=0.85, edgecolor="white")

    # 棒の上に数値ラベル
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                str(v), ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("施設数", fontsize=10)
    ax.set_title("コホート分布: 処置群の初回視聴月ごとの施設数", fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.25)
    ax.grid(True, alpha=0.2, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig

cohort_dist_fig = create_cohort_distribution_figure(did_results["cohort_distribution"])
cohort_dist_b64 = fig_to_base64(cohort_dist_fig, dpi=130)
print("  コホート分布グラフ生成完了")


# ================================================================
# 既存PNGの読み込み
# ================================================================
print("\n[既存PNG読み込み]")

existing_pngs = {}
for name in ["staggered_did_results.png", "cate_results.png", "cate_dynamic_effects.png"]:
    path = os.path.join(SCRIPT_DIR, name)
    if os.path.exists(path):
        existing_pngs[name] = png_to_base64(path)
        print(f"  {name} 読み込み完了")
    else:
        print(f"  {name} が見つかりません")
        existing_pngs[name] = ""


# ================================================================
# HTMLテンプレート
# ================================================================
HTML_TEMPLATE = Template("""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Staggered DID分析 総合レポート</title>
<style>
  :root {
    --primary: #1565C0;
    --accent: #FF6F00;
    --success: #2E7D32;
    --danger: #C62828;
    --bg: #FAFAFA;
    --card-bg: #FFFFFF;
    --border: #E0E0E0;
    --text: #212121;
    --text-secondary: #616161;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: "Yu Gothic", "Meiryo", "Hiragino Sans", sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
  }
  .container { max-width: 1100px; margin: 0 auto; padding: 20px; }
  header {
    background: linear-gradient(135deg, var(--primary), #0D47A1);
    color: white;
    padding: 30px 0;
    text-align: center;
  }
  header h1 { font-size: 1.8em; margin-bottom: 8px; }
  header p { opacity: 0.9; font-size: 0.95em; }

  nav {
    background: var(--card-bg);
    border-bottom: 1px solid var(--border);
    padding: 12px 0;
    position: sticky;
    top: 0;
    z-index: 100;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
  }
  nav .container { display: flex; flex-wrap: wrap; gap: 8px; padding: 0 20px; }
  nav a {
    text-decoration: none;
    color: var(--primary);
    padding: 4px 12px;
    border-radius: 4px;
    font-size: 0.85em;
    font-weight: bold;
    transition: background 0.2s;
  }
  nav a:hover { background: #E3F2FD; }

  section {
    background: var(--card-bg);
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    margin: 24px 0;
    padding: 28px;
  }
  h2 {
    color: var(--primary);
    font-size: 1.35em;
    border-bottom: 3px solid var(--primary);
    padding-bottom: 8px;
    margin-bottom: 20px;
  }
  h3 {
    color: #37474F;
    font-size: 1.1em;
    margin: 18px 0 10px 0;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0;
    font-size: 0.9em;
  }
  th, td {
    border: 1px solid var(--border);
    padding: 8px 12px;
    text-align: center;
  }
  th {
    background: #E3F2FD;
    font-weight: bold;
    color: #1565C0;
  }
  tr:nth-child(even) { background: #F5F5F5; }
  tr:hover { background: #E8F5E9; }

  .sig { color: var(--danger); font-weight: bold; }
  .ns { color: var(--text-secondary); }

  .img-container {
    text-align: center;
    margin: 16px 0;
  }
  .img-container img {
    max-width: 100%;
    height: auto;
    border: 1px solid var(--border);
    border-radius: 4px;
  }

  .param-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin: 12px 0;
  }
  .param-box {
    background: #F5F5F5;
    border-radius: 6px;
    padding: 14px;
    border-left: 4px solid var(--primary);
  }
  .param-box h4 {
    color: var(--primary);
    margin-bottom: 8px;
    font-size: 0.95em;
  }
  .param-box ul {
    list-style: none;
    font-size: 0.88em;
  }
  .param-box ul li { padding: 2px 0; }
  .param-box ul li::before { content: "\\25B8 "; color: var(--primary); }

  .highlight-box {
    background: #FFF8E1;
    border: 1px solid #FFE082;
    border-radius: 6px;
    padding: 14px 18px;
    margin: 12px 0;
    font-size: 0.92em;
  }

  .conclusion-box {
    background: #E8F5E9;
    border: 1px solid #A5D6A7;
    border-radius: 6px;
    padding: 18px;
    margin: 16px 0;
  }

  footer {
    text-align: center;
    padding: 20px;
    color: var(--text-secondary);
    font-size: 0.85em;
    border-top: 1px solid var(--border);
    margin-top: 32px;
  }

  @media print {
    @page { size: A4 landscape; margin: 15mm; }
    body { background: white; font-size: 10pt; }
    nav { display: none; }
    section { break-inside: avoid; box-shadow: none; border: 1px solid #ccc; }
    .img-container img { max-width: 90%; }
    header { background: white; color: black; border-bottom: 2px solid black; }
  }
</style>
</head>
<body>

<header>
  <div class="container">
    <h1>Staggered DID分析 総合レポート</h1>
    <p>デジタルコンテンツ視聴の因果効果検証 | 観測期間: 2023/4 - 2025/12</p>
  </div>
</header>

<nav>
  <div class="container">
    <a href="#sec1">1. 分析要件</a>
    <a href="#sec2">2. 解析対象集団の選定</a>
    <a href="#sec3">3. 基礎集計</a>
    <a href="#sec4">4. 視聴パターン</a>
    <a href="#sec5">5. DID推定結果</a>
    <a href="#sec6">6. CATE分析</a>
    <a href="#sec7">7. 結論</a>
  </div>
</nav>

<div class="container">

<!-- ============================================================ -->
<!-- Section 1: 分析要件定義 -->
<!-- ============================================================ -->
<section id="sec1">
<h2>1. 分析要件定義</h2>

<h3>1.1 分析目的</h3>
<p>デジタルコンテンツ（Webinar, e-contents, web講演会）の視聴が、品目コード={{ ent_product_code }}<の納入額に与える因果効果を推定する。視聴開始時期が施設医師ごとに異なる「ずれた処置（staggered treatment）」に対応したDID推定量を使用する。</p>
<p style="margin-top:8px;">分析対象は <strong>品目コード={{ ent_product_code }}</strong> (ENT) の売上データに限定し、医師は <strong>RW医師</strong> のみを使用する。</p>

<h3>1.2 データ構造</h3>
<div class="param-grid">
  <div class="param-box">
    <h4>売上データ (sales.csv)</h4>
    <ul>
      <li>日別・施設別・品目別の売上実績</li>
      <li>カラム: 日付, 施設（本院に合算）コード, DCF施設コード, 品目コード, 実績</li>
      <li>日付: YYYYMMDD文字列、実績: 数値文字列</li>
      <li>全品目: {{ n_sales_all_rows }} 行 → ENT品目: {{ n_sales_rows }} 行</li>
    </ul>
  </div>
  <div class="param-box">
    <h4>デジタル視聴データ (デジタル視聴データ.csv)</h4>
    <ul>
      <li>Webinar / e-contents の視聴ログ</li>
      <li>カラム: 活動日_dt, 品目コード, 活動種別, 活動種別コード, fac_honin, doc 等</li>
      <li>{{ n_digital_all_rows }} 行</li>
    </ul>
  </div>
  <div class="param-box">
    <h4>活動データ (活動データ.csv)</h4>
    <ul>
      <li>MR入力の活動記録 (web講演会を含む)</li>
      <li>カラム: 活動日_dt, 品目コード, 活動種別コード, 活動種別, fac_honin, doc 等</li>
      <li>{{ n_activity_all_rows }} 行 → web講演会抽出後に視聴データと結合</li>
    </ul>
  </div>
  <div class="param-box">
    <h4>RW医師リスト (rw_list.csv)</h4>
    <ul>
      <li>ENT対象医師リスト (品目カラムなし)</li>
      <li>カラム: doc, doc_name, fac_honin, fac_honin_name, fac, fac_name, seg</li>
      <li>seg非空 = RW対象</li>
      <li>全体: {{ n_rw_all_rows }} 行 → seg非空: {{ n_doctor_rows }} 行</li>
    </ul>
  </div>
</div>

<div class="highlight-box">
  <strong>視聴データの結合:</strong>
  デジタル視聴データ (Webinar + e-contents) と 活動データ (web講演会のみ抽出) を結合して分析用の視聴データとする。
  結合後: {{ n_viewing_combined_rows }} 行
</div>

<h3>1.3 分析パラメータ</h3>
<div class="param-grid">
  <div class="param-box">
    <h4>期間設定</h4>
    <ul>
      <li>観測期間: 2023/4 - 2025/12 (33ヶ月)</li>
      <li>Wash-out期間: 2ヶ月 (2023/4-5)</li>
      <li>最終適格月: 2025/9 (month 29)</li>
    </ul>
  </div>
  <div class="param-box">
    <h4>除外基準</h4>
    <ul>
      <li>売上データ: ENT品目コード ({{ ent_product_code }}) 以外を除外</li>
      <li>医師リスト: seg列が空欄の医師を除外</li>
      <li>視聴データ: ENT品目 + 対象チャネル (Webinar, e-contents, web講演会) のみ</li>
      <li>1施設に複数医師が所属 → 施設除外</li>
      <li>1医師が複数施設に所属 → 医師除外</li>
      <li>wash-out期間に視聴あり → 除外</li>
      <li>初回視聴が2025/10以降 → 除外</li>
    </ul>
  </div>
</div>

<h3>1.4 推定手法</h3>
<div class="param-grid">
  <div class="param-box">
    <h4>TWFE (Two-Way Fixed Effects)</h4>
    <ul>
      <li>標準的な双方向固定効果モデル</li>
      <li>施設固定効果 + 時間固定効果</li>
      <li>クラスターロバストSE（施設レベル）</li>
    </ul>
  </div>
  <div class="param-box">
    <h4>Callaway-Sant'Anna (2021)</h4>
    <ul>
      <li>ずれた処置に頑健な推定量</li>
      <li>コホート別ATT(g,t) → 動的効果集約</li>
      <li>Bootstrap SE (N=200全体, N=100チャネル別)</li>
    </ul>
  </div>
</div>

<h3>1.5 用語解説</h3>
<table>
  <tr><th>用語</th><th>補足説明</th></tr>
  <tr><td>DID (差の差分法)</td><td>処置群と対照群の「前後差の差」で因果効果を推定する準実験的手法</td></tr>
  <tr><td>ATT (Average Treatment effect on the Treated)</td><td>処置を受けた群における平均処置効果</td></tr>
  <tr><td>TWFE (Two-Way Fixed Effects)</td><td>個体（施設）と時点の両方の固定効果を制御する回帰モデル</td></tr>
  <tr><td>Callaway-Sant'Anna (CS)</td><td>処置開始時期が異なる場合のバイアスを補正するDID推定量</td></tr>
  <tr><td>CATE (Conditional Average Treatment Effect)</td><td>属性条件付きの平均処置効果（異質的処置効果）</td></tr>
  <tr><td>コホート</td><td>同一時期に処置を受け始めた施設のグループ</td></tr>
  <tr><td>イベント時間 (event time)</td><td>処置開始からの相対的な経過月数 (e=0が処置開始月)</td></tr>
  <tr><td>Bootstrap</td><td>リサンプリングにより標準誤差・信頼区間を算出する統計手法</td></tr>
  <tr><td>95% CI (信頼区間)</td><td>真の効果がこの範囲に含まれる確率が95%である区間</td></tr>
  <tr><td>SE (標準誤差)</td><td>推定値のばらつきの大きさ、小さいほど推定が安定</td></tr>
  <tr><td>p値</td><td>帰無仮説（効果ゼロ）が正しい場合にこの推定値以上が観測される確率</td></tr>
  <tr><td>クラスターロバストSE</td><td>施設内の観測値の相関を考慮した標準誤差の補正</td></tr>
  <tr><td>解析対象集団の選定フロー</td><td>データソースから段階的な除外基準を適用して最終解析対象を確定する過程の図示</td></tr>
  <tr><td>fac_honin (施設本院コード)</td><td>本院に合算したコードで、分析の施設単位として使用</td></tr>
</table>
</section>

<!-- ============================================================ -->
<!-- Section 2: 解析対象集団の選定 -->
<!-- ============================================================ -->
<section id="sec2">
<h2>2. 解析対象集団の選定</h2>
<p>解析対象集団の選定フローを以下に示す。売上データ・RW医師リスト・視聴データの各ソースからENT品目を抽出し、施設-医師の1対1対応・wash-out・遅延視聴者の除外を経て、最終的な処置群・対照群を確定した。</p>
<div class="img-container">
  <img src="data:image/png;base64,{{ consort_b64 }}" alt="解析対象集団の選定">
</div>

<h3>除外フロー集計</h3>
<table>
  <tr>
    <th>ステップ</th>
    <th>条件</th>
    <th>除外数</th>
    <th>残余数</th>
  </tr>
  <tr>
    <td>初期</td>
    <td>全品目売上データ</td>
    <td>-</td>
    <td>{{ flow.total_delivery_rows }} 行</td>
  </tr>
  <tr>
    <td>[0a] ENT品目フィルタ</td>
    <td>品目コード = {{ ent_product_code }} のみ</td>
    <td>{{ flow.total_delivery_rows - flow.ent_delivery_rows }} 行</td>
    <td>{{ flow.ent_delivery_rows }} 行</td>
  </tr>
  <tr>
    <td>[0b] RW医師フィルタ</td>
    <td>rw_list.csv の seg 非空</td>
    <td>{{ flow.total_rw_list - flow.ent_rw_doctors }} 行</td>
    <td>{{ flow.ent_rw_doctors }} 行 ({{ flow.total_facilities }}施設 / {{ flow.total_doctors }}医師)</td>
  </tr>
  <tr>
    <td>[0c] 視聴データ結合</td>
    <td>デジタル視聴 + 活動(web講演会) ENT品目</td>
    <td>{{ flow.total_viewing_rows - flow.viewing_after_filter }} 行</td>
    <td>{{ flow.viewing_after_filter }} 行</td>
  </tr>
  <tr>
    <td>[A] 複数医師施設</td>
    <td>1施設に2名以上の医師</td>
    <td>{{ flow.multi_doc_facilities }}施設</td>
    <td>{{ flow.single_doc_facilities }}施設</td>
  </tr>
  <tr>
    <td>[B] 複数施設所属</td>
    <td>1医師が2施設以上に所属</td>
    <td>{{ flow.multi_fac_doctors }}医師</td>
    <td>{{ flow.clean_pairs }}施設</td>
  </tr>
  <tr>
    <td>[C] Wash-out視聴</td>
    <td>2023/4-5に視聴あり</td>
    <td>{{ flow.washout_excluded }}医師</td>
    <td>{{ flow.clean_pairs - flow.washout_excluded }}施設</td>
  </tr>
  <tr>
    <td>[D] 遅延視聴者</td>
    <td>初回視聴 >= 2025/10</td>
    <td>{{ flow.late_excluded }}医師</td>
    <td>{{ flow.clean_pairs - flow.washout_excluded - flow.late_excluded }}施設</td>
  </tr>
  <tr style="font-weight: bold; background: #E8F5E9;">
    <td>最終</td>
    <td>分析対象</td>
    <td>-</td>
    <td>{{ flow.final_total }}施設 (処置{{ flow.final_treated }} + 対照{{ flow.final_control }})</td>
  </tr>
</table>
</section>

<!-- ============================================================ -->
<!-- Section 3: 基礎集計 -->
<!-- ============================================================ -->
<section id="sec3">
<h2>3. 基礎集計</h2>

<h3>3.1 処置群・対照群の記述統計</h3>
<table>
  <tr>
    <th>群</th>
    <th>施設数</th>
    <th>月次売上額 平均</th>
    <th>月次売上額 SD</th>
    <th>パネル行数</th>
  </tr>
  <tr>
    <td>対照群 (未視聴)</td>
    <td>{{ did.n_control }}</td>
    <td>{{ "%.1f"|format(desc.control.mean) }}</td>
    <td>{{ "%.1f"|format(desc.control.std) }}</td>
    <td>{{ desc.control.count }}</td>
  </tr>
  <tr>
    <td>処置群 (視聴)</td>
    <td>{{ did.n_treated }}</td>
    <td>{{ "%.1f"|format(desc.treated.mean) }}</td>
    <td>{{ "%.1f"|format(desc.treated.std) }}</td>
    <td>{{ desc.treated.count }}</td>
  </tr>
</table>

<h3>3.2 コホート分布</h3>
<p>処置群の初回視聴月ごとの施設数分布:</p>
<div class="img-container">
  <img src="data:image/png;base64,{{ cohort_dist_b64 }}" alt="コホート分布">
</div>

{% if attr_dist %}
<h3>3.3 属性分布 (処置群 / 対照群)</h3>
{% for attr_name, dist in attr_dist.items() %}
<h4 style="margin-top:12px; color:#37474F;">{{ attr_name }}</h4>
<table>
  <tr>
    <th>群</th>
    {% for level in dist.treated.keys() %}
    <th>{{ level }}</th>
    {% endfor %}
  </tr>
  <tr>
    <td>処置群</td>
    {% for level, n in dist.treated.items() %}
    <td>{{ n }}</td>
    {% endfor %}
  </tr>
  <tr>
    <td>対照群</td>
    {% for level, n in dist.control.items() %}
    <td>{{ n }}</td>
    {% endfor %}
  </tr>
</table>
{% endfor %}
{% endif %}
</section>

<!-- ============================================================ -->
<!-- Section 4: 視聴パターン -->
<!-- ============================================================ -->
<section id="sec4">
<h2>4. 視聴パターンの可視化</h2>
<p>除外された医師と解析対象の医師の視聴パターンを比較する。</p>

<div class="highlight-box">
  <strong>凡例:</strong>
  (a) wash-out期間（2023/4-5）に視聴がある医師 → 除外 /
  (b) 複数施設に所属する医師 → 除外 /
  (c) 初回視聴が2025/10以降の医師 → 除外 /
  (d) 解析対象（処置群）の代表的視聴パターン
</div>

<div class="img-container">
  <img src="data:image/png;base64,{{ viewing_b64 }}" alt="視聴パターン可視化">
</div>
</section>

<!-- ============================================================ -->
<!-- Section 5: DID推定結果 -->
<!-- ============================================================ -->
<section id="sec5">
<h2>5. DID推定結果</h2>

<h3>5.1 全体推定結果の比較</h3>
<table>
  <tr>
    <th>手法</th>
    <th>ATT <small>(処置群の平均処置効果)</small></th>
    <th>SE <small>(標準誤差)</small></th>
    <th>p値</th>
    <th>95% CI <small>(信頼区間)</small></th>
    <th>有意性</th>
  </tr>
  <tr>
    <td>TWFE (全体)</td>
    <td>{{ "%.2f"|format(twfe.att) }}</td>
    <td>{{ "%.2f"|format(twfe.se) }}</td>
    <td>{{ "%.6f"|format(twfe.p) }}</td>
    <td>[{{ "%.2f"|format(twfe.ci_lo) }}, {{ "%.2f"|format(twfe.ci_hi) }}]</td>
    <td class="{{ 'sig' if twfe.sig != 'n.s.' else 'ns' }}">{{ twfe.sig }}</td>
  </tr>
  <tr>
    <td>CS (全体)</td>
    <td>{{ "%.2f"|format(cs.att) }}</td>
    <td>{{ "%.2f"|format(cs.se) }}</td>
    <td>{{ "%.6f"|format(cs.p) }}</td>
    <td>[{{ "%.2f"|format(cs.ci_lo) }}, {{ "%.2f"|format(cs.ci_hi) }}]</td>
    <td class="{{ 'sig' if cs.sig != 'n.s.' else 'ns' }}">{{ cs.sig }}</td>
  </tr>
  {% for ch_name, ch in channels.items() %}
  <tr>
    <td>CS ({{ ch_name }})</td>
    <td>{{ "%.2f"|format(ch.att) }}</td>
    <td>{{ "%.2f"|format(ch.se) }}</td>
    <td>{{ "%.6f"|format(ch.p) }}</td>
    <td>[{{ "%.2f"|format(ch.ci_lo) }}, {{ "%.2f"|format(ch.ci_hi) }}]</td>
    <td class="{{ 'sig' if ch.sig != 'n.s.' else 'ns' }}">{{ ch.sig }}</td>
  </tr>
  {% endfor %}
</table>

<h3>5.2 CS動的効果 (全体)</h3>
<table>
  <tr>
    <th>イベント時間</th>
    <th>ATT</th>
    <th>SE</th>
    <th>95% CI下限</th>
    <th>95% CI上限</th>
  </tr>
  {% for row in cs.dynamic %}
  <tr{% if row.event_time == 0 %} style="background:#FFF3E0; font-weight:bold;"{% endif %}>
    <td>e={{ row.event_time }}</td>
    <td>{{ "%.2f"|format(row.att) }}</td>
    <td>{{ "%.2f"|format(row.se) }}</td>
    <td>{{ "%.2f"|format(row.ci_lo) }}</td>
    <td>{{ "%.2f"|format(row.ci_hi) }}</td>
  </tr>
  {% endfor %}
</table>

<h3>5.3 可視化</h3>
{% if png_did %}
<div class="img-container">
  <img src="data:image/png;base64,{{ png_did }}" alt="Staggered DID Results">
</div>
{% else %}
<p>staggered_did_results.png が見つかりません。</p>
{% endif %}

{% if twfe_robust %}
<h3>5.4 ロバストネスチェック: MR活動共変量</h3>
<p>MR活動（面談、面談_アポ、説明会等）の月次実施回数を共変量として追加した場合の推定結果。
デジタル視聴の効果推定がMR活動の時変交絡に頑健かどうかを検証する。</p>

<table>
  <tr>
    <th>モデル</th>
    <th>ATT</th>
    <th>SE</th>
    <th>p値</th>
    <th>95% CI</th>
    <th>有意性</th>
  </tr>
  <tr>
    <td>TWFE (メイン)</td>
    <td>{{ "%.2f"|format(twfe.att) }}</td>
    <td>{{ "%.2f"|format(twfe.se) }}</td>
    <td>{{ "%.6f"|format(twfe.p) }}</td>
    <td>[{{ "%.2f"|format(twfe.ci_lo) }}, {{ "%.2f"|format(twfe.ci_hi) }}]</td>
    <td class="{{ 'sig' if twfe.sig != 'n.s.' else 'ns' }}">{{ twfe.sig }}</td>
  </tr>
  <tr>
    <td>TWFE (+MR活動共変量)</td>
    <td>{{ "%.2f"|format(twfe_robust.att) }}</td>
    <td>{{ "%.2f"|format(twfe_robust.se) }}</td>
    <td>{{ "%.6f"|format(twfe_robust.p) }}</td>
    <td>[{{ "%.2f"|format(twfe_robust.ci_lo) }}, {{ "%.2f"|format(twfe_robust.ci_hi) }}]</td>
    <td class="{{ 'sig' if twfe_robust.sig != 'n.s.' else 'ns' }}">{{ twfe_robust.sig }}</td>
  </tr>
</table>

<table>
  <tr>
    <th>共変量</th>
    <th>係数</th>
    <th>SE</th>
    <th>p値</th>
    <th>有意性</th>
  </tr>
  <tr>
    <td>MR活動回数</td>
    <td>{{ "%.2f"|format(twfe_robust.mr_activity_coef) }}</td>
    <td>{{ "%.2f"|format(twfe_robust.mr_activity_se) }}</td>
    <td>{{ "%.4f"|format(twfe_robust.mr_activity_p) }}</td>
    <td class="{{ 'sig' if twfe_robust.mr_activity_sig != 'n.s.' else 'ns' }}">{{ twfe_robust.mr_activity_sig }}</td>
  </tr>
</table>

<p style="margin-top:8px;">ATT変化率: <strong>{{ "%.1f"|format(twfe_robust.att_change_pct) }}%</strong></p>

<div class="highlight-box">
  <strong>解釈:</strong>
  MR活動（面談等）を共変量として追加してもATTの変化が小さい場合（目安: 変化率10%未満）、
  MR活動による時変交絡の影響は限定的であり、メインTWFE推定の信頼性が支持される。
</div>
{% endif %}
</section>

<!-- ============================================================ -->
<!-- Section 6: CATE分析 -->
<!-- ============================================================ -->
<section id="sec6">
<h2>6. CATE分析 (条件付き平均処置効果)</h2>

<h3>6.1 サブグループ別ATT推定値</h3>
{% for dim in cate_dims %}
<h4 style="margin-top:14px; color:#37474F;">{{ dim.name }}</h4>
<table>
  <tr>
    <th>レベル</th>
    <th>N</th>
    <th>ATT</th>
    <th>SE</th>
    <th>95% CI</th>
    {% if true_mods and dim.name in true_mods %}
    <th>DGP modifier</th>
    {% endif %}
  </tr>
  {% for level in dim.levels %}
  {% set r = cate[dim.name][level] %}
  <tr>
    <td>{{ level }}</td>
    <td>{{ r.n }}</td>
    {% if r.att is not none %}
    <td>{{ "%.1f"|format(r.att) }}</td>
    <td>{{ "%.1f"|format(r.se) }}</td>
    <td>[{{ "%.1f"|format(r.ci_lo) }}, {{ "%.1f"|format(r.ci_hi) }}]</td>
    {% else %}
    <td>N/A</td><td>N/A</td><td>N/A</td>
    {% endif %}
    {% if true_mods and dim.name in true_mods %}
    <td>{{ true_mods.get(dim.name, {}).get(level, "N/A") }}</td>
    {% endif %}
  </tr>
  {% endfor %}
</table>
{% endfor %}

{% if diff_tests %}
<h3>6.2 サブグループ間差の検定</h3>
{% for dim_name, diffs in diff_tests.items() %}
<h4 style="margin-top:12px; color:#37474F;">{{ dim_name }}</h4>
<table>
  <tr>
    <th>比較</th>
    <th>差</th>
    <th>SE</th>
    <th>p値</th>
    <th>有意性</th>
  </tr>
  {% for key, d in diffs.items() %}
  <tr>
    <td>{{ key }}</td>
    <td>{{ "%.1f"|format(d.diff) }}</td>
    <td>{{ "%.1f"|format(d.se) if d.se is not none else "N/A" }}</td>
    <td>{{ "%.4f"|format(d.p) if d.p is not none else "N/A" }}</td>
    <td class="{{ 'sig' if d.sig not in ['n.s.', 'N/A'] else 'ns' }}">{{ d.sig }}</td>
  </tr>
  {% endfor %}
</table>
{% endfor %}
{% endif %}

<h3>6.3 Forest Plot</h3>
{% if png_cate %}
<div class="img-container">
  <img src="data:image/png;base64,{{ png_cate }}" alt="CATE Forest Plot">
</div>
{% endif %}

<h3>6.4 動的効果 (サブグループ別)</h3>
{% if png_cate_dyn %}
<div class="img-container">
  <img src="data:image/png;base64,{{ png_cate_dyn }}" alt="CATE Dynamic Effects">
</div>
{% endif %}
</section>

<!-- ============================================================ -->
<!-- Section 7: 結論 -->
<!-- ============================================================ -->
<section id="sec7">
<h2>7. 結論・主な知見</h2>

<div class="conclusion-box">
<h3>全体効果</h3>
<ul style="margin:10px 0; padding-left:20px;">
  <li>TWFE推定 <small>(双方向固定効果)</small>: ATT <small>(処置群の平均処置効果)</small> = {{ "%.2f"|format(twfe.att) }} (SE={{ "%.2f"|format(twfe.se) }}, {{ twfe.sig }})</li>
  <li>CS推定 <small>(Callaway-Sant'Anna)</small>: ATT = {{ "%.2f"|format(cs.att) }} (SE={{ "%.2f"|format(cs.se) }}, {{ cs.sig }})</li>
</ul>

<h3>チャネル別効果 (CS推定)</h3>
<ul style="margin:10px 0; padding-left:20px;">
  {% for ch_name, ch in channels.items() %}
  <li>{{ ch_name }}: ATT = {{ "%.2f"|format(ch.att) }} (N={{ ch.n_treated }}, {{ ch.sig }})</li>
  {% endfor %}
</ul>

<h3>CATE <small>(条件付き平均処置効果 / 異質的処置効果)</small></h3>
<ul style="margin:10px 0; padding-left:20px;">
  {% for dim in cate_dims %}
  {% set valid_levels = [] %}
  {% for level in dim.levels %}
    {% if cate[dim.name][level].att is not none %}
      {% if valid_levels.append(level) %}{% endif %}
    {% endif %}
  {% endfor %}
  {% if valid_levels|length >= 2 %}
  <li><strong>{{ dim.name }}:</strong>
    {% for level in valid_levels %}
    {{ level }}({{ "%.1f"|format(cate[dim.name][level].att) }}){{ ", " if not loop.last else "" }}
    {% endfor %}
  </li>
  {% endif %}
  {% endfor %}
</ul>
</div>

<div class="highlight-box">
  <strong>注意事項:</strong><br>
  - サブグループのNが小さいため検出力は限定的<br>
  - 各サブグループのATTは共通の対照群を使用<br>
  - 本分析はシミュレーションデータに基づく方法論検証
</div>
</section>

</div>

<footer>
  <p>Staggered DID分析 総合レポート | 自動生成</p>
  <p>ブラウザの「印刷 → PDF保存」でPDF出力可能</p>
</footer>

</body>
</html>
""")


# ================================================================
# テンプレートデータ構築 & HTML生成
# ================================================================
print("\n[HTML生成]")

# チャネル結果をdot-access可能な辞書に変換
class DotDict(dict):
    __getattr__ = dict.__getitem__

channels = {}
for ch_name, ch_data in did_results.get("cs_channel", {}).items():
    channels[ch_name] = DotDict(ch_data)

# CATE結果をdot-access可能に
cate_data = {}
for dim_name, dim_data in cate_results.get("cate", {}).items():
    cate_data[dim_name] = {}
    for level, level_data in dim_data.items():
        cate_data[dim_name][level] = DotDict(level_data)

# diff_tests
diff_tests = {}
for dim_name, diffs in cate_results.get("diff_tests", {}).items():
    diff_tests[dim_name] = {}
    for key, d in diffs.items():
        diff_tests[dim_name][key] = DotDict(d)

template_data = {
    # 基本情報
    "ent_product_code": ENT_PRODUCT_CODE,
    "n_sales_all_rows": f"{n_sales_all:,}",
    "n_sales_rows": f"{len(daily):,}",
    "n_digital_all_rows": f"{n_digital_all:,}",
    "n_activity_all_rows": f"{n_activity_all:,}",
    "n_viewing_combined_rows": f"{n_viewing_combined:,}",
    "n_rw_all_rows": f"{n_rw_all:,}",
    "n_doctor_rows": f"{len(doctor_master):,}",

    # 除外フロー
    "flow": DotDict(did_results["exclusion_flow"]),

    # 記述統計
    "did": DotDict(did_results),
    "desc": DotDict({
        "control": DotDict(did_results["descriptive_stats"]["control"]),
        "treated": DotDict(did_results["descriptive_stats"]["treated"]),
    }),

    # 属性分布
    "attr_dist": {
        k: DotDict({
            "treated": v.get("treated", {}),
            "control": v.get("control", {}),
        })
        for k, v in cate_results.get("attr_distribution", {}).items()
    },

    # TWFE/CS結果
    "twfe": DotDict(did_results["twfe"]),
    "twfe_robust": DotDict(did_results["twfe_robust"]) if "twfe_robust" in did_results else None,
    "cs": DotDict(did_results["cs_overall"]),

    # チャネル別
    "channels": channels,

    # CATE
    "cate_dims": cate_results.get("dimensions", []),
    "cate": cate_data,
    "true_mods": cate_results.get("true_modifiers", {}),
    "diff_tests": diff_tests,

    # 画像
    "consort_b64": consort_b64,
    "viewing_b64": viewing_b64,
    "cohort_dist_b64": cohort_dist_b64,
    "png_did": existing_pngs.get("staggered_did_results.png", ""),
    "png_cate": existing_pngs.get("cate_results.png", ""),
    "png_cate_dyn": existing_pngs.get("cate_dynamic_effects.png", ""),
}

html_content = HTML_TEMPLATE.render(**template_data)

output_path = os.path.join(REPORTS_DIR, "analysis_report.html")
with open(output_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"  HTMLレポートを保存: {output_path}")
print(f"  ファイルサイズ: {os.path.getsize(output_path) / 1024:.0f} KB")
print("\n  ブラウザで開いて確認してください。")
print("  「印刷 → PDF保存」でPDF出力も可能です。")
print("\n" + "=" * 60)
print(" レポート生成完了")
print("=" * 60)
