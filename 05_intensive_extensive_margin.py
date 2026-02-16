"""
===================================================================
医師視聴パターン分析: Intensive vs Extensive Margin
===================================================================
分析目的:
  1. 同じ医師への複数回視聴（深さ）vs 視聴医師層拡大（広さ）の効果比較
  2. 定常視聴群・単発視聴群・未視聴群の実績推移比較
  3. Intensive Margin (既存医師への追加視聴) vs Extensive Margin (新規医師獲得) の効果推定

手法:
  - 施設×月次レベルで Intensive/Extensive 指標を構築
  - TWFE回帰で両指標の係数を推定
  - 医師視聴頻度別の実績推移を可視化
===================================================================
"""

import os
import warnings
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")

for _font in ["Yu Gothic", "MS Gothic", "Meiryo", "Hiragino Sans", "IPAexGothic"]:
    try:
        matplotlib.rcParams["font.family"] = _font
        break
    except Exception:
        pass
matplotlib.rcParams["axes.unicode_minus"] = False

# === データファイル・カラム設定 (02と同一) ===
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

START_DATE = "2023-04-01"
N_MONTHS = 33
WASHOUT_MONTHS = 2
LAST_ELIGIBLE_MONTH = 29

# 視聴頻度の閾値設定
FREQUENT_THRESHOLD = 3  # 定常視聴: 3回以上


# ================================================================
# データ読み込み + 除外フロー (02と同一ロジック)
# ================================================================
print("=" * 70)
print(" 医師視聴パターン分析: Intensive vs Extensive Margin")
print("=" * 70)

# 1. RW医師リスト
rw_list = pd.read_csv(os.path.join(DATA_DIR, FILE_RW_LIST))
doctor_master = rw_list[rw_list["seg"].notna() & (rw_list["seg"] != "")].copy()
doctor_master = doctor_master.rename(columns={"fac_honin": "facility_id", "doc": "doctor_id"})

# 2. 売上データ
sales_raw = pd.read_csv(os.path.join(DATA_DIR, FILE_SALES), dtype=str)
sales_raw["実績"] = pd.to_numeric(sales_raw["実績"], errors="coerce").fillna(0)
sales_raw["日付"] = pd.to_datetime(sales_raw["日付"], format="mixed")
daily = sales_raw[sales_raw["品目コード"].str.strip() == ENT_PRODUCT_CODE].copy()
daily = daily.rename(columns={
    "日付": "delivery_date",
    "施設（本院に合算）コード": "facility_id",
    "実績": "amount",
})

# 3. デジタル視聴データ
digital_raw = pd.read_csv(os.path.join(DATA_DIR, FILE_DIGITAL))
digital_raw["品目コード"] = digital_raw["品目コード"].astype(str).str.strip().str.zfill(5)
digital = digital_raw[digital_raw["品目コード"] == ENT_PRODUCT_CODE].copy()

# 4. 活動データ → web講演会のみ
activity_raw = pd.read_csv(os.path.join(DATA_DIR, FILE_ACTIVITY))
activity_raw["品目コード"] = activity_raw["品目コード"].astype(str).str.strip().str.zfill(5)
web_lecture = activity_raw[
    (activity_raw["品目コード"] == ENT_PRODUCT_CODE)
    & (activity_raw["活動種別"] == ACTIVITY_CHANNEL_FILTER)
].copy()

# 5. 視聴データ結合
common_cols = ["活動日_dt", "品目コード", "活動種別", "活動種別コード", "fac_honin", "doc"]
viewing = pd.concat([digital[common_cols], web_lecture[common_cols]], ignore_index=True)
viewing = viewing.rename(columns={
    "活動日_dt": "view_date",
    "fac_honin": "facility_id",
    "doc": "doctor_id",
    "活動種別": "channel_category",
})
viewing["view_date"] = pd.to_datetime(viewing["view_date"], format="mixed")

months = pd.date_range(start=START_DATE, periods=N_MONTHS, freq="MS")

print(f"\n[データ読み込み]")
print(f"  売上データ(ENT品目): {len(daily):,} 行")
print(f"  RW医師リスト(seg非空): {len(doctor_master)} 行")
print(f"  視聴データ結合: {len(viewing):,} 行")

# --- 除外フロー ---
print("\n[除外フロー]")
docs_per_fac = doctor_master.groupby("facility_id")["doctor_id"].nunique()
single_doc_facs = set(docs_per_fac[docs_per_fac == 1].index)

facs_per_doc = doctor_master.groupby("doctor_id")["facility_id"].nunique()
single_fac_docs = set(facs_per_doc[facs_per_doc == 1].index)

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
# Part 1: 医師レベルの視聴回数集計
# ================================================================
print("\n" + "=" * 70)
print(" Part 1: 医師レベルの視聴回数集計")
print("=" * 70)

# 処置群医師の全期間視聴回数
viewing_treated = viewing_after_washout[
    viewing_after_washout["doctor_id"].isin(treated_doc_ids)
].copy()

doc_view_counts = (
    viewing_treated.groupby("doctor_id")
    .size()
    .reset_index(name="total_views")
)

# 視聴頻度による医師分類
def classify_viewing_pattern(views):
    if views == 0:
        return "未視聴"
    elif views <= 2:
        return "単発視聴"
    else:
        return "定常視聴"

doc_view_counts["viewing_pattern"] = doc_view_counts["total_views"].apply(classify_viewing_pattern)

# 未視聴医師を追加
control_docs = pd.DataFrame({
    "doctor_id": list(control_doc_ids),
    "total_views": 0,
    "viewing_pattern": "未視聴"
})

all_doc_patterns = pd.concat([doc_view_counts, control_docs], ignore_index=True)
all_doc_patterns["facility_id"] = all_doc_patterns["doctor_id"].map(doc_to_fac)

pattern_dist = all_doc_patterns["viewing_pattern"].value_counts().sort_index()
print(f"\n[医師視聴パターン分類]")
print(f"  閾値: 単発視聴=1-2回, 定常視聴>={FREQUENT_THRESHOLD}回")
print(f"\n  分布:")
for pattern, count in pattern_dist.items():
    pct = count / len(all_doc_patterns) * 100
    print(f"    {pattern}: {count:>4}医師 ({pct:>5.1f}%)")

# 視聴回数の基本統計
viewing_docs = doc_view_counts[doc_view_counts["total_views"] > 0]
if len(viewing_docs) > 0:
    print(f"\n  視聴医師の視聴回数統計:")
    print(f"    平均: {viewing_docs['total_views'].mean():.1f}回")
    print(f"    中央値: {viewing_docs['total_views'].median():.0f}回")
    print(f"    最大: {viewing_docs['total_views'].max():.0f}回")
    print(f"    最小: {viewing_docs['total_views'].min():.0f}回")


# ================================================================
# Part 2: 施設×月次レベルのIntensive/Extensive指標構築
# ================================================================
print("\n" + "=" * 70)
print(" Part 2: Intensive/Extensive Margin指標構築")
print("=" * 70)

# 月次インデックスを追加
viewing_treated["month_index"] = (
    (viewing_treated["view_date"].dt.year - 2023) * 12
    + viewing_treated["view_date"].dt.month - 4
)

# 施設×月×医師の視聴回数
fac_month_doc_views = (
    viewing_treated.groupby(["facility_id", "month_index", "doctor_id"])
    .size()
    .reset_index(name="views_in_month")
)

# 各施設×月で、その月までに視聴したことがある医師のセット
cumulative_viewers = []
for fac_id in analysis_fac_ids:
    fac_views = fac_month_doc_views[
        fac_month_doc_views["facility_id"] == fac_id
    ].copy()

    cumulative_docs = set()
    for month_idx in range(N_MONTHS):
        month_docs = set(
            fac_views[fac_views["month_index"] == month_idx]["doctor_id"]
        )

        # Extensive: 新規視聴医師数（前月までに未視聴）
        new_docs = month_docs - cumulative_docs
        extensive = len(new_docs)

        # Intensive: 既存視聴医師の当月視聴回数平均
        existing_docs = month_docs & cumulative_docs
        if len(existing_docs) > 0:
            intensive_views = fac_views[
                (fac_views["month_index"] == month_idx)
                & (fac_views["doctor_id"].isin(existing_docs))
            ]["views_in_month"].sum()
            intensive = intensive_views / len(existing_docs)
        else:
            intensive = 0

        cumulative_viewers.append({
            "facility_id": fac_id,
            "month_index": month_idx,
            "extensive_margin": extensive,  # 新規視聴医師数
            "intensive_margin": intensive,  # 既存視聴医師の平均視聴回数
            "cumulative_viewers": len(cumulative_docs),  # 累積視聴医師数
        })

        cumulative_docs.update(new_docs)

margin_data = pd.DataFrame(cumulative_viewers)

print(f"\n  施設×月次 Margin指標:")
print(f"    Extensive Margin (新規視聴医師数) 平均: {margin_data['extensive_margin'].mean():.3f}")
print(f"    Extensive Margin 最大: {margin_data['extensive_margin'].max():.0f}")
print(f"    Intensive Margin (既存医師平均視聴回数) 平均: {margin_data[margin_data['intensive_margin'] > 0]['intensive_margin'].mean():.3f}")
print(f"    Intensive Margin 最大: {margin_data['intensive_margin'].max():.1f}")


# ================================================================
# Part 3: パネルデータ構築
# ================================================================
print("\n" + "=" * 70)
print(" Part 3: パネルデータ構築")
print("=" * 70)

daily_target = daily[daily["facility_id"].isin(analysis_fac_ids)].copy()
daily_target["month_index"] = (
    (daily_target["delivery_date"].dt.year - 2023) * 12
    + daily_target["delivery_date"].dt.month - 4
)

monthly = (
    daily_target.groupby(["facility_id", "month_index"])["amount"]
    .sum().reset_index()
)

full_idx = pd.MultiIndex.from_product(
    [sorted(analysis_fac_ids), list(range(N_MONTHS))],
    names=["facility_id", "month_index"]
)
panel = (
    monthly.set_index(["facility_id", "month_index"])
    .reindex(full_idx, fill_value=0).reset_index()
)

# Margin指標をマージ
panel = panel.merge(margin_data, on=["facility_id", "month_index"], how="left")
panel[["extensive_margin", "intensive_margin", "cumulative_viewers"]] = (
    panel[["extensive_margin", "intensive_margin", "cumulative_viewers"]].fillna(0)
)

# 処置群ダミー
panel["doctor_id"] = panel["facility_id"].map(fac_to_doc)
panel["unit_id"] = panel["facility_id"]

first_view_eligible = first_view[
    first_view["doctor_id"].isin(treated_doc_ids)
][["doctor_id", "first_view_month"]].copy()
first_view_eligible["facility_id"] = first_view_eligible["doctor_id"].map(doc_to_fac)

panel = panel.merge(
    first_view_eligible[["facility_id", "first_view_month"]],
    on="facility_id", how="left"
)
panel["treated"] = panel["first_view_month"].notna().astype(int)
panel["cohort_month"] = panel["first_view_month"]

# Post処置ダミー
mask_t = panel["cohort_month"].notna()
panel["post"] = 0
panel.loc[mask_t, "post"] = (
    panel.loc[mask_t, "month_index"] >= panel.loc[mask_t, "cohort_month"]
).astype(int)

print(f"  パネル行数: {len(panel):,}")
print(f"  処置群施設数: {panel[panel['treated'] == 1]['unit_id'].nunique()}")
print(f"  対照群施設数: {panel[panel['treated'] == 0]['unit_id'].nunique()}")

# 視聴パターン情報をマージ
panel = panel.merge(
    all_doc_patterns[["doctor_id", "viewing_pattern", "total_views"]],
    on="doctor_id", how="left"
)


# ================================================================
# Part 4: TWFE推定 (Intensive vs Extensive Margin)
# ================================================================
print("\n" + "=" * 70)
print(" Part 4: TWFE推定 (Intensive vs Extensive Margin)")
print("=" * 70)

# 処置後のみでIntensive/Extensiveの効果を推定
panel_post = panel[panel["post"] == 1].copy().reset_index(drop=True)

if len(panel_post) > 0:
    # NaN/Infをチェック・除去
    panel_post = panel_post.replace([np.inf, -np.inf], np.nan)
    panel_post = panel_post.dropna(subset=["amount", "intensive_margin", "extensive_margin"]).reset_index(drop=True)

    if len(panel_post) == 0:
        print("\n  警告: 処置後データにNaN/Infが多く、推定不可")
        beta_intensive = beta_extensive = 0
        se_intensive = se_extensive = 0
        pval_intensive = pval_extensive = 1.0
        sig_intensive = sig_extensive = "N/A"
    else:
        y = panel_post["amount"].values

        # 固定効果ダミー
        unit_dum = pd.get_dummies(panel_post["unit_id"], prefix="u", drop_first=True, dtype=float)
        time_dum = pd.get_dummies(panel_post["month_index"], prefix="t", drop_first=True, dtype=float)

        X = pd.concat([
            pd.DataFrame({
                "const": 1.0,
                "intensive": panel_post["intensive_margin"].values,
                "extensive": panel_post["extensive_margin"].values,
            }, index=panel_post.index),
            unit_dum,
            time_dum
        ], axis=1)

        # 最終的なNaN/Infチェック (X全体)
        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        y = y[X.index]
        panel_post = panel_post.loc[X.index]

        if len(X) == 0:
            print("\n  警告: 説明変数にNaN/Inf が多く、推定不可")
            beta_intensive = beta_extensive = 0
            se_intensive = se_extensive = 0
            pval_intensive = pval_extensive = 1.0
            sig_intensive = sig_extensive = "N/A"
        else:
            try:
                model = sm.OLS(y, X).fit(
                    cov_type="cluster", cov_kwds={"groups": panel_post["unit_id"].values}
                )

                beta_intensive = model.params["intensive"]
                se_intensive = model.bse["intensive"]
                pval_intensive = model.pvalues["intensive"]
                sig_intensive = (
                    "***" if pval_intensive < 0.001 else "**" if pval_intensive < 0.01
                    else "*" if pval_intensive < 0.05 else "n.s."
                )

                beta_extensive = model.params["extensive"]
                se_extensive = model.bse["extensive"]
                pval_extensive = model.pvalues["extensive"]
                sig_extensive = (
                    "***" if pval_extensive < 0.001 else "**" if pval_extensive < 0.01
                    else "*" if pval_extensive < 0.05 else "n.s."
                )

                print(f"\n  === Intensive vs Extensive Margin ===")
                print(f"  Intensive Margin (既存医師への追加視聴):")
                print(f"    係数: {beta_intensive:.3f}")
                print(f"    SE:   {se_intensive:.3f}")
                print(f"    p値:  {pval_intensive:.6f} {sig_intensive}")

                print(f"\n  Extensive Margin (新規医師獲得):")
                print(f"    係数: {beta_extensive:.3f}")
                print(f"    SE:   {se_extensive:.3f}")
                print(f"    p値:  {pval_extensive:.6f} {sig_extensive}")

                # 効果の比較
                if beta_intensive > beta_extensive:
                    print(f"\n  → 【深さ】既存医師への複数回視聴が効果的 (β1={beta_intensive:.3f} > β2={beta_extensive:.3f})")
                elif beta_extensive > beta_intensive:
                    print(f"\n  → 【広さ】視聴医師層の拡大が効果的 (β2={beta_extensive:.3f} > β1={beta_intensive:.3f})")
                else:
                    print(f"\n  → 同程度の効果")

            except Exception as e:
                print(f"\n  警告: 回帰推定でエラー: {e}")
                beta_intensive = beta_extensive = 0
                se_intensive = se_extensive = 0
                pval_intensive = pval_extensive = 1.0
                sig_intensive = sig_extensive = "N/A"
else:
    print("\n  警告: 処置後データなし")
    beta_intensive = beta_extensive = 0
    se_intensive = se_extensive = 0
    pval_intensive = pval_extensive = 1.0
    sig_intensive = sig_extensive = "N/A"


# ================================================================
# Part 5: 視聴パターン別の実績推移比較
# ================================================================
print("\n" + "=" * 70)
print(" Part 5: 視聴パターン別の実績推移比較")
print("=" * 70)

# 各グループの平均実績推移
pattern_trends = (
    panel.groupby(["month_index", "viewing_pattern"])["amount"]
    .mean().unstack(fill_value=0)
)

print(f"\n  視聴パターン別 平均納入額 (全期間):")
for pattern in ["未視聴", "単発視聴", "定常視聴"]:
    if pattern in pattern_trends.columns:
        mean_val = pattern_trends[pattern].mean()
        print(f"    {pattern}: {mean_val:.1f}")

# 処置後の平均実績
post_period = panel[panel["month_index"] >= WASHOUT_MONTHS]
post_pattern_means = post_period.groupby("viewing_pattern")["amount"].mean()
print(f"\n  視聴パターン別 平均納入額 (wash-out後):")
for pattern in ["未視聴", "単発視聴", "定常視聴"]:
    if pattern in post_pattern_means.index:
        print(f"    {pattern}: {post_pattern_means[pattern]:.1f}")


# ================================================================
# Part 6: 可視化
# ================================================================
print("\n" + "=" * 70)
print(" Part 6: 可視化")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(
    "医師視聴パターン分析: Intensive vs Extensive Margin",
    fontsize=13, fontweight="bold"
)

# (a) 視聴パターン別の実績推移
ax = axes[0, 0]
colors = {"未視聴": "#1565C0", "単発視聴": "#FF9800", "定常視聴": "#4CAF50"}
for pattern in ["未視聴", "単発視聴", "定常視聴"]:
    if pattern in pattern_trends.columns:
        ax.plot(
            pattern_trends.index,
            pattern_trends[pattern],
            marker="o", ms=3, label=pattern,
            color=colors.get(pattern, "gray")
        )

ax.axvline(WASHOUT_MONTHS - 0.5, color="gray", ls=":", lw=0.8, label="wash-out")
ax.set_xlabel("月 (0=2023/4)")
ax.set_ylabel("平均納入額")
ax.set_title("(a) 視聴パターン別 実績推移")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (b) Intensive/Extensive Margin の時系列
ax = axes[0, 1]
treated_panel = panel[panel["treated"] == 1].copy()
margin_agg = treated_panel.groupby("month_index")[
    ["intensive_margin", "extensive_margin"]
].mean()

ax2 = ax.twinx()
ax.bar(margin_agg.index, margin_agg["extensive_margin"],
       alpha=0.6, color="#FF6F00", label="Extensive (新規医師数)")
ax2.plot(margin_agg.index, margin_agg["intensive_margin"],
         marker="o", ms=4, color="#1565C0", label="Intensive (平均視聴回数)")

ax.set_xlabel("月 (0=2023/4)")
ax.set_ylabel("Extensive Margin (新規医師数)", color="#FF6F00")
ax2.set_ylabel("Intensive Margin (視聴回数)", color="#1565C0")
ax.set_title("(b) Intensive/Extensive Margin 時系列")
ax.tick_params(axis='y', labelcolor="#FF6F00")
ax2.tick_params(axis='y', labelcolor="#1565C0")
ax.grid(True, alpha=0.2)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

# (c) 医師視聴回数分布
ax = axes[1, 0]
if len(viewing_docs) > 0:
    ax.hist(viewing_docs["total_views"], bins=20, color="#4CAF50",
            alpha=0.7, edgecolor="white")
    ax.axvline(FREQUENT_THRESHOLD - 0.5, color="red", ls="--", lw=1.5,
               label=f"定常視聴閾値 ({FREQUENT_THRESHOLD}回)")
    ax.set_xlabel("視聴回数")
    ax.set_ylabel("医師数")
    ax.set_title(f"(c) 医師視聴回数分布 (N={len(viewing_docs)})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

# (d) 視聴パターン別の施設数
ax = axes[1, 1]
pattern_counts = all_doc_patterns["viewing_pattern"].value_counts().sort_index()
bars = ax.bar(
    range(len(pattern_counts)),
    pattern_counts.values,
    color=[colors.get(p, "gray") for p in pattern_counts.index],
    alpha=0.8, edgecolor="white"
)

for bar, val in zip(bars, pattern_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
            str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_xticks(range(len(pattern_counts)))
ax.set_xticklabels(pattern_counts.index)
ax.set_ylabel("医師数")
ax.set_title("(d) 視聴パターン別 医師数分布")
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
out_path = os.path.join(SCRIPT_DIR, "physician_viewing_analysis.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  図を保存: {out_path}")


# ================================================================
# Part 7: 結論
# ================================================================
print("\n" + "=" * 70)
print(" 結論")
print("=" * 70)

print(f"""
  === 医師視聴パターン分類 ===
  未視聴群     : {pattern_dist.get('未視聴', 0)} 医師
  単発視聴群   : {pattern_dist.get('単発視聴', 0)} 医師 (1-2回)
  定常視聴群   : {pattern_dist.get('定常視聴', 0)} 医師 ({FREQUENT_THRESHOLD}回以上)

  === Intensive vs Extensive Margin ===
  Intensive Margin (既存医師への追加視聴):
    係数 = {beta_intensive:.3f} (SE={se_intensive:.3f}, {sig_intensive})

  Extensive Margin (新規医師獲得):
    係数 = {beta_extensive:.3f} (SE={se_extensive:.3f}, {sig_extensive})

  === 戦略的示唆 ===""")

if beta_intensive > beta_extensive * 1.2:
    print(f"  → 【深さ重視】同じ医師への複数回アプローチが効果的")
    print(f"     既存視聴医師への継続的な視聴促進を優先すべき")
elif beta_extensive > beta_intensive * 1.2:
    print(f"  → 【広さ重視】視聴医師層の拡大が効果的")
    print(f"     未視聴医師への初回視聴促進を優先すべき")
else:
    print(f"  → 【バランス型】両方の戦略を並行して実施すべき")

print(f"""
  === 視聴パターン別実績 (wash-out後平均) ===""")
for pattern in ["未視聴", "単発視聴", "定常視聴"]:
    if pattern in post_pattern_means.index:
        print(f"  {pattern}: {post_pattern_means[pattern]:.1f}")


# ================================================================
# JSON結果保存
# ================================================================
results_dir = os.path.join(SCRIPT_DIR, "results")
os.makedirs(results_dir, exist_ok=True)

results_json = {
    "viewing_pattern_distribution": {
        pattern: int(count) for pattern, count in pattern_dist.items()
    },
    "viewing_statistics": {
        "mean": float(viewing_docs["total_views"].mean()) if len(viewing_docs) > 0 else 0,
        "median": float(viewing_docs["total_views"].median()) if len(viewing_docs) > 0 else 0,
        "max": int(viewing_docs["total_views"].max()) if len(viewing_docs) > 0 else 0,
        "min": int(viewing_docs["total_views"].min()) if len(viewing_docs) > 0 else 0,
    },
    "margin_analysis": {
        "intensive_margin": {
            "coefficient": float(beta_intensive),
            "se": float(se_intensive),
            "p": float(pval_intensive),
            "sig": sig_intensive,
        },
        "extensive_margin": {
            "coefficient": float(beta_extensive),
            "se": float(se_extensive),
            "p": float(pval_extensive),
            "sig": sig_extensive,
        },
        "recommendation": (
            "深さ重視" if beta_intensive > beta_extensive * 1.2
            else "広さ重視" if beta_extensive > beta_intensive * 1.2
            else "バランス型"
        ),
    },
    "pattern_means": {
        "all_period": {
            pattern: float(pattern_trends[pattern].mean())
            for pattern in pattern_trends.columns
        },
        "post_washout": {
            pattern: float(post_pattern_means[pattern])
            for pattern in post_pattern_means.index
        },
    },
    "margin_time_series": {
        "intensive": margin_agg["intensive_margin"].to_dict(),
        "extensive": margin_agg["extensive_margin"].to_dict(),
    },
}

json_path = os.path.join(results_dir, "physician_viewing_analysis.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(results_json, f, ensure_ascii=False, indent=2)
print(f"\n  結果をJSON保存: {json_path}")

print("\n" + "=" * 70)
print(" 分析完了")
print("=" * 70)
