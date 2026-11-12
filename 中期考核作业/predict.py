"""
妖怪之山魔力值预测

思路：
  1. 每年同一天的值求平均，得到一条年周期曲线
  2. 最近一年整体比历史平均高/低多少，按一定比例挪一下
  3. 周末比工作日高一点，补上
  4. 预测区间用验证时的误差来估

运行：python predict.py
输出：forecast.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

DEN = 76.87          # MASE 的分母：训练集上 365 日季节朴素的平均绝对误差
STRENGTH = 0.4       # 水平修正的强度，试出来的，见报告
SMOOTH = 9           # 年周期曲线的平滑窗口
YEARS = None         # None 表示用全部年份
ORIGINS = ["2022-12-31", "2023-12-31", "2024-12-31", "2025-06-30"]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def describe(tr):
    print("按月平均：")
    m = tr.groupby(tr["date"].dt.month)["magic"].mean()
    for k, v in m.items():
        print("  %2d 月  %6.1f" % (k, v))
    print("每年 1-6 月平均：")
    t = tr[tr["date"].dt.month <= 6].groupby(tr["date"].dt.year)["magic"].mean()
    for k, v in t.items():
        print("  %d  %6.1f" % (k, v))
    print("上半年最低 %.1f，最高 %.1f，相差 %.1f" % (t.min(), t.max(), t.max() - t.min()))


def bias_check(tr):
    """能不能给所有预测统一补一个常数？"""
    print("如果给每个起点补一个最优常数：")
    bs, cs = [], []
    for o in ORIGINS:
        origin = pd.Timestamp(o)
        m = (tr["date"] <= origin).to_numpy()
        hd = tr.loc[m, "date"].reset_index(drop=True)
        hv = tr.loc[m, "magic"].to_numpy()
        mask = (tr["date"] > origin) & (tr["date"] <= origin + pd.Timedelta(days=180))
        td = tr.loc[mask, "date"].reset_index(drop=True)
        tv = tr.loc[mask, "magic"].to_numpy()
        pred, _, _ = fit_predict(hd, hv, td)
        h = np.arange(1, len(tv) + 1)

        def wm(b):
            v = []
            for lo, hi in [(1, 7), (1, 30), (1, 90), (1, 180)]:
                k = (h >= lo) & (h <= hi)
                v.append(mase(tv[k], pred[k] + b))
            return v[0] * 0.15 + v[1] * 0.25 + v[2] * 0.25 + v[3] * 0.35

        cands = np.arange(-80, 81, 1)
        bestc = min(cands, key=wm)
        print("  %s  原 %.3f  最优常数 %+d  修正后 %.3f"
              % (o, wm(0), bestc, wm(bestc)))
        bs.append(wm(0))
        cs.append(wm(bestc))
    print("  平均  原 %.3f  修正后 %.3f" % (np.mean(bs), np.mean(cs)))


def load():
    tr = pd.read_csv(DATA / "magic_train.csv", parse_dates=["date"])
    fu = pd.read_csv(DATA / "forecast_dates.csv", parse_dates=["date"])
    return tr, fu


def day_profile(dates, values):
    """按一年中的第几天求均值，再做平滑，去掉单日噪声。"""
    s = pd.Series(values, index=pd.DatetimeIndex(dates))
    if YEARS:
        s = s[s.index.year > s.index.year.max() - YEARS]
    p = s.groupby(s.index.day_of_year).mean()
    # 2 月 29 号只有闰年有，补齐再插值
    p = p.reindex(range(1, 367)).interpolate()
    p = p.rolling(SMOOTH, center=True, min_periods=1).mean()
    p.iloc[-1] = p.iloc[-2]
    return p


def fit_predict(hist_dates, hist_values, target_dates):
    prof = day_profile(hist_dates, hist_values)

    doy_hist = hist_dates.dt.day_of_year.clip(upper=366)
    resid = hist_values - prof.reindex(doy_hist).to_numpy()

    # 最近一年的整体水平和这条曲线差多少
    offset = hist_values[-365:].mean() - prof.reindex(doy_hist).to_numpy()[-365:].mean()

    # 星期几的影响
    dow_effect = pd.Series(resid).groupby(hist_dates.dt.day_of_week).mean()
    dow_effect = dow_effect - dow_effect.mean()

    doy_t = target_dates.dt.day_of_year.clip(upper=366)
    pred = prof.reindex(doy_t).to_numpy() + STRENGTH * offset
    pred = pred + dow_effect.reindex(target_dates.dt.day_of_week).to_numpy()

    return pred, prof, offset


def mase(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred)) / DEN


def backtest(tr):
    print("滚动起点验证")
    print("%-12s %8s %8s %8s %8s %8s" % ("起点", "1-7", "1-30", "1-90", "1-180", "加权"))
    rows = []
    all_err = []
    for o in ORIGINS:
        origin = pd.Timestamp(o)
        m = (tr["date"] <= origin).to_numpy()
        hd = tr.loc[m, "date"].reset_index(drop=True)
        hv = tr.loc[m, "magic"].to_numpy()

        mask = (tr["date"] > origin) & (tr["date"] <= origin + pd.Timedelta(days=180))
        td = tr.loc[mask, "date"].reset_index(drop=True)
        tv = tr.loc[mask, "magic"].to_numpy()
        if len(tv) == 0:
            continue

        pred, _, _ = fit_predict(hd, hv, td)
        h = np.arange(1, len(tv) + 1)

        vals = []
        for lo, hi in [(1, 7), (1, 30), (1, 90), (1, 180)]:
            k = (h >= lo) & (h <= hi)
            vals.append(mase(tv[k], pred[k]))
        w = vals[0] * 0.15 + vals[1] * 0.25 + vals[2] * 0.25 + vals[3] * 0.35
        print("%-12s %8.2f %8.2f %8.2f %8.2f %8.3f" % (o, vals[0], vals[1], vals[2], vals[3], w))
        rows.append([o] + vals + [w])
        all_err.append(tv - pred)

    err = np.concatenate(all_err)
    print("合并残差：均值 %.2f，标准差 %.2f" % (err.mean(), err.std()))
    return pd.DataFrame(rows, columns=["origin", "w7", "w30", "w90", "w180", "weighted"]), err


def error_by_month(tr):
    """看看一年里哪些月份误差大。"""
    prof = day_profile(tr["date"], tr["magic"].to_numpy())
    doy = tr["date"].dt.day_of_year.clip(upper=366)
    resid = tr["magic"].to_numpy() - prof.reindex(doy).to_numpy()
    s = pd.Series(np.abs(resid)).groupby(tr["date"].dt.month).mean()
    print("各月平均绝对残差：")
    for m, v in s.items():
        print("  %2d 月  %5.1f" % (m, v))
    return s


def avg_mase(tr):
    """四个起点的加权 MASE 平均。"""
    ws = []
    for o in ORIGINS:
        origin = pd.Timestamp(o)
        m = (tr["date"] <= origin).to_numpy()
        hd = tr.loc[m, "date"].reset_index(drop=True)
        hv = tr.loc[m, "magic"].to_numpy()
        mask = (tr["date"] > origin) & (tr["date"] <= origin + pd.Timedelta(days=180))
        td = tr.loc[mask, "date"].reset_index(drop=True)
        tv = tr.loc[mask, "magic"].to_numpy()
        if len(tv) == 0:
            continue
        pred, _, _ = fit_predict(hd, hv, td)
        h = np.arange(1, len(tv) + 1)
        v = []
        for lo, hi in [(1, 7), (1, 30), (1, 90), (1, 180)]:
            k = (h >= lo) & (h <= hi)
            v.append(mase(tv[k], pred[k]))
        ws.append(v[0] * 0.15 + v[1] * 0.25 + v[2] * 0.25 + v[3] * 0.35)
    return np.mean(ws)


def tune_profile(tr):
    """平滑窗口和用几年的数据，也试一下。"""
    global SMOOTH, YEARS
    print("平滑窗口：")
    for v in [1, 5, 7, 9, 11, 15, 21]:
        SMOOTH = v
        print("  %2d 天  %.3f" % (v, avg_mase(tr)))
    SMOOTH = 9
    print("数据范围：")
    for y, lab in [(3, "近3年"), (5, "近5年"), (None, "全部10年")]:
        YEARS = y
        print("  %-6s %.3f" % (lab, avg_mase(tr)))
    YEARS = None


def tune_strength(tr):
    """修正强度试了几档，看哪一档之后不再变好。"""
    global STRENGTH
    old = STRENGTH
    print("试不同的修正强度（四个起点加权 MASE 平均）：")
    for s in [0.0, 0.2, 0.3, 0.4, 0.5]:
        STRENGTH = s
        ws = []
        for o in ORIGINS:
            origin = pd.Timestamp(o)
            m = (tr["date"] <= origin).to_numpy()
            hd = tr.loc[m, "date"].reset_index(drop=True)
            hv = tr.loc[m, "magic"].to_numpy()
            mask = (tr["date"] > origin) & (tr["date"] <= origin + pd.Timedelta(days=180))
            td = tr.loc[mask, "date"].reset_index(drop=True)
            tv = tr.loc[mask, "magic"].to_numpy()
            if len(tv) == 0:
                continue
            pred, _, _ = fit_predict(hd, hv, td)
            h = np.arange(1, len(tv) + 1)
            vals = []
            for lo, hi in [(1, 7), (1, 30), (1, 90), (1, 180)]:
                k = (h >= lo) & (h <= hi)
                vals.append(mase(tv[k], pred[k]))
            ws.append(vals[0] * 0.15 + vals[1] * 0.25 + vals[2] * 0.25 + vals[3] * 0.35)
        print("  强度 %.1f  平均加权 MASE %.3f" % (s, np.mean(ws)))
    STRENGTH = old


def make_plots(tr, fu, pred):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(tr["date"], tr["magic"], lw=0.5, color="#2b6cb0")
    ax.axvline(fu["date"].min(), color="#c53030", ls="--")
    ax.set_title("十年的日魔力值")
    ax.set_ylabel("魔力值 (LU)")
    fig.tight_layout()
    fig.savefig(HERE / "images" / "图1_十年曲线.png", dpi=130)
    plt.close(fig)

    yearly = tr.groupby(tr["date"].dt.year)["magic"].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(yearly.index.astype(str), yearly.values, color="#4a7ebb")
    ax.axhline(tr["magic"].mean(), color="#c53030", ls="--", label="十年总平均")
    ax.set_title("每年的平均魔力值")
    ax.set_ylabel("LU")
    ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / "images" / "图2_每年平均水平.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(tr["date"][-365:], tr["magic"][-365:], lw=1, label="2025 实际")
    ax.plot(fu["date"], pred, lw=1.5, color="#c53030", label="2026 预测")
    ax.set_title("最近一年与未来 180 天预测")
    ax.set_ylabel("LU")
    ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / "images" / "图3_预测结果.png", dpi=130)
    plt.close(fig)
    print("图片已保存到 images/")


def main():
    tr, fu = load()
    print("训练集 %d 行，%s ~ %s" % (len(tr), tr["date"].min().date(), tr["date"].max().date()))
    print("需要预测 %d 天，%s ~ %s" % (len(fu), fu["date"].min().date(), fu["date"].max().date()))
    print()
    describe(tr)
    print()

    table, err = backtest(tr)
    print()
    print("四个起点加权 MASE 平均：%.3f" % table["weighted"].mean())
    print()
    error_by_month(tr)
    print()
    tune_strength(tr)
    print()
    tune_profile(tr)
    print()
    bias_check(tr)

    # 区间宽度：用验证残差的标准差
    k = 1.28
    width = k * err.std()
    print("区间半宽 = %.1f LU（%.2f × %.1f）" % (width, k, err.std()))

    # 用全部数据重新拟合，预测未来
    pred, prof, offset = fit_predict(tr["date"], tr["magic"].to_numpy(), fu["date"])
    print("水平修正量 = %.1f LU（强度 %.1f）" % (offset, STRENGTH))

    # 星期效应到底有多大
    resid_all = tr["magic"].to_numpy() - prof.reindex(
        tr["date"].dt.day_of_year.clip(upper=366)).to_numpy()
    dow_all = pd.Series(resid_all).groupby(tr["date"].dt.day_of_week).mean()
    print("星期效应范围 = %.1f ~ %.1f LU" % (dow_all.min(), dow_all.max()))
    print("预测范围 %.1f ~ %.1f LU" % (pred.min(), pred.max()))

    out = pd.DataFrame({
        "date": fu["date"].dt.strftime("%Y-%m-%d"),
        "y_pred": np.round(pred, 3),
        "y_lower": np.round(pred - width, 3),
        "y_upper": np.round(pred + width, 3),
    })
    out.to_csv(HERE / "forecast.csv", index=False)
    print("已写入 forecast.csv，共 %d 行" % len(out))

    make_plots(tr, fu, pred)

    # 验证集的覆盖率（顺便看一下区间定得准不准）
    print()
    print("说明：区间宽度按验证残差定，训练集内覆盖率约 %.1f%%" % (
        100 * np.mean(np.abs(err) <= width)))


if __name__ == "__main__":
    main()
