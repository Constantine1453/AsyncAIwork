import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

data = {
    "姓名": ["A", "B", "C", "D", "E", "Z"],
    "Python": [85, 92, 78, 96, 88, 80],
    "数学": [76, 89, 82, 95, 80, 85],
    "英语": [90, 85, 75, 91, 86, 78]
}

df = pd.DataFrame(data)

df["平均分"] = df[["Python", "数学", "英语"]].mean(axis=1)

print("学生成绩：")
print(df)

print("\n平均成绩：")
print(df[["姓名", "平均分"]])

print("\n全班平均分：", round(df["平均分"].mean(), 2))

fig = plt.figure(figsize=(9, 9))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.5], hspace=0.35)

ax1 = fig.add_subplot(gs[0])
ax1.axis("off")
show = df.copy()
show["平均分"] = show["平均分"].round(2)
tb = ax1.table(cellText=show.values, colLabels=show.columns,
               loc="center", cellLoc="center")
tb.auto_set_font_size(False)
tb.set_fontsize(12)
tb.scale(1, 2.0)
ax1.set_title("学生成绩汇总（新增学生 Z）", fontsize=15, pad=12)

ax2 = fig.add_subplot(gs[1])
bars = ax2.bar(df["姓名"], df["平均分"])
bars[-1].set_color("tomato")
ax2.set_title("学生平均成绩")
ax2.set_xlabel("学生")
ax2.set_ylabel("平均分")
ax2.set_ylim(0, 100)
for i, v in enumerate(df["平均分"]):
    ax2.text(i, v + 1.5, f"{v:.2f}", ha="center", fontsize=10)

plt.savefig("t1-2_结果.png", dpi=150, bbox_inches="tight")
plt.show()
