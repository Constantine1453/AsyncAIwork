import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

data = {
    "姓名": ["A", "B", "C", "D", "E"],
    "Python": [85, 92, 78, 96, 88],
    "数学": [76, 89, 82, 95, 80],
    "英语": [90, 85, 75, 91, 86]
}

df = pd.DataFrame(data)
df["平均分"] = df[["Python", "数学", "英语"]].mean(axis=1)

print(df[["姓名", "平均分"]])

plt.figure(figsize=(9, 6))
plt.plot(df["姓名"], df["平均分"], marker="o", linewidth=2, color="steelblue")
plt.title("学生平均成绩")
plt.xlabel("学生")
plt.ylabel("平均分")
plt.ylim(0, 100)
plt.grid(alpha=0.3)
for i, v in enumerate(df["平均分"]):
    plt.text(i, v + 2, f"{v:.2f}", ha="center", fontsize=10)

plt.savefig("t1-3_折线图.png", dpi=150, bbox_inches="tight")
plt.show()
