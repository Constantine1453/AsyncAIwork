import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

data = np.loadtxt("data1.csv", delimiter="\t")
x = data[:, 0]
y = data[:, 1]

w = 0.0
b = 0.0
lr = 0.005
epochs = 8000
losses = []

for i in range(epochs):
    y_pred = w * x + b
    error = y_pred - y
    loss = np.mean(error ** 2)
    dw = np.mean(2 * error * x)
    db = np.mean(2 * error)
    w = w - lr * dw
    b = b - lr * db
    losses.append(loss)
    if (i + 1) % 2000 == 0:
        print(f"第 {i + 1:5d} 轮  损失 = {loss:.4f}  w = {w:.4f}  b = {b:.4f}")

print("\n训练完成")
print(f"样本数   : {len(x)}")
print(f"学习率   : {lr}")
print(f"训练轮数 : {epochs}")
print(f"w = {w:.4f}")
print(f"b = {b:.4f}")
print(f"最终损失 : {losses[-1]:.4f}")

plt.figure(figsize=(8, 6))
plt.scatter(x, y, label="真实数据")
plt.plot(x, w * x + b, color="red", linewidth=2,
         label=f"拟合直线  y = {w:.4f}x + {b:.4f}")
plt.xlabel("x")
plt.ylabel("y")
plt.title("梯度下降线性回归拟合结果")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("t2-1_拟合.png", dpi=150, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(losses, color="darkorange", linewidth=2)
plt.xlabel("训练轮数")
plt.ylabel("损失（均方误差）")
plt.title("损失下降曲线")
plt.grid(alpha=0.3)
plt.savefig("t2-1_损失.png", dpi=150, bbox_inches="tight")
plt.show()
