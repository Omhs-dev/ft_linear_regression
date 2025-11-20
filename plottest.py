# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
import predict

# x_values = [1, 2, 3, 4]
# y_values = [1, 2, 2.5, 4]
# data = pd.read_csv("data.csv")

# x = data["km"]
# y = data["price"]

# plt.scatter(x, y)
# plt.xlabel("Mileage")
# plt.ylabel("Price")

# x = np.array(x)
# y = np.array(y)

# # In my case I'm suppose to use my Gradient Descent algo
# w, b = np.polyfit(x, y, 1)
# print(w)
# print(b)

# plt.plot(x, predict.prediction(x, b, w), color="red")
# plt.show()


# save as gradient_50.py and run: python3 gradient_50.py

# X = [1, 2, 3, 4]
# Y = [1, 2, 2.5, 4]
X = [240000.0, 139800.0, 150500.0, 185530.0, 176000.0, 114800.0, 166800.0, 89000.0, 144500.0, 84000.0, 82029.0, 63060.0, 74000.0, 97500.0, 67000.0, 76025.0, 48235.0, 93000.0, 60949.0, 65674.0, 54000.0, 68500.0, 22899.0, 61789.0]
Y = [3650.0, 3800.0, 4400.0, 4450.0, 5250.0, 5350.0, 5800.0, 5990.0, 5999.0, 6200.0, 6390.0, 6390.0, 6600.0, 6800.0, 6800.0, 6900.0, 6900.0, 6990.0, 7490.0, 7555.0, 7990.0, 7990.0, 7990.0, 8290.0]

m = len(X)

# Min-max normalization
xmin, xmax = min(X), max(X)
ymin, ymax = min(Y), max(Y)
range_x = xmax - xmin
range_y = ymax - ymin
Xn = [(x - xmin) / range_x for x in X]
Yn = [(y - ymin) / range_y for y in Y]

def compute_cost_norm(w, b):
    total = 0.0
    for xi, yi in zip(Xn, Yn):
        y_pred = w * xi + b
        total += (y_pred - yi) ** 2
    return total / (2 * m)

def gradients_norm(w, b):
    dw = 0.0
    db = 0.0
    for xi, yi in zip(Xn, Yn):
        y_pred = w * xi + b
        dw += (y_pred - yi) * xi
        db += (y_pred - yi)
    return dw / m, db / m

# gradient descent parameters
w = 0.0   # normalized theta1
b = 0.0   # normalized theta0
lr = 0.01
max_iter = 1000

rows = []
for it in range(max_iter + 1):  # include iteration 0..50
    cost_norm = compute_cost_norm(w, b)

    # denormalize parameters (min-max)
    W = w * (range_y / range_x)    # slope in original units
    B = b * range_y + ymin - W * xmin

    # cost on original data using denormalized params
    total_orig = 0.0
    for xi, yi in zip(X, Y):
        y_pred_orig = W * xi + B
        total_orig += (y_pred_orig - yi) ** 2
    cost_orig = total_orig / (2 * m)

    rows.append((it, w, b, cost_norm, W, B, cost_orig))

    # batch update
    dw, db = gradients_norm(w, b)
    w = w - lr * dw
    b = b - lr * db

# Print nicely
print("Iter |   w_norm    |   b_norm    | cost_norm  |  W_denorm   |  B_denorm   | cost_orig")
print("-" * 85)
for it, w, b, cost_norm, W, B, cost_orig in rows:
    print(f"{it:4d} | {w:11.8f} | {b:11.8f} | {cost_norm:10.8f} | {W:11.8f} | {B:11.8f} | {cost_orig:10.8f}")
