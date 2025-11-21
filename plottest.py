import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import predict

x_values = [1, 2, 3, 4]
y_values = [1, 2, 2.5, 4]
data = pd.read_csv("data.csv")

x = data["km"]
y = data["price"]

plt.scatter(x, y)
plt.xlabel("Mileage")
plt.ylabel("Price")

x = np.array(x)
y = np.array(y)

# In my case I'm suppose to use my Gradient Descent algo
w, b = np.polyfit(x, y, 1)
print(w)
print(b)

plt.plot(x, predict.prediction(x, b, w), color="red")
plt.show()
