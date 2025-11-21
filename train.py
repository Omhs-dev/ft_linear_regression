import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import traceback

import predict

def print_result(t0, t1, td0, td1, cost):
	results = [
		("Theta0_norm", t0),
		("Theta1_norm", t1),
		("Theta0_denorm", td0),
		("Theta1_denorm", td1),
		("Final Cost", cost),
	]
	print("-" * 49)
	for label, val in results:
		print(f"{label}\t|\t{val}\t|")
		print("-" * 49)

def set_thetas(theta0, theta1):
	try:
		with open("values.json", "w") as json_file:
			json.dump({"Theta0": theta0, "Theta1": theta1}, json_file)
	except FileNotFoundError:
		print("Error: file 'value.json' not found")
		return None
	except json.JSONDecodeError:
		print("Error: failed to decode JSON file")
		return None

def normalize(data):
	data_range = max(data) - min(data)
	data_norm = [(val - min(data)) / data_range for val in data]
	return data_norm

def denormalize(theta0, theta1, x_data, y_data):
	x_range = max(x_data) - min(x_data)
	y_range = max(y_data) - min(y_data)
	dernorm_t1 = theta1 * ( y_range / x_range)
	dernorm_t0 = theta0 * (y_range) + min(y_data) - dernorm_t1 * min(x_data)
	return float(dernorm_t0), float(dernorm_t1)

def error(w, b, x, y):
	return (b + w*x) - y

def w_theta1_gradient(w, b, x_data, y_data):
	m = len(x_data)
	errors_sum = 0
	for i in range(0, len(x_data)):
		errors_sum += error(w, b, x_data[i], y_data[i]) * x_data[i]
	return (1/m) * errors_sum

def b_theta0_gradient(w, b, x_data, y_data):
	m = len(x_data)
	errors_sum = 0
	for i in range(0, len(x_data)):
		errors_sum += error(w, b, x_data[i], y_data[i])
	return (1/m) *errors_sum

def gradient_update_rule(theta1, theta0, x_data, y_data):
	delta_lr = 0.1
	theta1_grad = w_theta1_gradient(theta1, theta0, x_data, y_data)
	theta0_grad = b_theta0_gradient(theta1, theta0, x_data, y_data)
	theta0 -= (delta_lr * theta0_grad)
	theta1 -= (delta_lr * theta1_grad)
	return (theta0, theta1)

def squared_error(w, b, x_data, y_data):
	res = 0
	for i in range(0, len(x_data)):
		res += ((error(w, b, x_data[i], y_data[i]))**2)
	return res

def cost_function(w, b, x_data, y_data):
	m = len(x_data)
	sq_err_sum = squared_error(w, b, x_data, y_data)
	cost = (1/(2 * m)) * sq_err_sum
	return cost

def launch_train(theta1, theta0, x_data, y_data):
	iter_num = 1000

	cost_history = []
	for i in range(iter_num):
		curr_cost = cost_function(theta1, theta0, x_data, y_data)
		cost_history.append(curr_cost)
		if i > 0:
			theta0, theta1 = gradient_update_rule(theta1, theta0, x_data, y_data)
	print(f"Iteration {i} \ncost = {curr_cost}\n")
	return theta0, theta1, iter_num, cost_history

def visualize_regression(theta0, theta1, x_data, y_data):
	x_data = np.array(x_data)
	y_data = np.array(y_data)
	prediction = predict.prediction(x_data, theta0, theta1)
	plt.scatter(x_data, y_data)
	plt.xlabel("Mileage(km)")
	plt.ylabel("Price(Euro)")
	plt.plot(x_data, prediction, color="red")
	plt.show()

def visualize_cost(iter_n, cost_history):
	plt.plot(range(iter_n), cost_history)
	plt.xlabel("Iterations")
	plt.ylabel("Cost")
	plt.show()

def main():
	theta0 = 0
	theta1 = 0
	try:
		data = pd.read_csv("data.csv")
		x_mileage = data["km"].tolist()
		y_price = data["price"].tolist()
		x_mileage_n = normalize(x_mileage)
		y_price_n = normalize(y_price)
		train_model = launch_train(theta1, theta0, x_mileage_n, y_price_n)
		theta0_n, theta1_n, iterations, cost_hist = train_model
		theta0_d, theta1_d = denormalize(theta0_n, theta1_n, x_mileage, y_price)
		print_result(theta0_n, theta1_n, theta0_d, theta1_d, cost_hist[-1])
		set_thetas(theta0_d, theta1_d)
		# visualize_regression(theta0_d, theta1_d, x_mileage, y_price)
		visualize_cost(iterations, cost_hist)
	except FileNotFoundError as e:
		print(f"Error: File not found. Details: {e}")
	except KeyError as e:
		print(f"Error: Missing column. Details: {e}")
	except Exception as e:
		print(f"An unexpected error occurred: {e}")
		traceback.print_exc()

if __name__ == "__main__":
	main()