import pandas as pd
import matplotlib.pyplot as plt
import json
import csv

import predict

def load_data():
	x_mileage = 0
	y_price = 0
	dataset = []
	try:
		with open("data.csv", 'r') as file:
			csv_reader = csv.reader(file)

			next(csv_reader)
			for line in csv_reader:
				# print(line)
				try:
					dataset.append([float(value) for value in line])
				except ValueError:
					print("Error: value is not a valide number")
					return None
			x_mileage = [row[0] for row in dataset]
			y_price = [row[1] for row in dataset]

			return (x_mileage, y_price)
	except FileNotFoundError:
		print("Error: file no found")
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



x_data, y_data = load_data()
t0, t1 = denormalize(0.3832180235039033, 0.07276542047951033, x_data, y_data)
print(f"dernorm_t0: {t0} --- denorm_t1: {t1}")

def error(w, b, x, y):
	return (b + w*x) - y

def w_theta1_gradient(w, b, x_values, y_values):
	m = len(x_values)
	errors_sum = 0
	for i in range(0, len(x_values)):
		errors_sum += error(w, b, x_values[i], y_values[i]) * x_values[i]
	return (1/m) * errors_sum

# print("w grad: %f" % (w_theta1_gradient(0, 0, [1, 2, 3, 4], [1, 2, 2.5, 4])))

def b_theta0_gradient(w, b, x_values, y_values):
	m = len(x_values)
	errors_sum = 0
	for i in range(0, len(x_values)):
		errors_sum += error(w, b, x_values[i], y_values[i])
	return (1/m) *errors_sum

#Gradient upate rule
def gradient_update_rule(theta1, theta0, x_values, y_values):
	delta_lr = 0.1
	theta1_grad = w_theta1_gradient(theta1, theta0, x_values, y_values)
	theta0_grad = b_theta0_gradient(theta1, theta0, x_values, y_values)
	# print("theta1_grad: %f" % theta1_grad)
	# print("theta0_grad: %f" % theta0_grad)	
	theta0 -= (delta_lr * theta0_grad)
	theta1 -= (delta_lr * theta1_grad)
	return (theta0, theta1)

def squared_error(w, b, x_values, y_values):
	res = 0
	for i in range(0, len(x_values)):
		res += ((error(w, b, x_values[i], y_values[i]))**2)
	return res

#cost function
def cost_function(w, b, x_values, y_values):
	m = len(x_values)
	sq_err_sum = squared_error(w, b, x_values, y_values)
	cost = (1/(2 * m)) * sq_err_sum
	return cost

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

def launch_train(theta1, theta0, x_values, y_values):
	max_iterations = 1000
	tolerance = 1e-7
	verbose = True

	prev_cost = cost_function(theta1, theta0, x_values, y_values)
	if verbose:
		print("Initial Cost: %s" % prev_cost)

	cost_history = [prev_cost]
	print("Iter	|   	Theta1		|   	Theta0		|		Cost	|")
	for i in range(max_iterations+1):
		curr_cost = cost_function(theta1, theta0, x_values, y_values)
		cost_history.append(curr_cost)

		if i > 0:
			theta0, theta1 = gradient_update_rule(theta1, theta0, x_values, y_values)
		print(f"{i}\t|\t{theta1}\t|\t{theta0}\t|\t{curr_cost}\t|")
		if verbose and i % 100 == 0:
			print(f"Iteration {i}: cost = {curr_cost}\ntheat0 = {theta0}\ntheta1 = {theta1}")

	print(f"Curr_cost: {curr_cost}")
	if verbose:
		print("Reached max iteration")
		print(f"Final Cost: {curr_cost}")
	return theta0, theta1, cost_history

def main():
	theta0 = 0
	theta1 = 0
	try:
		x_mileage, y_price = load_data()
		# x_values = [1, 2, 3, 4]
		# y_values = [1, 2, 2.5, 4]
		x_values = [240000.0, 139800.0, 150500.0, 185530.0, 176000.0, 114800.0, 166800.0, 89000.0, 144500.0, 84000.0, 82029.0, 63060.0, 74000.0, 97500.0, 67000.0, 76025.0, 48235.0, 93000.0, 60949.0, 65674.0, 54000.0, 68500.0, 22899.0, 61789.0]
		y_values = [3650.0, 3800.0, 4400.0, 4450.0, 5250.0, 5350.0, 5800.0, 5990.0, 5999.0, 6200.0, 6390.0, 6390.0, 6600.0, 6800.0, 6800.0, 6900.0, 6900.0, 6990.0, 7490.0, 7555.0, 7990.0, 7990.0, 7990.0, 8290.0]

		# print("mileage: ", x_mileage)
		# print("price: ", y_price)

		scaled_mileage = normalize(x_values)
		scaled_price = normalize(y_values)
		new_theta0, new_theta1, cost_history = launch_train(theta1, theta0, scaled_mileage, scaled_price)

		t0, t1 = denormalize(new_theta0, new_theta1, x_values, y_values)
		print(f"Theta1_norm\t\t|\tTheta0_norm\t\t|\tTheta1_denorm\t\t|\tTheta0_denorm\t\t|\tFinal Costd")
		print(f"{new_theta1}\t|\t{new_theta0}\t|\t{t0}\t|\t{t1}|\t{cost_history[-1]}")

		# set_thetas(t0, t1)
	except TypeError:
		print("Error: TypeError")
		return None
	except None:
		print("Error: None")
		return None
	except UnboundLocalError:
		print("Error: Incorrect assignment")
		return None

if __name__ == "__main__":
	main()