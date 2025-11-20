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
	df = pd.DataFrame({'Value': data})
	df['Normalized'] = (df["Value"] - df["Value"].min()) / (df["Value"].max() - df["Value"].min())
	return df['Normalized'].tolist()

def denormalize(theta0, theta1, x_data, y_data):
	df = pd.DataFrame({'x': x_data, 'y' : y_data})
	df['x_range'] = df['x'].max() - df["x"].min()
	df['y_range'] = df['y'].max() - df['y'].min()
	df['Theta1_denorm'] = theta1 * (df["y_range"] / df["x_range"])
	df['Theta0_denorm'] = theta0 * (df["y_range"]) + df["y"].min() - df["Theta1_denorm"] * df["x"].min()
	return (float(df["Theta0_denorm"].iloc[0]), float(df["Theta1_denorm"].iloc[0]))

# x_data, y_data = load_data()

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
	return errors_sum/m

#Gradient upate rule
def gradient_update_rule(theta1, theta0, x_values, y_values):
	delta_lr = 0.1
	theta1_grad = w_theta1_gradient(theta1, theta0, x_values, y_values)
	theta0_grad = b_theta0_gradient(theta1, theta0, x_values, y_values)
	theta0 -= (delta_lr * theta0_grad)
	theta1 -= (delta_lr * theta1_grad)
	# print("b_theta0: %f" % theta0)
	# print("w_theta1: %f" % theta1)	
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
	max_iterations = 10
	tolerance = 1e-7
	verbose = True

	prev_cost = cost_function(theta1, theta0, x_values, y_values)
	if verbose:
		print("Initial Cost: %s" % prev_cost)

	cost_history = [prev_cost]
	print("Iter	|   	Theta1		|   	Theta0		|		Cost	|")
	for i in range(max_iterations+1):
		theta0, theta1 = gradient_update_rule(theta1, theta0, x_values, y_values)
		curr_cost = cost_function(theta1, theta0, x_values, y_values)
		cost_history.append(curr_cost)

		prev_cost = curr_cost

		print(f"{i}	|	{theta1}	| {theta0}	|	{curr_cost}	|")
		# if verbose and i % 100 == 0:
			# print(f"Iteration {i}: cost = {curr_cost}\ntheat0 = {theta0}\ntheta1 = {theta1}")
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
		# print("scaled_mileage: %s" % scaled_mileage)
		# print("scaled_price: %s" % scaled_price)

		new_theta0, new_theta1, cost_history = launch_train(theta1, theta0, scaled_mileage, scaled_price)

		t0, t1 = denormalize(new_theta0, new_theta1, x_values, y_values)
		print(f"theta1_norm		|	theta0_norm		|	theta1_denorm		|	theta0_denorm		")
		print(f"{new_theta1:11.8f} 	| 	{new_theta0} 	| 	{t1}	|	{t0}")

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