import csv
import json
import pandas as pd
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
	df['Normalized'] = (df - df.min()) / (df.max() - df.min())
	return df['Normalized'].tolist()

def denormalize(theta0, theta1, x_data, y_data):
	dernorm_t1 = theta1 * ((max(y_data) - min(y_data)) / (max(x_data) - min(x_data)))
	dernorm_t0 = theta0 * ((max(y_data) - min(y_data))) + min(y_data) - dernorm_t1 * min(x_data)
	return float(dernorm_t0), float(dernorm_t1)

denormalize(0.27991881046254985, 0.31875607394238736, [1, 2, 3, 4], [1, 2, 2.5, 4])

x_data, y_data = load_data()

def error(w, b, x, y):
		return predict.prediction(x, w, b) - y

def calculate_errors(w, b, x_values, y_values):
	errors = []
	for i in range(0, len(x_values)):
		errors.append(error(w, b, x_values[i], y_values[i]))
	return errors
# print("calc errors: %s" % (calculate_errors(0, 0, [1, 2, 3, 4], [1, 2, 2.5, 4])))
# all_errors(w_theta1, b_theta0)
def b_theta0_gradient_errors(w, b, x_values, y_values):
	errors = []
	for i in range(0, len(x_values)):
		errors.append(error(w, b, x_values[i], y_values[i]))
	return errors
# print("errors: %s" % b_theta0_gradient_errors(0, 0, [1, 2, 3, 4], [1, 2, 2.5, 4]))


def w_theta1_gradient_errors(w, b, x_values, y_values):
	errors = []
	for i in range(0, len(x_values)):
		errors.append(error(w, b, x_values[i], y_values[i]) * x_values[i])
	return errors

# print("w errors: %s" % (w_theta1_gradient_errors(0, 0, [1, 2, 3, 4], [1, 2, 2.5, 4])))
# print("w errors sum: %f" % (sum(w_theta1_gradient_errors(0, 0, [1, 2, 3, 4], [1, 2, 2.5, 4]))))
#Gradient
def w_theta1_gradient(w, b, x_values, y_values):
	return (sum(w_theta1_gradient_errors(w, b, x_values, y_values)) / len(x_values))

# print("w grad: %f" % (w_theta1_gradient(0, 0, [1, 2, 3, 4], [1, 2, 2.5, 4])))

def b_theta0_gradient(w, b, x_values, y_values):
	return (sum(b_theta0_gradient_errors(w, b, x_values, y_values)) / len(x_values))

#Gradient upate rule
def gradient_update_rule(theta1, theta0, x_values, y_values):
	delta_lr = 0.01
	theta1_grad = w_theta1_gradient(theta1, theta0, x_values, y_values)
	theta0_grad = b_theta0_gradient(theta1, theta0, x_values, y_values)
	theta0 = theta0 - (delta_lr * theta0_grad)
	theta1 = theta1 - (delta_lr * theta1_grad)
	print("b_theta0: %f" % theta0)
	print("w_theta1: %f" % theta1)	
	return (theta0, theta1)
# t1, t0 = (gradient_update_rule(0, 0, [1, 2, 3, 4], [1, 2, 2.5, 4]))
# print(f"t1: {t1} and t0: {t0}")

def squared_error(w, b, x_values, y_values):
	res = []
	for i in range(0, len(x_values)):
		res.append(pow(error(w, b, x_values[i], y_values[i]), 2))
	return res

def sum_squarred_errors(w, b, x_values, y_values):
	return sum(squared_error(w, b, x_values, y_values))

#cost function
def cost_function(w, b, x_values, y_values):
	# error = predict.prediction(x_values, w, b) - y_values
	cost = 1/(2 * len(x_values)) * sum_squarred_errors(w, b, x_values, y_values)
	return cost
# prev_cost1 = cost_function(0, 0, [1, 2, 3, 4], [1, 2, 2.5, 4])
# print("cost check: %s" % prev_cost1)

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
	iteration = 0
	max_iterations = 1000
	tolerence = 0.000001
	prev_cost = cost_function(theta1, theta0, x_values, y_values)
	print("cost: %s" % prev_cost)

	while iteration < max_iterations:
		print("Interation: ", iteration)
		theta0, theta1 = gradient_update_rule(theta1, theta0, x_values, y_values)
		curr_cost = cost_function(theta1, theta0, x_values, y_values)

		if (prev_cost - curr_cost) < tolerence:
			print("Converged at iteration %d" % iteration)
			return theta0, theta1
			# break

		prev_cost = curr_cost
		print("current cost1: %f" % prev_cost)

		iteration += 1

def main():
	theta0 = 0
	theta1 = 0
	try:
		x_values = [1, 2, 3, 4]
		y_values = [1, 2, 2.5, 4]

		x_mileage, y_price = load_data()

		# print("mileage: ", x_mileage)
		# print("price: ", y_price)

		scaled_mileage = normalize(x_mileage)
		scaled_price = normalize(y_price)
		print("scaled_mileage: %s" % scaled_mileage)
		print("scaled_price: %s" % scaled_price)

		new_theta0, new_theta1 = launch_train(theta1, theta0, scaled_mileage, scaled_price)
		print(f"new scaled theta0: {new_theta0} and new scaled theta1: {new_theta1}")

		t0, t1 = denormalize(new_theta0, new_theta1, x_mileage, y_price)
		print(f"new theta0: {t0} and new theta1: {t1}")

		set_thetas(t0, t1)
	except TypeError:
		print("Error")
		return None
	except None:
		print("Error")
		return None
	except UnboundLocalError:
		print("Error: Incorrect assignment")
		return None

if __name__ == "__main__":
	main()