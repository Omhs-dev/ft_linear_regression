import csv
import pandas as pd
import predict

x_mileage = 0
y_price = 0
dataset = []

def load_data():
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

# def denormalize(data):

x_data, y_data = load_data()

def error(w, b, x, y):
		return predict.prediction(x, w, b) - y

def calculate_errors(w, b, x_values, y_values):
	errors = []
	for i in range(0, len(x_values)):
		errors.append(error(w, b, x_values[i], y_values[i]))
	return errors

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

# print("errors: %s" % (w_theta1_gradient_errors(0, 0, [1, 2, 3, 4], [1, 2, 2.5, 4])))
#Gradient
def w_theta1_gradient(w, b, x_values, y_values):
	return (sum(w_theta1_gradient_errors(w, b, x_values, y_values)) / len(x_values))

# print("errors sum: %f" % (w_theta1_gradient(0, 0, [1, 2, 3, 4], [1, 2, 2.5, 4])))

def b_theta0_gradient(w, b, x_values, y_values):
	return (sum(b_theta0_gradient_errors(w, b, x_values, y_values)) / len(x_values))

#Gradient upate rule
def gradient_update_rule(theta1, theta0, x_values, y_values):
	learning_rate = 0.01
	theta0 = theta0 - (learning_rate * (b_theta0_gradient(theta1, theta0, x_values, y_values)))
	theta1 = theta1 - (learning_rate * (w_theta1_gradient(theta1, theta0, x_values, y_values)))
	print("b_theta0: %f" % theta0)
	print("w_theta1: %f" % theta1)	
	return (theta0, theta1)

def squared_error(w, b, x_values, y_values):
	res = []
	for i in range(0, len(x_values)):
		res.append(pow(error(w, b, x_values[i], y_values[i]), 2))
	return res

def sum_squarred_errors(w, b, x_values, y_values):
	return sum(squared_error(w, b, x_values, y_values))

#cost function
def cost_function(w, b, x_values, y_values):
	cost = 1/(2 *len(x_values)) * sum_squarred_errors(w, b, x_values, y_values)
	return cost
prev_cost1 = cost_function(0, 0, [1, 2, 3, 4], [1, 2, 2.5, 4])
print("cost check: %s" % prev_cost1)

def launch_train(theta1, theta0, x_values, y_values):
	iteration = 0
	max_iterations = 1000
	tolerence = 0.000001
	prev_cost = cost_function(theta1, theta0, x_values, y_values)
	print("cost: %s" % prev_cost)

	while iteration < max_iterations:
		print("Interation: ", iteration)
		theta1, theta0 = gradient_update_rule(theta1, theta0, x_values, y_values)
		curr_cost = cost_function(theta1, theta0, x_values, y_values)

		if (prev_cost - curr_cost) < tolerence:
			print("Converged at iteration %d" % iteration)
			break

		prev_cost = curr_cost
		print("current cost1: %f" % prev_cost)

		iteration += 1


def main():
	b_theta0 = 0
	w_theta1 = 0
	try:
		print("this the main")

		x_values = [1, 2, 3, 4]
		y_values = [1, 2, 2.5, 4]

		scaled_x = normalize(x_values)
		scaled_y = normalize(y_values)
		print("scaled_x: %s" % scaled_x)
		print("scaled_y: %s" % scaled_y)

		launch_train(w_theta1, b_theta0, scaled_x, scaled_y)
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