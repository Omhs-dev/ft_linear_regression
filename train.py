import csv
# import pandas as pd
import predict

b_theta0 = 0
w_theta1 = 0
x_mileage = 0
y_price = 0
dataset = []
error_list= []

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

'''
def normalize(data):
	# value = [34, 2, 54, 63]

	df = pd.DataFrame({'Value': data})

	df['Normalized'] = (df - df.min()) / (df.max() - df.min())
	print(df)

x_data, y_data = load_data()

normalize(x_data)
'''
#Derivative
	#I already have the derivated of dJ/d_theta1 and dJ/d_theta0 based on the formula
#Prediction
#Errors []
def error(w, b, x, y):
# error = pred[i] - y[i]
# pred = b + wx
		return predict.prediction(x, w, b) - y
print("Error: %d" % error(0, 0, 1, 1))

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
def gradient_update_rule(x_values, y_values):
	learning_rate = 0.1

	global b_theta0
	global w_theta1

	prev_b_theta0 = b_theta0
	prev_w_theta1 = w_theta1
	print("prev_b_theta0: %f" % prev_b_theta0)
	print("prev_w_theta1: %f" % prev_w_theta1)

	b_theta0 = b_theta0 - (learning_rate * (b_theta0_gradient(prev_w_theta1, b_theta0, x_values, y_values)))
	w_theta1 = w_theta1 - (learning_rate * (w_theta1_gradient(w_theta1, prev_b_theta0, x_values, y_values)))
	print("b_theta0: %f" % b_theta0)
	print("w_theta1: %f" % w_theta1)	
	return (b_theta0, w_theta1)

x_mileage, y_price = load_data();
new_theta0, new_theta1 = gradient_update_rule(x_mileage, y_price)
# print("b: %f" % new_theta0)
# print("w: %f" % new_theta1)
# print("xmileage: %s" % x_mileage)
# print("yprice: %s" % y_price)

def squared_error(w, b, x_values, y_values):
	res = []
	for i in range(0, len(x_values)):
		res.append(pow(error(w, b, x_values[i], y_values[i]), 2))
	return res

def sum_squarred_errors(w, b, x_values, y_values):
	return sum(squared_error(w, b, x_values, y_values))

print("errors sum: %s" % (squared_error(0, 0, [1, 2, 3, 4], [1, 2, 2.5, 4])))
print("sum errors sum: %s" % (sum_squarred_errors(0, 0, [1, 2, 3, 4], [1, 2, 2.5, 4])))

#cost function
def cost_function(w, b, x_values, y_values):
	cost = 0

	cost = 1/(2 *len(x_values)) * sum_squarred_errors(w, b, x_values, y_values)
	return cost
	#prediction
	
# 	#errors
# 	#squared errors
# 	#sum
# 	#cost

# cost_function()

print("cost: %s" % (cost_function(0, 0, [1, 2, 3, 4], [1, 2, 2.5, 4])))
