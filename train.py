import csv
import predict

w_theta1 = 0
b_theta0 = 0
x_mileage = 0
y_price = 0
learning_rate = 0.1
dataset = []
error_list= []

with open("data.csv", 'r') as file:
	csv_reader = csv.reader(file)

	next(csv_reader)
	for line in csv_reader:
		# print(line)
		dataset.append([float(value) for value in line])

x_mileage = [row[0] for row in dataset]
y_price = [row[1] for row in dataset]

# print("X abcisse: \n %s" % x_mileage)
# print("Y abcisse: \n %s" % y_price)

#Derivative
	#I already have the derivated of dJ/d_theta1 and dJ/d_theta0 based on the formula
#Prediction
#Errors []
def error(w, b, x, y):
# error = pred[i] - y[i]
# pred = b + wx
		return predict.prediction(x, w, b) - y
print("Error: %d" % error(0, 0, 2, 4))

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
print("errors: %s" % b_theta0_gradient_errors(0, 0, [1, 2, 3, 4], [1, 2, 2.5, 4]))

def w_theta1_gradient_errors(w, b, x_values, y_values):
	errors = []
	for i in range(0, len(x_values)):
		errors.append(error(w, b, x_values[i], y_values[i]) * x_values[i])
	return errors

# print("errors sum: %f" % (sum(w_theta1_gradient_errors(0, 0, [1, 2, 3, 4], [1, 2, 2.5, 4])) / len([1, 2, 3, 4])))
#Gradient
def w_theta1_gradient(w, b, x_values, y_values):
	return (sum(w_theta1_gradient_errors(w, b, x_values, y_values)) * 2 / len(x_values))

def b_theta0_gradient(w, b, x_values, y_values):
	return (sum(b_theta0_gradient_errors(w, b, x_values, y_values)) * 2 / len(x_values))


#Gradient upate rule
def gradient_update_rule(x_values, y_values):
	global b_theta0
	global w_theta1
	gradient_update = []

	prev_b_theta0 = b_theta0
	prev_w_theta1 = w_theta1
	
	b_theta0 = b_theta0 - (learning_rate * (b_theta0_gradient(prev_w_theta1, b_theta0, x_values, y_values)))
	w_theta1 = w_theta1 - (learning_rate * (w_theta1_gradient(w_theta1, prev_b_theta0, x_values, y_values)))
	gradient_update.append(w_theta1)
	gradient_update.append(b_theta0)
	return gradient_update

# def b_theta0_gradient_update(x_values, y_values):
# 	global b_theta0
# 	b_theta0 = b_theta0 - (learning_rate * (b_theta0_gradient(w_theta1, b_theta0, x_values, y_values)))
# 	return b_theta0

print("update w: %s" % (gradient_update_rule([1, 2, 3, 4], [1, 2, 2.5, 4])))
# print("update b: %s" % (b_theta0_gradient_update([1, 2, 3, 4], [1, 2, 2.5, 4])))

#cost function
	#prediction
	#errors
	#squared errors
	#sum
	#cost