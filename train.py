import csv
import predict

w_theta1 = 0
b_theta0 = 0
x_mileage = 0
y_price = 0
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

def all_errors(w, b):
	errors = []
	for i in range(0, len(x_mileage)):
		errors.append(error(w, b, x_mileage[i], y_price[i]))
	# print("errors: %s" % errors)
	return errors

all_errors(w_theta1, b_theta0)
#sum of Errors
#Gradient
#Gradient upate rule
#cost function
	#prediction
	#errors
	#squared errors
	#sum
	#cost