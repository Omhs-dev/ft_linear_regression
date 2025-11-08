import csv
import predict

w_theta1 = 0
b_theta0 = 0
x_mileage = 0
y_price = 0
dataset = []

with open("data.csv", 'r') as file:
	csv_reader = csv.reader(file)

	next(csv_reader)
	for line in csv_reader:
		# print(line)
		dataset.append([float(value) for value in line])

X = [row[0] for row in dataset]
Y = [row[1] for row in dataset]

# print("X abcisse: \n %s" % X)
# print("Y abcisse: \n %s" % Y)

#Derivative
	#I already have the derivated of dJ/d_theta1 and dJ/d_theta0 based on the formula
#Prediction
#Errors []
def errors():
	print("something")
#sum of Errors
#Gradient
#Gradient upate rule
#cost function
	#prediction
	#errors
	#squared errors
	#sum
	#cost