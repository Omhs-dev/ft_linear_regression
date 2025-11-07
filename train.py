import csv
import predict

dataset = []

with open("data.csv", 'r') as file:
	csv_reader = csv.reader(file)

	next(csv_reader)
	for line in csv_reader:
		# print(line)
		dataset.append(line)

print(dataset)

#Derivative
#Prediction
#Errors
#Gradient
#Gradient upate rule
#cost function
	#prediction
	#errors
	#squared errors
	#sum
	#cost/m