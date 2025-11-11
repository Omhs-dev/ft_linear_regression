# linear regression
import json

try:
	with open("values.json", "r") as file:
		read_file = json.load(file)
		theta0 = read_file.get("Theta0")
		theta1 = read_file.get("Theta1")
		print(f"Theta0: {theta0}, Theta1: {theta1}")
except FileNotFoundError:
	print("Error: file 'value.json' not found")

def prediction(mileage, theta0, theta1):
	return theta0 + theta1 * mileage

def main():
	w = -0.05
	b = 30

	mileage = 200;

	price = prediction(mileage, b, w)

	print("predicted price: %d" % price)

if __name__ == "__main__":
	main()