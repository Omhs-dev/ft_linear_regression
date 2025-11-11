# linear regression
import json
import sys

# print("Total arguments:", len(sys.argv))
# print("Script name:", sys.argv[0])
# print("Arguments:", sys.argv[1:])
# print("argv length: %d" % len(sys.argv))

def get_thetas():
	try:
		with open("values.json", "r") as file:
			read_file = json.load(file)
			theta0 = read_file.get("Theta0")
			theta1 = read_file.get("Theta1")

			if theta0 is None or theta1 is None:
				print("Error: Theta0 or Theta1 is not found in JSON file")
				return None
			try:
				theta0 = float(theta0)
				theta1 = float(theta1)
			except ValueError:
				print("Error: ThetaO ro Theta 1 is not a valide number")
				return None

			return (theta0, theta1)
	except FileNotFoundError:
		print("Error: file 'value.json' not found")
		return None
	except json.JSONDecodeError:
		print("Error: failed to decode JSON file")
		return None

def get_mileage():
	mileage = float(sys.argv[1])
	return mileage

print("mileage: %f" % get_mileage())
print(isinstance(get_mileage(), float))
def prediction(mileage, theta0, theta1):
	return theta0 + theta1 * mileage

def main():
		if len(sys.argv) <= 2:
			mileage = get_mileage();
		theta0, theta1 = get_thetas()
		print(f"Theta0: {theta0}, Theta1: {theta1}")

		price = prediction(mileage, theta0, theta1)

		print("predicted price: %d" % price)
if __name__ == "__main__":
	main()