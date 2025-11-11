# linear regression
import json

def get_thetas():
	try:
		with open("values.json", "r") as file:
			read_file = json.load(file)
			theta0 = read_file.get("Theta0")
			theta1 = read_file.get("Theta1")

			if not isinstance(theta0, (int)) and isinstance(theta1, (int)):
				return None
			if theta0 is None or theta1 is None:
				print("Error: Theta0 or Theta1 is not found in JSON file")
				return None
			return (theta0, theta1)
	except FileNotFoundError:
		print("Error: file 'value.json' not found")
		return None
	except json.JSONDecodeError:
		print("Error: failed to decode JSON file")
		return None

def prediction(mileage, theta0, theta1):
	return theta0 + theta1 * mileage

def main():
		theta0, theta1 = get_thetas()
		print(f"Theta0: {theta0}, Theta1: {theta1}")

		mileage = 360000;

		price = prediction(mileage, theta0, theta1)

		print("predicted price: %d" % price)
if __name__ == "__main__":
	main()