import json
import os

def prediction(mileage, theta0, theta1):
	return theta0 + theta1 * mileage

def get_thetas():
	theta0 = 0
	theta1 = 0
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
		return theta0, theta1
	except json.JSONDecodeError:
		print("Error: failed to decode JSON file")
		return None

def get_mileage():
	try:
		mileage = input("Provide mileage: ")
		if int(mileage) < 0:
			raise Exception
		return int(mileage)
	except ValueError:
		return None

def main():
	try:
		mileage = get_mileage();
		os.system('clear')
		theta0, theta1 = get_thetas()
		price = prediction(mileage, theta0, theta1)
		print("Predicted price: %d" % price)
	except TypeError:
		print("Error: mileage value is not a valid number")
		return None
	except Exception:
		print("Please a Valid and Positive number!")
		return None

if __name__ == "__main__":
	main()
