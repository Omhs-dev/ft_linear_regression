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
	try:
		mileage = float(sys.argv[1])
	except ValueError:
		return None

	return mileage

def prediction(mileage, theta0, theta1):
	return theta0 + theta1 * mileage

def main():
		try:
			if len(sys.argv) <= 2:
				mileage = get_mileage();
				theta0, theta1 = get_thetas()
				price = prediction(mileage, theta0, theta1)

				print("predicted price: %d" % price)
		except IndexError:
			print("No Argument: mileage has not been provided!")
		except TypeError:
			print("Error: mileage value is not a valid number")
			return None
if __name__ == "__main__":
	main()