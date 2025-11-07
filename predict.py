# linear regression

def prediction(mileage, teta0, teta1):
	return teta0 + teta1 * mileage

def main():
	w = -0.05
	b = 30

	mileage = 200;

	price = prediction(mileage, b, w)

	print("predicted price: %d" % price)

if __name__ == "__main__":
	main()