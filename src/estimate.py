import csv
import sys

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
PURPLE = "\033[95m"
CYAN = "\033[96m"
GRAY = "\033[97m"
BLACK = "\033[98m"
RESET = "\033[0m"

def load(fil):
	try:
		with open(fil, mode="r") as file:
			reader = csv.reader(file)
			headers = next(reader)
			thetas = next(reader)
			thetas = [float(theta) for theta in thetas]
		if len(thetas) != len(headers):
			raise
		return thetas, headers
	except:
		print(RED + "No properly trained file found!" + RESET)
		return [0,0], ["Independent Variable", "Dependent Variable"]

def get(name):
	try:
		return float(input(BLUE + name + ": " + YELLOW))
	except:
		print(RED + "Input a Number!" + RESET)
		return get(name)

def main():
	if len(sys.argv) != 2:
		print(RED + "Pass Weights Data!" + RESET)
		sys.exit(1)
	theta, headers = load(sys.argv[1])
	n = len(theta)

	prediction = theta[n-1]
	for i in range (n-1):
		prediction += theta[i] * get(headers[i])

	print(GREEN + f"Estimated {headers[n-1]}: ")
	print(CYAN + f"{prediction:.2f}" + RESET)

if __name__ == "__main__":
	main()