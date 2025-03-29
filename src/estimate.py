import numpy as np
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

def loadHeaders(fil):
	try:
		return np.genfromtxt(fil, delimiter=",", dtype=str, max_rows=1)
	except Exception as e:
		print(RED + "Error: " + str(e) + RESET)
		sys.exit(1)

def load():
	try:
		return np.load("thetas.npy")
	except:
		print(RED + "Improper trained file!" + RESET)
		return 0, 0

def get(name):
	try:
		return float(input(BLUE + name + ": " + YELLOW))
	except:
		print(RED + "Input a Number!" + RESET)
		sys.exit(1)

def main():
	if len(sys.argv) > 2:
		print(RED + "Pass Only the Training File!" + RESET)
		sys.exit(1)

	headers = loadHeaders(sys.argv[1]) if len(sys.argv) == 2 else None
	theta = load()
	n = len(theta)

	if (headers is not None and n != len(headers)) or (headers is None and len(sys.argv) > 1):
		print(RED + "Header and Trained File Don't Match!" + RESET)
		sys.exit(1)

	prediction = theta[0]
	for i in range (n-1):
		prediction += theta[i+1] * get(headers[i] if headers is not None else f"Dependent Variable {i+1}")

	print(GREEN + f"Estimated {headers[n-1] if headers is not None else 'Independent Variable'}: ")
	print(CYAN + f"{prediction:0.4f}".rstrip('0').rstrip('.') + RESET)

if __name__ == "__main__":
	main()