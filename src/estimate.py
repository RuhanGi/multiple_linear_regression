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
		data = np.load("thetas.npy", allow_pickle=True).item()
		if len(data['theta']) != len(data['headers']):
			raise
		return data['theta'], data['headers']
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
	theta, headers = load()
	n = len(theta)

	prediction = theta[0]
	for i in range (n-1):
		prediction += theta[i+1] * get(headers[i])

	print(GREEN + f"Estimated {headers[n-1]}: ")
	print(CYAN + f"{prediction:0.4f}".rstrip('0').rstrip('.') + RESET)

if __name__ == "__main__":
	main()