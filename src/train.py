import matplotlib.pyplot as plt
import numpy as np
import sys
import csv

RED = "\033[91m"
RESET = "\033[0m"

def rmse(act, prd):
	assert len(act) == len(prd), "Length mismatch between actual and predicted values."
	return np.sqrt(np.mean((act - prd) ** 2))

def mae(act, prd):
	assert len(act) == len(prd), "Length mismatch between actual and predicted values."
	return np.mean(np.abs(act - prd))

def rsqr(act, prd):
	assert len(act) == len(prd), "Length mismatch between actual and predicted values."
	mean_act = np.mean(act)
	sse = np.sum((act - prd) ** 2)
	tss = np.sum((act - mean_act) ** 2)
	return 1 - (sse / tss) if tss != 0 else 0

def r(x, y):
	assert len(x) == len(y), "Length mismatch between x and y values."
	meanx, meany = np.mean(x), np.mean(y)
	num = np.sum((x - meanx) * (y - meany))
	den = np.sqrt(np.sum((x - meanx) ** 2) * np.sum((y - meany) ** 2))
	return num / (den) if den != 0 else 0

def normalize(data):
	return (data - np.min(data)) / (np.max(data) - np.min(data))

def plot(headers, data, th0, th1):
	try:
		kms, prices = data[:, 0], data[:, 1]
		predictions = th0 + th1 * kms

		fig, ax = plt.subplots()
		ax.scatter(kms, prices, color='red', label="Data")
		ax.plot(kms, predictions, color='blue', label="Regression Line")
		plt.text(min(kms)+0.75*(max(kms)-min(kms)), min(prices)+0.8*(max(prices)-min(prices)), 
					f"Precision:\n"
					f"- RMSE = {rmse(prices, predictions):.2f}\n"
					f"- MAE = {mae(prices, predictions):.2f}\n"
					f"- R² = {rsqr(prices, predictions):.4f}", color = 'blue')
		plt.text(min(kms)+0.2*(max(kms)-min(kms)), min(predictions)+0.3*(max(predictions)-min(predictions)), 
					f"y = ({th0:.2f}) + ({th1:.4g})x", color = 'blue')
		plt.text(min(kms)+0.1*(max(kms)-min(kms)), min(prices)+0.1*(max(prices)-min(prices)), 
					f"Correlation:\nr = {r(kms, prices):.4f}", color = 'red')
		plt.xlabel(headers[0])
		plt.ylabel(headers[1])
		plt.title("Trained Data")
		def on_key(event):
			if event.key == "escape":
				plt.close(fig)
		fig.canvas.mpl_connect("key_press_event", on_key)
		plt.show()
	except Exception as e:
		print(RED + "Error in Plotting: " + str(e) + RESET)
		sys.exit(1)

def loadData(fil):
	try:
		headers = np.genfromtxt(fil, delimiter=",", dtype=str, max_rows=1)
		return headers, np.loadtxt(fil, delimiter=",", skiprows=1)
	except Exception as e:
		print(RED + "Error: " + str(e) + RESET)
		sys.exit(1)

def epoch(ndata, act, th):
	learningRate = 0.001
	npred = th[0] + np.dot(ndata, th[1:])
	m = len(ndata)
	tmp0 = learningRate / m * np.sum(npred - act)
	tmpr = learningRate / m * np.dot((npred - act).T, ndata)
	return th - np.concatenate(([tmp0], tmpr))

def rang(data):
	return np.max(data) - np.min(data)

def trainModel(data, n):
	try:
		th = np.zeros(n)
		ndata = data.copy()
		for i in range(n):
			ndata[:, i] = normalize(data[:, i])
		ndata, act = ndata[:, :-1], ndata[:, -1]
		maxiterations = 1000000
		tolerance = 10**-8
		while True:
			prvth = th.copy()
			th = epoch(ndata, act, th)
			maxiterations -= 1
			if maxiterations % 1000 == 0 and np.all(np.abs(th - prvth) < tolerance) or maxiterations == 0:
				break

		for i in range(n-1):
			th[i+1] *= rang(data[:,n-1]) / rang(data[:,i])
		th[0] = np.min(data[:, n-1]) + th[0] * rang(data[:, n-1]) - np.sum(th[1:] * np.array([np.min(data[:, i]) for i in range(n-1)]))
		return th
	except Exception as e:
		print(RED + "Error: " + str(e) + RESET)
		sys.exit(1)

def main():
	if len(sys.argv) != 2:
		print(RED + "Pass Training Data!" + RESET)
		sys.exit(1)
	headers, data = loadData(sys.argv[1])
	n = len(headers)
	if n < 2:
		print(RED + "Needs Atleast One Dependent Variable!" + RESET)
		sys.exit(1)
	
	th = trainModel(data, n)
	print("R² =", r(data[:,-1], th[0] + np.dot(data[:,:-1], th[1:])))
	np.save("thetas.npy", th)

if __name__ == "__main__":
	main()