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

def arsqr(ind, prd, n, p):
	assert len(ind) == len(prd), "Length mismatch between actual and predicted values."
	num = (1 - rsqr(ind, prd)) * (n - 1)
	den = (n - p - 1)
	return 1 - num/den if den != 0 else 0

def r(x, y):
	assert len(x) == len(y), "Length mismatch between x and y values."
	meanx, meany = np.mean(x), np.mean(y)
	num = np.sum((x - meanx) * (y - meany))
	den = np.sqrt(np.sum((x - meanx) ** 2) * np.sum((y - meany) ** 2))
	return num / den if den != 0 else 0

def rpartial(xy, xx, yy):
	return -xy / np.sqrt(xx * yy) if xx * yy != 0 else 0

def loadData(fil):
	try:
		headers = np.genfromtxt(fil, delimiter=",", dtype=str, max_rows=1)
		data = np.loadtxt(fil, delimiter=",", skiprows=1)
		data = data[~(np.isnan(data) | np.isinf(data)).any(axis=1)]
		return headers, data
	except Exception as e:
		print(RED + "Error: " + str(e) + RESET)
		sys.exit(1)

def epoch(ndata, act, th):
	learningRate = 0.05
	m = len(ndata)
	diff = (th[0] + np.dot(ndata, th[1:])) - act
	th[0] -= learningRate / m * np.sum(diff)
	th[1:] -= learningRate / m * (diff @ ndata)
	return th

def trainModel(data, n):
	try:
		th = np.zeros(n)
		mins = np.min(data, axis=0)
		maxs = np.max(data, axis=0)
		ranges = maxs - mins
		ranges[ranges == 0] = 1
		ndata = (data - mins) / ranges
		ndata, act = ndata[:, :-1], ndata[:, -1]
		maxiterations = 100000
		tolerance = 10**-7
		
		for i in range(maxiterations):
			prvth = th.copy()
			th = epoch(ndata, act, th)
			maxDiff = np.max(np.abs(th - prvth))
			print(f"\rEpoch [{i}/{maxiterations}]: {maxDiff:.6f}",end="")
			if maxDiff < tolerance:
				print("\r" + " " * 30 + "\r",end="")
				break

		for i in range(n-1):
			th[i+1] *= ranges[n-1] / ranges[i]
		th[0] = mins[-1] + th[0] * ranges[-1] - np.sum(th[1:] * mins[:-1])
		return th
	except Exception as e:
		print(RED + "Error: " + str(e) + RESET)
		sys.exit(1)

def plot(headers, full, n, th):
	try:
		data, ind = full[:,:-1], full[:,-1]
		prd = th[0] + np.dot(data, th[1:])
		diff = ind - prd

		fig, ax = plt.subplots(2, n-1, figsize=(9, 6))
		fig.subplots_adjust(hspace=0.3, wspace=0.5)
		equation = f"{headers[n-1]} = {th[0]:.4f}" + "".join([f" + {th[i+1]:.4f} * {headers[i]}" for i in range(n-1)])
		metrics = (
			f"\n\nEvaluation Metrics\n"
			f"{equation}"
			f"\nRMSE = {rmse(ind, prd):.2f}, "
			f"MAE = {mae(ind, prd):.2f}, "
			f"R² = {rsqr(ind, prd):.4f}"
		)
		if n == 2:
			ax = np.reshape(ax, (2, 1))
		else:
			metrics += f", Adjusted R² = {arsqr(ind, prd, len(ind), n-1):.4f}"

		fig.suptitle(metrics, fontsize=12, ha='center', va='center', y=0.98)
		means = np.mean(data, axis=0)
		Rmat = np.array([[r(full[:, i], full[:, j]) for j in range(n)] for i in range(n)])
		Rinv = np.linalg.inv(Rmat)
		for i in range(n-1):
			line = th[0] + th[i+1] * data[:, i] + sum(th[j+1] * means[j] if j != i else 0 for j in range(n-1))
			ax[0, i].scatter(data[:, i], ind, color='red', label="Data")
			ax[0, i].plot(data[:, i], line, color='blue', label="Average Line")
			ax[0, i].set_xlabel(headers[i])
			ax[0, i].set_ylabel(headers[n-1])
			xlim, ylim = ax[0, i].get_xlim(), ax[0, i].get_ylim()
			ax[0, i].text(xlim[1] - 0.45 * (xlim[1]-xlim[0]), ylim[0] + 0.01 * (ylim[1]-ylim[0]),
					f"r={Rmat[i, -1]:.4f}\n"
					+ (f"r*={rpartial(Rinv[i,-1], Rinv[i,i], Rinv[-1,-1]):.4f}" if n > 2 else ""), color = 'navy')
		for i in range(n-1):
			ax[1, i].scatter(data[:, i], diff, color='gray', label="Residuals")
			ax[1, i].axhline(y=0, color='black', label="Axis")
			ax[1, i].set_xlabel(headers[i])
			ax[1, i].set_ylabel('Residual')

		def on_key(event):
			if event.key == "escape":
				plt.close(fig)
		fig.canvas.mpl_connect("key_press_event", on_key)
		plt.show()

	except Exception as e:
		print(RED + "Error in Plotting: " + str(e) + RESET)
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
	with open("thetas.csv", mode="w", newline="") as file:
		writer = csv.writer(file)
		writer.writerow(headers)
		writer.writerow(list(th[1:]) + [th[0]])
	plot(headers, data, n, th)

if __name__ == "__main__":
	main()