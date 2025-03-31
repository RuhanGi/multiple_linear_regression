import csv
import random

def generate_data(n, filename):
	with open(filename, mode='w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(['Age (years)', 'Taste Rating', 'Availability (units)', 'Price ($)'])
		r = 0.05
		for _ in range(n):
			age = random.uniform(0, 100)
			taste = random.uniform(0, 10)
			units = random.uniform(1, 200000)
			y = (50 + 5 * age + 10 * taste - 1/2000 * units) * random.uniform(1-r, 1+r)
			writer.writerow([age, taste, units, y])

n = 100
filename = "test/line.csv"

generate_data(n, filename)
