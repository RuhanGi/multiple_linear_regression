SRCDIR = src
TSTDIR = test

THETA = thetas.npy

.SILENT:

all:
	python3 $(TSTDIR)/gencsv.py
	python3 $(SRCDIR)/train.py $(TSTDIR)/line.csv \
	&& printf "\x1B[32m Model Trained!\x1B[0m\n" || true

e:
	python3 $(SRCDIR)/estimate.py $(TSTDIR)/line.csv

clean:
	rm -rf $(TSTDIR)/line.csv

fclean: clean
	rm -rf $(THETA)

gpush: fclean
	git add .
	git commit -m "hyperparameters tweaked"
	git push

re: fclean all
