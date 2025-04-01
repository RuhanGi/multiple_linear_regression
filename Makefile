SRCDIR = src
TSTDIR = test

THETA = thetas.npy

.SILENT:

all:
	python3 $(TSTDIR)/gencsv.py
	python3 $(SRCDIR)/train.py $(TSTDIR)/line.csv \
	&& printf "\x1B[32m Model Trained!\x1B[0m\n" || true

s:
	python3 $(SRCDIR)/train.py $(TSTDIR)/cardata.csv \
	&& printf "\x1B[32m Model Trained!\x1B[0m\n" || true

e:
	python3 $(SRCDIR)/estimate.py || true

clean:
	rm -rf $(TSTDIR)/line.csv

fclean: clean
	rm -rf $(THETA)

gpush: fclean
	git add .
	git commit -m "r modded"
	git push

re: fclean all
