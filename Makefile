SRCDIR = src
TSTDIR = test

THETA = thetas.npy

PKGS = numpy matplotlib

.SILENT:

check:
	for pkg in $(PKGS); do \
		if ! python3 -c "import $$pkg" 2>/dev/null; then \
			pip3 install $$pkg; \
		fi; \
	done

all: check
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
	git commit -m "added package check"
	git push

re: fclean all
