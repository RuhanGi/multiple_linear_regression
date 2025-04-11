SRCDIR = src
TSTDIR = test

THETA = thetas.csv

PKGS = numpy matplotlib

.SILENT:

all: check
	python3 $(SRCDIR)/train.py $(TSTDIR)/cardata.csv \
	&& printf "\x1B[32m Model Trained!\x1B[0m\n" || true

check:
	for pkg in $(PKGS); do \
		if ! python3 -c "import $$pkg" 2>/dev/null; then \
			pip3 install $$pkg; \
		fi; \
	done

s:
	python3 $(TSTDIR)/gencsv.py
	python3 $(SRCDIR)/train.py $(TSTDIR)/line.csv \
	&& printf "\x1B[32m Model Trained!\x1B[0m\n" || true

e:
	python3 $(SRCDIR)/estimate.py $(THETA)

clean:
	rm -rf $(TSTDIR)/line.csv

fclean: clean
	rm -rf $(THETA)

gpush: fclean
	git add .
	git commit -m "tweaks"
	git push

re: fclean all
