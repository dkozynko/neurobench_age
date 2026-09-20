.PHONY: assets pdf clean verify arxiv

PYTHON ?= uv run --frozen --extra manuscript python

assets:
	cd .. && $(PYTHON) scripts/build_manuscript_assets.py render --evidence results/canonical/prospective --output manuscript/generated
	cd .. && $(PYTHON) scripts/build_submission_assets.py

pdf: assets
	@if command -v pdflatex >/dev/null 2>&1 && command -v bibtex >/dev/null 2>&1; then \
		pdflatex -interaction=nonstopmode -halt-on-error main.tex && \
		bibtex main && \
		pdflatex -interaction=nonstopmode -halt-on-error main.tex && \
		pdflatex -interaction=nonstopmode -halt-on-error main.tex; \
	elif command -v latexmk >/dev/null 2>&1; then \
		latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex; \
	elif command -v tectonic >/dev/null 2>&1; then \
		tectonic main.tex --keep-logs; \
	else \
		echo "Install pdfLaTeX and BibTeX, latexmk, or tectonic." >&2; exit 1; \
	fi

clean:
	@if command -v latexmk >/dev/null 2>&1; then latexmk -C main.tex; fi
	rm -f main.aux main.bbl main.blg main.fdb_latexmk main.fls main.log main.out main.synctex.gz main.toc main.pdf

verify: assets pdf
	@! grep -E "undefined references|Citation .* undefined|There were undefined|LaTeX Font Warning|Overfull" main.log

arxiv: verify
	cd .. && $(PYTHON) scripts/build_arxiv_source.py
