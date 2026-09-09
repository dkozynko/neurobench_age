.PHONY: assets pdf clean verify

assets:
	cd .. && uv run --frozen --extra manuscript python scripts/build_manuscript_assets.py render --evidence results/canonical/prospective --output manuscript/generated

pdf: assets
	@if command -v latexmk >/dev/null 2>&1; then \
		latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex; \
	elif command -v tectonic >/dev/null 2>&1; then \
		tectonic main.tex --keep-logs; \
	else \
		echo "A LaTeX engine is required: install latexmk or tectonic." >&2; exit 1; \
	fi

clean:
	@if command -v latexmk >/dev/null 2>&1; then latexmk -C main.tex; fi
	rm -f main.aux main.bbl main.blg main.fdb_latexmk main.fls main.log main.out main.synctex.gz main.toc main.pdf

verify: assets pdf
	@! grep -E "undefined references|Citation .* undefined|There were undefined" main.log
