#!/usr/bin/env python3
"""Package only the manuscript's explicit TeX dependencies for arXiv."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REFERENCES = re.compile(r"\\(input|include|includegraphics|bibliography)(?:\[[^\]]*\])?\{([^}]+)\}")
SUFFIX = {"input": ".tex", "include": ".tex", "includegraphics": ".pdf", "bibliography": ".bib"}


def package(repository: Path, output: Path) -> list[str]:
    repository = repository.resolve()
    manuscript = repository / "manuscript"
    payloads: dict[str, bytes] = {}

    def archive_name(path: Path) -> str:
        if path.is_relative_to(manuscript):
            return path.relative_to(manuscript).as_posix()
        if path.is_relative_to(repository / "results"):
            return "assets/" + path.relative_to(repository / "results").as_posix()
        raise ValueError(f"dependency is outside manuscript/results: {path.name}")

    def collect(path: Path) -> str:
        path = path.resolve()
        name = archive_name(path)
        if name in payloads:
            return name
        if not path.is_file():
            raise ValueError(f"missing manuscript dependency: {name}")
        if path.suffix not in {".tex", ".bib", ".bbl", ".bst", ".pdf", ".png", ".jpg", ".jpeg"}:
            raise ValueError(f"unsupported manuscript dependency: {name}")
        payloads[name] = path.read_bytes()
        if path.suffix == ".tex":
            text = re.sub(r"(?<!\\)%[^\n]*", "", payloads[name].decode())

            def replace(match: re.Match) -> str:
                command, target = match.groups()
                if "\\" in target or Path(target).is_absolute():
                    raise ValueError("only explicit relative manuscript dependencies are supported")
                dependencies = []
                for part in target.split(",") if command == "bibliography" else [target]:
                    relative = Path(part.strip())
                    if not relative.suffix:
                        relative = relative.with_suffix(SUFFIX[command])
                    dependency = collect(manuscript / relative)
                    if command == "bibliography":
                        dependency = dependency.removesuffix(".bib")
                    dependencies.append(dependency)
                return match.group(0).replace("{" + target + "}", "{" + ",".join(dependencies) + "}")

            text = REFERENCES.sub(replace, text)
            for style in re.findall(r"\\bibliographystyle\{([^}]+)\}", text):
                if (manuscript / (style + ".bst")).is_file():
                    collect(manuscript / (style + ".bst"))
            payloads[name] = text.encode()
        return name

    collect(manuscript / "main.tex")
    if (manuscript / "main.bbl").is_file():
        collect(manuscript / "main.bbl")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in sorted(payloads.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, payload)
    return sorted(payloads)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/arxiv-source.zip")
    args = parser.parse_args()
    try:
        files = package(args.repository_root, args.output)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Packaged {len(files)} manuscript dependencies: {args.output}")


if __name__ == "__main__":
    main()
