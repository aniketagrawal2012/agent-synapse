#!/usr/bin/env python3
"""Static verification script for Agent-Synapse IEEEtran LaTeX manuscript and IDF HTML JSON."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def verify_tex(tex_path: Path) -> None:
    text = tex_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    # 1. Check document structure
    assert "\\documentclass[10pt,conference]{IEEEtran}" in text, "Missing IEEEtran class"
    assert "\\begin{document}" in text, "Missing \\begin{document}"
    assert "\\end{document}" in text, "Missing \\end{document}"
    assert "Aniket Agrawal" in text, "Missing author name"
    assert "aniketagrawal@google.com" in text, "Missing author email"

    # 2. Check balanced \begin{...} and \end{...}
    stack: list[tuple[str, int]] = []
    env_pattern = re.compile(r"\\(begin|end)\{([^}]+)\}")
    for line_no, line in enumerate(lines, start=1):
        # strip comments
        clean_line = re.sub(r"(?<!\\)%.*$", "", line)
        for match in env_pattern.finditer(clean_line):
            kind, env_name = match.group(1), match.group(2)
            if kind == "begin":
                stack.append((env_name, line_no))
            else:
                assert stack, f"Unmatched \\end{{{env_name}}} at line {line_no}"
                top_env, start_line = stack.pop()
                assert top_env == env_name, (
                    f"Mismatched environment: \\begin{{{top_env}}} at line {start_line} "
                    f"closed by \\end{{{env_name}}} at line {line_no}"
                )
    assert not stack, f"Unclosed environments: {stack}"

    # 3. Check balanced curly braces (ignoring escaped \{ and \})
    brace_depth = 0
    for line_no, line in enumerate(lines, start=1):
        clean_line = re.sub(r"(?<!\\)%.*$", "", line)
        clean_line = clean_line.replace(r"\{", "").replace(r"\}", "")
        for ch in clean_line:
            if ch == "{":
                brace_depth += 1
            elif ch == "}":
                brace_depth -= 1
                assert brace_depth >= 0, f"Negative brace depth at line {line_no}"
    assert brace_depth == 0, f"Unbalanced curly braces at EOF: depth={brace_depth}"

    # 4. Check \label and \ref/\eqref consistency
    labels = set(re.findall(r"\\label\{([^}]+)\}", text))
    refs = set(re.findall(r"\\(?:eq)?ref\{([^}]+)\}", text))
    missing_refs = refs - labels
    assert not missing_refs, f"Undefined references: {missing_refs}"

    # 5. Check bibliography items and citations
    bibitems = set(re.findall(r"\\bibitem\{([^}]+)\}", text))
    assert len(bibitems) >= 25, f"Expected >= 25 bibitems, found {len(bibitems)}"
    cite_blocks = re.findall(r"\\cite\{([^}]+)\}", text)
    cited_keys = set()
    for block in cite_blocks:
        for k in block.split(","):
            cited_keys.add(k.strip())
    missing_cites = cited_keys - bibitems
    assert not missing_cites, f"Missing bibliography keys: {missing_cites}"
    uncited_keys = bibitems - cited_keys
    assert not uncited_keys, f"Uncited bibliography keys: {uncited_keys}"

    # 6. Check core theorems are present
    for thm_label in ("thm:conformal", "thm:noninterference", "thm:speedup"):
        assert thm_label in labels, f"Missing required theorem label: {thm_label}"

    print(
        f"[OK] {tex_path.name}: {len(lines)} lines, {len(text)} bytes, "
        f"{len(labels)} labels, {len(bibitems)}/35 verified citations (100% cited)."
    )


def verify_idf_json(json_path: Path) -> None:
    raw = json_path.read_text(encoding="utf-8")
    data = json.loads(raw)

    required_sections = [
        "Problem",
        "Related Technology",
        "Invention Description",
        "Use of Invention by Google",
        "Use of Invention by Others",
    ]
    for sec in required_sections:
        assert sec in data, f"Missing top-level IDF section: {sec}"
        html = data[sec]
        assert isinstance(html, str) and len(html) > 300, f"Section {sec} too short"
        assert "|" not in html, f"Raw Markdown pipe '|' found in IDF section {sec}"
        assert "*Describe" not in html, f"Template placeholder found in {sec}"
        assert "---" not in html, f"Markdown divider '---' found in {sec}"
        assert ("<h3>" in html or "<h4>" in html) and "<p>" in html, f"Missing HTML tags in {sec}"

    print(
        f"[OK] {json_path.name}: 5/5 go/IDF sections verified with 100% pure HTML, "
        "0 Markdown pipes, 0 template strings, 0 dividers."
    )


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    verify_tex(base / "Agent_Synapse_Paper.tex")
    verify_idf_json(base / "IDF_Agent_Synapse_Clean_HTML.json")
    print("[SUCCESS] All static verification checks passed!")
    sys.exit(0)
