"""Consistency checks on paper/manuscript.json: citations vs. reference list, numbering, placeholders."""
import json
import re
from pathlib import Path

OUT = Path(__file__).resolve().parent
M = json.loads((OUT / "manuscript.json").read_text())
text = " ".join(b.get("text", "") + " " + " ".join(b.get("items", []) if b["type"] != "references" else []) + " " + b.get("caption", "")
                + " " + " ".join(" ".join(map(str, r)) for r in b.get("rows", [])) for b in M["blocks"])
refs = next(b["items"] for b in M["blocks"] if b["type"] == "references")

# reference keys: first author surname + year
keys = []
for r in refs:
    m = re.match(r"([A-ZÇĞİÖŞÜ][^,.]+)[,.].*?\((\d{4})\)", r)
    keys.append((m.group(1).strip(), m.group(2)) if m else (r[:20], "?"))
# citations in text: "Surname et al., 2020", "Surname & Surname, 2020", "Surname, 2020", "Surname (2020)", "Surname et al. (2020)"
cited = set(re.findall(r"([A-ZÇĞİÖŞÜ][\w'’\-]+)(?: et al\.| & [A-ZÇĞİÖŞÜ][\w'’\-]+)?,? \(?(\d{4})\)?", text))
problems = []
for (surname, year) in keys:
    first = surname.split()[0]
    if not any(c[0].startswith(first[:5]) and c[1] == year for c in cited):
        problems.append(f"reference never cited: {surname} ({year})")
for (s, y) in sorted(cited):
    if not any(k[0].startswith(s[:5]) and k[1] == y for k in keys) and s not in ("Section", "Table", "Figure", "Eq", "Law", "Eqs", "Fig", "In", "H1", "H2",
                                                                               "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"):
        problems.append(f"citation without reference: {s} ({y})")
# numbering
for kind in ("Table", "Figure"):
    nums = [int(n) for n in re.findall(rf"{kind} (\d+)\.", " ".join(b.get("caption", "") for b in M["blocks"]))]
    if nums != list(range(1, len(nums) + 1)):
        problems.append(f"{kind} numbering out of order: {nums}")
n_pending = text.count("[pending]")
print(f"pending placeholders in text: {n_pending}")
print("\n".join(problems) if problems else "citations and numbering consistent")
