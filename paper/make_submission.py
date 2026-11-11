"""
make_submission.py
------------------
Assembles the files for the journal submission from the generated manuscripts and paper/statements.py:

    01_Cover_letter.docx                     to the editor (not anonymized)
    02_Title_page.docx                       author details, CRediT, funding, declarations (not anonymized)
    03_Manuscript_anonymized.docx            from make_manuscript.py --anon
    04_Highlights.docx
    05_Response_to_reviewers.docx            from make_response.py --anon
    06_Declaration_of_interest.docx
    07_CRediT_author_statement.docx
    Figures/Figure_<n>.pdf                   in manuscript order
    code_and_predictions_anonymized.zip      supplementary material for the reviewers

and the public code package (ebys-routing-code.zip).  Text inside [[...]] must be completed by the authors.

    python paper/make_manuscript.py && python paper/make_manuscript.py --anon
    python paper/make_response.py --anon
    python paper/make_submission.py <output dir>
"""
import json
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

from docx import Document
from docx.enum.text import WD_COLOR_INDEX
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
sys.path.insert(0, str(OUT))
import statements as S   # noqa: E402

IDENTIFYING = ["Düzce", "Duzce", "DÜZCE", "Manisa", "Celal", "cbu.edu", "kurbankotan", "Kurban", "Kotan", "KOTAN", "Kırışoğlu", "Kirisoglu", "Serdar"]
PAPER_TOOLS = ["make_manuscript.py", "make_response.py", "make_submission.py", "statements.py", "render_docx.js", "check_manuscript.py"]


# ----------------------------------------------------------------------------- Word helpers
def new_doc(author):
    d = Document()
    st = d.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(12)
    st.element.rPr.rFonts.set("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}eastAsia", "Times New Roman")
    for s in ("Heading 1", "Heading 2", "Title"):        # drop the theme fonts and colours of the default template
        hs = d.styles[s]; hs.font.name = "Times New Roman"; hs.font.color.rgb = RGBColor(0, 0, 0)
        for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
            hs.element.rPr.rFonts.attrib.pop(qn(attr), None)
    d.core_properties.author = author; d.core_properties.last_modified_by = author
    return d


def rich(par, text, size=None):
    """[[to be completed]] -> bold + yellow highlight; **bold**; ^{superscript}."""
    for tok in re.split(r"(\[\[[^\]]+\]\]|\*\*[^*]+\*\*|\^\{[^}]+\})", text):
        if not tok:
            continue
        if tok.startswith("[["):
            r = par.add_run(tok); r.bold = True; r.font.highlight_color = WD_COLOR_INDEX.YELLOW
        elif tok.startswith("**"):
            r = par.add_run(tok[2:-2]); r.bold = True
        elif tok.startswith("^{"):
            r = par.add_run(tok[2:-1]); r.font.superscript = True
        else:
            r = par.add_run(tok)
        if size:
            r.font.size = Pt(size)
    return par


def para(d, text="", size=None, after=6):
    p = d.add_paragraph(); rich(p, text, size); p.paragraph_format.space_after = Pt(after); return p


def heading(d, text, level=1):
    h = d.add_heading(text, level=level); return h


# ----------------------------------------------------------------------------- documents
def cover_letter(path):
    d = new_doc(S.AUTHORS[0]["name"])
    for line in [S.AUTHORS[0]["name"], *S.AFFILIATIONS["a"].split(", ", 2)[:2], ", ".join(S.AFFILIATIONS["a"].split(", ")[2:]), S.EMAIL]:
        para(d, line, after=0)
    para(d, "", after=6)
    para(d, "To the Editor", after=0); para(d, "Expert Systems with Applications", after=12)
    para(d, f"**Subject:** New submission of a revised version of manuscript {S.PREVIOUS_ID}", after=12)
    para(d, "Dear Editor,")
    para(d, f"We are pleased to submit the manuscript entitled “{S.TITLE}” for consideration as a Research Paper in Expert Systems with Applications.")
    para(d, f"The manuscript is a new submission of a substantially revised version of manuscript {S.PREVIOUS_ID} (“{S.PREVIOUS_TITLE}”), which "
            "was declined after review. We thank the editor and the reviewers for their detailed comments. They led us to redesign the study rather "
            "than edit it: the experimental protocol, the set of models, the evaluation and the text were replaced. In particular:")
    for b in [
        "the evaluation is now chronological and leakage-controlled (training up to 2021, validation and test on the two halves of 2022), every "
        "choice is made on validation data, and bootstrap intervals, grouped cross-validation and McNemar tests with Holm correction are reported;",
        "Turkish text is processed natively, and machine translation, the core of the earlier pipeline, is evaluated as a controlled ablation "
        "(it lowers the accuracy of the lexical model by about four points);",
        "the baselines requested by the reviewers were added: TF-IDF with linear models, eleven sentence encoders, fine-tuned BERTurk and "
        "XLM-RoBERTa, and three large language models used zero-shot and with retrieved past documents;",
        "the routing mechanisms requested by the reviewers are implemented and evaluated: top-k suggestion, calibrated deferral to staff, a "
        "hierarchical back-off to department offices and a cost-aware cascade from a local model to a large language model;",
        "all text is de-identified before any model is trained or queried;",
        "the complete pipeline, the partitions and the prediction files of every system are released, and every number in the manuscript is "
        "generated from these files.",
    ]:
        p = d.add_paragraph(style="List Bullet"); rich(p, b); p.paragraph_format.space_after = Pt(3)
    para(d, "A point-by-point account of how each comment of the previous round was addressed is provided in the response document.")
    para(d, "We believe that the paper fits the scope of Expert Systems with Applications: it evaluates an intelligent decision-support system "
            "for a recurrent administrative decision, the routing of incoming official documents to the responsible unit, under conditions that "
            "match deployment (chronological evaluation, confidence-based deferral to staff, cost and latency of each option), and it reports "
            "which components are worth their cost.")
    para(d, "We confirm that the manuscript is original, that it has not been published and is not under consideration elsewhere, and that both "
            "authors have approved its submission. The authors declare no competing interests, and the use of generative AI in the preparation "
            "of the work is disclosed in the manuscript. "
            + f"The use of the EDMS records was permitted in writing by {S.PERMISSION['body']} ({S.PERMISSION['date']}, no. "
              f"{S.PERMISSION['number']}) for their anonymised use in academic studies."
            + " Author details are given on the separate title page only, and the code is provided to the reviewers as an anonymized "
              "supplementary archive.")
    para(d, "Yours sincerely,", after=18)
    para(d, S.AUTHORS[0]["name"], after=0)
    para(d, "Corresponding author, on behalf of both authors (" + " and ".join(a["name"] for a in S.AUTHORS) + ")", after=0)
    d.save(path)


def title_page(path):
    d = new_doc("; ".join(a["name"] for a in S.AUTHORS))
    p = d.add_paragraph(); r = p.add_run(S.TITLE); r.bold = True; r.font.size = Pt(14); p.paragraph_format.space_after = Pt(12)
    para(d, ", ".join(a["name"] + "^{" + a["aff"] + (",*" if a["corresponding"] else "") + "}" for a in S.AUTHORS), after=6)
    for k, v in S.AFFILIATIONS.items():
        para(d, f"^{{{k}}} {v}", size=11, after=2)
    corr = next(a for a in S.AUTHORS if a["corresponding"])
    para(d, f"* Corresponding author: {corr['name']}, {S.AFFILIATIONS[corr['aff']]}. E-mail: {S.EMAIL}", size=11, after=12)
    for h, t in [("CRediT authorship contribution statement", S.CREDIT), ("Declaration of competing interest", S.COMPETING),
                 ("Funding", S.FUNDING), ("Ethics statement", S.ethics(False)), ("Data and code availability", S.data(False)),
                 (S.AI_TITLE, S.AI)]:
        heading(d, h, 2); para(d, t)
    d.save(path)


def highlights(path, items):
    d = new_doc("Anonymous")
    heading(d, "Highlights", 1)
    for it in items:
        assert len(it) <= 85, it
        p = d.add_paragraph(style="List Bullet"); rich(p, it)
    d.save(path)


def declaration_of_interest(path):
    d = new_doc("Anonymous")
    heading(d, "Declaration of interests", 1)
    para(d, f"Manuscript: {S.TITLE}", size=11, after=12)
    para(d, "☒ " + S.COMPETING)
    para(d, "☐ The authors declare the following financial interests/personal relationships which may be considered as potential competing "
            "interests:")
    d.save(path)


def credit(path):
    d = new_doc("; ".join(a["name"] for a in S.AUTHORS))
    heading(d, "CRediT authorship contribution statement", 1)
    para(d, S.CREDIT)
    d.save(path)


# ----------------------------------------------------------------------------- code archives
def package_files(anon):
    files = ["README.md", "requirements.txt", "EBYS_Routing_Colab.ipynb", "build_notebook.py"]
    files += [] if anon else ["LICENSE", ".gitignore"]          # the licence names the copyright holder
    files += sorted(f"src/{p.name}" for p in (ROOT / "src").glob("*.py"))
    files += [f"paper/{n}" for n in ["extra_stats.py"] + ([] if anon else PAPER_TOOLS)]
    files += sorted(f"figures/{p.name}" for p in (ROOT / "figures").iterdir() if p.suffix in (".png", ".pdf"))
    files += sorted(f"results/{p.name}" for p in (ROOT / "results").iterdir() if p.is_file() and p.suffix in (".json", ".csv", ".xlsx"))
    for sub in ["tables", "preds", "preds_val", "val_grid", "ft_runs_v2"]:
        for r, _, fs in os.walk(ROOT / "results" / sub):
            files += sorted(str(Path(r, f).relative_to(ROOT)) for f in fs)
    bad = [f for f in files if f.endswith(".npy") or "masked" in f or "/llm/" in f or "__pycache__" in f or f.startswith(("data/", "cache/", "logs/"))]
    assert not bad, bad
    return files


def write_archive(out_zip, anon):
    key_pat = re.compile(rb"(sk-ant-[A-Za-z0-9_\-]{20,}|sk-proj-[A-Za-z0-9_\-]{20,}|sk-[A-Za-z0-9]{32,}|AIza[0-9A-Za-z_\-]{30,})")
    id_pat = re.compile("|".join(map(re.escape, IDENTIFYING)).encode())
    tmp = Path(str(out_zip) + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in package_files(anon):
            b = (ROOT / f).read_bytes()
            if anon and f == "README.md":
                b = (b.decode("utf-8").replace("(Kotan & Kırışoğlu)", "(anonymous authors, under review)")
                     .replace("extra_stats.py, make_manuscript.py + render_docx.js (manuscript generated from the result files)",
                              "extra_stats.py (statistics quoted in the paper)").encode("utf-8"))
            assert not key_pat.search(b), f
            if anon:
                assert not id_pat.search(b), (f, sorted(set(m.group().decode(errors="replace") for m in id_pat.finditer(b))))
            zi = zipfile.ZipInfo(f"edms-nlp-routing/{f}", date_time=(2026, 11, 10, 0, 0, 0)); zi.external_attr = 0o644 << 16
            z.writestr(zi, b, compress_type=zipfile.ZIP_DEFLATED)
    tmp.replace(out_zip)
    assert zipfile.ZipFile(out_zip).testzip() is None
    return len(package_files(anon))


# ----------------------------------------------------------------------------- main
def main():
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / "submission"
    dest.mkdir(parents=True, exist_ok=True)
    M = json.loads((OUT / "manuscript_anon.json").read_text(encoding="utf-8"))
    cover_letter(dest / "01_Cover_letter.docx")
    title_page(dest / "02_Title_page.docx")
    shutil.copy(OUT / "Manuscript_anonymized.docx", dest / "03_Manuscript_anonymized.docx")
    highlights(dest / "04_Highlights.docx", next(b for b in M["blocks"] if b["type"] == "highlights")["items"])
    shutil.copy(OUT / "Response_to_reviewers_anonymized.docx", dest / "05_Response_to_reviewers.docx")
    declaration_of_interest(dest / "06_Declaration_of_interest.docx")
    credit(dest / "07_CRediT_author_statement.docx")
    (dest / "Figures").mkdir(exist_ok=True)
    figs = [b["path"] for b in M["blocks"] if b["type"] == "figure"]
    for i, f in enumerate(figs, 1):
        shutil.copy(ROOT / f.replace(".png", ".pdf"), dest / "Figures" / f"Figure_{i}.pdf")
    n_anon = write_archive(dest / S.ANON_ARCHIVE, anon=True)
    n_pub = write_archive(ROOT / "ebys-routing-code.zip", anon=False)
    print(f"submission files in {dest}; figures: {len(figs)}; anonymized archive: {n_anon} files; public package: {n_pub} files")


if __name__ == "__main__":
    main()
