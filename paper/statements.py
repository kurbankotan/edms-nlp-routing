"""
statements.py
-------------
Declarations shared by the manuscript (full and anonymized), the title page and the cover letter,
so that every submission file states the same facts.  Text inside [[...]] must be completed by the
authors before submission (the renderer highlights it).
"""
REPO_URL = "https://github.com/kurbankotan/edms-nlp-routing"
ANON_ARCHIVE = "code_and_predictions_anonymized.zip"
TBC = "[[TO BE COMPLETED BY THE AUTHORS: {}]]"

TITLE = ("Confidence-aware routing of institutional documents in electronic document management systems: "
         "a leakage-controlled benchmark of lexical, embedding, fine-tuned and large language model classifiers")
PREVIOUS_ID = "ESWA-D-26-37092"
PREVIOUS_TITLE = ("Enhancement of document routing procedures in electronic document management systems through natural language "
                  "processing and machine learning algorithms")

AUTHORS = [
    {"name": "Kurban Kotan", "aff": "a", "corresponding": True},
    {"name": "Serdar Kırışoğlu", "aff": "b", "corresponding": False},
]
AFFILIATIONS = {
    "a": "Department of Artificial Intelligence and Machine Learning, Faculty of Computer and Information Sciences, Manisa Celal Bayar University, "
         "45140 Manisa, Türkiye",
    "b": "Department of Computer Engineering, Faculty of Engineering, Düzce University, 81620 Düzce, Türkiye",
}
EMAIL = "kurban.kotan@cbu.edu.tr"

CREDIT = ("**Kurban Kotan:** Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, "
          "Visualization, Writing – original draft, Writing – review & editing. **Serdar Kırışoğlu:** Supervision, Project administration, "
          "Resources, Writing – review & editing.")
COMPETING = ("The authors declare that they have no known competing financial interests or personal relationships that could have appeared "
             "to influence the work reported in this paper.")
FUNDING = "This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors."

AI_TITLE = "Declaration of generative AI and AI-assisted technologies in the manuscript preparation process"
AI = ("During the preparation of this work the authors used Claude (Anthropic) in order to help plan the experiments, write and test the "
      "analysis code, run the experiments, check the consistency of the reported numbers, verify the references, and draft and edit the "
      "manuscript text. After using this tool, the authors reviewed and edited the content as needed and take full responsibility for the "
      "content of the published article.")


PERMISSION = {"body": "the Rectorate of Düzce University", "date": "18 April 2024", "number": "E-55012428-100-422213",
              "period": "1 January 2020 to 1 January 2023"}


def ethics(anon):
    if anon:
        permission = ("The use of the records was permitted in writing by the university that holds them, for their anonymised use in academic "
                      "studies (the approving body, date and reference number are given on the title page, which is withheld from the reviewers "
                      "to preserve anonymity).")
    else:
        permission = (f"The use of the records was permitted in writing by {PERMISSION['body']} ({PERMISSION['date']}, no. {PERMISSION['number']}), "
                      "which approved the anonymised use, in academic studies, of the non-confidential documents of the Information Technologies "
                      "Department and the Health, Culture and Sports Department in the electronic document management system for the period from "
                      f"{PERMISSION['period']}.")
    return ("The study is a retrospective analysis of administrative records and involved no human participants or interventions. " + permission
            + " No separate ethics committee approval was obtained, because the study analyses existing administrative records under this "
              "written permission and involves no human participants."
            + " The records were stored and processed on the cloud services named in Section 3.4. All reported experiments use de-identified text "
              "only (Section 4.2), and only de-identified text was sent to external LLM services.")


def data(anon):
    where = (f"provided to the reviewers as an anonymized supplementary archive ({ANON_ARCHIVE}); the address of the public repository is "
             "given on the title page" if anon else f"available at {REPO_URL} under the MIT licence")
    return ("The source code of the complete pipeline (dataset construction, preprocessing, all models, evaluation, figures), the document "
            "identifiers of every partition and fold, the hyper-parameters and seeds, the prediction files of every system and the LLM prompts "
            "are " + where + ". The raw EDMS export cannot be shared because it contains personal data; requests for access to de-identified "
            "records can be addressed to the corresponding author and are subject to the permission of the data controller.")
