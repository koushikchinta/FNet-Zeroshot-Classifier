import kagglehub
from kagglehub import KaggleDatasetAdapter
from datasets import Dataset, concatenate_datasets
import random
import re
from embedding import _tokenizer


from utils import *


# ============================================================
# TAG DEFINITIONS
# ============================================================

tags = {
 "Administrative-Guide": "How-to manuals, user handbooks, and end-user instructional documentation. Characterized by step-by-step numbered instructions, troubleshooting tables, screenshots, and FAQ sections. Tone is helpful and explanatory; aimed at enabling a reader to perform a task. Unlike a Policy, it is informational rather than mandatory. Unlike a Procedure, the audience is broader (end users, new hires) and the scope is not tied to a specific internal business workflow.",
 "Administrative-Policy": "Formal governing documents, corporate mandates, and high-level rules of conduct. Uses authoritative, legally-leaning language such as \"shall,\" \"must,\" \"is strictly prohibited,\" and \"compliance requirements.\" Covers ethics, security, governance, data handling, and HR policy. Unlike a Procedure, this defines what is required, not how to do it. Unlike a Guide, adherence is mandatory.",
 "Administrative-Procedure": "Standard Operating Procedures (SOPs), tactical workflows, and execution checklists tied to a specific internal business process. Contains numbered steps, role assignments (\"who approves what\"), RACI tables, decision points, and flowcharts or \"Input → Process → Output\" descriptions. Unlike a Policy, it defines how to execute, not what is required. Unlike a Guide, the audience is internal/operational and the steps are a formal requirement.",
 "Communications & PR-Analyst Relations": "Materials prepared for or about industry analysts (Gartner, Forrester, IDC). Includes analyst briefing decks, RFI/RFP responses to analyst inquiries, Magic Quadrant / Wave submissions, and analyst day presentations. Focuses on \"market share,\" \"competitive positioning,\" \"technology roadmap,\" and product strategy. Dense, jargon-heavy, and oriented toward third-party validation.",
 "Communications & PR-Newsletters": "Recurring internal or external bulletin-style updates. Casual, modular, magazine-like layout with employee spotlights, \"save the date\" sections, leadership notes, and cultural announcements. Issued on a regular cadence (weekly, monthly). Unlike Marketing-Campaign Assets, the purpose is ongoing community/internal engagement, not driving a campaign conversion.",
 "Communications & PR-Press Releases": "Official media announcements distributed for journalists to pick up. Starts with a \"Dateline\" (City, State — Date), followed by a headline, body, executive quotes, and a boilerplate \"About the Company\" section at the end. Optimized for copy-paste into news coverage.",
 "Communications & PR-Public Relations": "Strategic brand management, crisis communication, and media-handling materials. Includes messaging pillars, talking points, executive Q&A prep, crisis response playbooks, and media training documents. Unlike a Press Release, this is the internal strategy/preparation, not the externally-distributed announcement. Unlike a Newsletter, the tone is strategic and confidential rather than informational.",
 "Engineering/IT-Code": "Source code for production applications. Files ending in .py, .java, .cpp, .js, .ts, .go, .cs, etc. Contains classes, methods, algorithms, business logic, and developer comments. Unlike Scripts, this is product/application logic rather than automation glue, and is typically organized into modules, packages, and tests.",
 "Engineering/IT-Configs": "Structured configuration data that drives system or application behavior. YAML, JSON, XML, TOML, INI, .env, properties files, Terraform/Helm/Kubernetes manifests. Contains port numbers, hostnames, environment variables (PROD/DEV/STAGING), feature flags, credentials references, and connection strings.",
 "Engineering/IT-Logs": "Raw runtime output from servers, applications, or infrastructure. Characterized by repeated timestamps, log levels (DEBUG/INFO/WARN/ERROR/FATAL), thread or request IDs, and error stack traces. Generally appendable, semi-structured text not intended for human authorship.","Engineering/IT-Scripts": "Automation and utility code that orchestrates tasks rather than implementing application logic. Bash, PowerShell, batch, and standalone Python/Node helper scripts for deployments, builds, cron jobs, ETL glue, sysadmin tasks, and one-off data manipulation. Unlike Code, this is workflow/automation rather than product functionality, and is typically shorter and task-focused.",
 "Engineering/IT-Technical Documentation": "Engineer- and developer-facing documentation. Includes API specifications (OpenAPI/Swagger), request/response JSON schemas, architecture diagrams (described in text), system design docs, RFCs, runbooks, README files, and product/technical requirements documents (PRDs/TRDs). Explains how the system works to a technical reader.",
 "Finance-Accounts": "Transactional, per-account financial data. Bank account statements, credit card ledgers, wire transfer confirmations, expense reimbursement logs, and account reconciliations. Each line is an individual debit or credit with a date, counterparty, and ending balance. Unlike Financial Records, the view is per-account/transactional rather than aggregated; unlike Invoices, these are settled transactions rather than payment requests.",
 "Finance-Earnings Related": "Investor-facing and externally-disclosed financial materials. Quarterly earnings releases, 10-Q and 10-K SEC filings, investor day decks, earnings call transcripts, and forward-looking guidance memos. Focuses on EBITDA, year-over-year growth, segment revenue, guidance, and dividends. Unlike Financial Records, the audience is external investors and analysts, not internal accounting/audit.",
 "Finance-Financial Records": "High-level internal accounting and audit documents. Balance sheets, profit & loss statements (P&L), general ledger summaries, trial balances, audit reports, and tax returns. Dense with currency symbols, fiscal year references, and multi-column accounting tables. Unlike Accounts, the view is aggregated/summary rather than per-transaction; unlike Earnings Related, the audience is internal/accounting/auditor, not external investors.",
 "Finance-Invoices": "Direct requests for payment, issued either from vendors to the company or to customers from the company. Contains \"Bill To\" / \"Ship To\" blocks, invoice number and date, line-item descriptions of goods or services, unit prices, tax calculations, totals, and payment terms (Net-30/Net-60). Distinguished from Accounts in that an invoice is a request for payment, not a settled transaction.",
 "HR-Employee records": "Administrative files about specific individuals throughout the employee lifecycle. Offer letters, resumes/CVs, performance evaluations, promotion history, compensation and payroll setup, training records, and disciplinary documentation. Heavy in PII — Social Security numbers, home addresses, employee IDs, dates of birth. Distinguished from Health data by focus on employment status and history rather than medical or health-status information.",
 "HR-Health data": "Sensitive medical and health-status documentation. Disability accommodation requests, FMLA / leave-of-absence files, health insurance claims, workers' compensation filings, occupational health records, and ergonomic assessments. Contains clinical jargon, physician notes, ICD codes, and benefit-plan details. Subject to HIPAA-like privacy regimes; distinguished from Employee records by focus on physical/mental health rather than employment administration.",
 "Legal & Compliance-Contracts": "Legally binding agreements between two or more parties. MSAs, SOWs, NDAs, license agreements, employment agreements, vendor contracts, and amendments. Contains clauses like \"Indemnification,\" \"Termination for Cause,\" \"Confidentiality,\" \"Governing Law,\" and \"Limitation of Liability.\" Identifiable by signature blocks, execution dates, and capitalized party definitions (\"The Company,\" \"The Provider\").","Legal & Compliance-Disputes": "Records of legal conflict, litigation, claims, and formal complaints. Cease-and-desist letters, arbitration filings, demand notices, legal briefs, discovery requests, settlement agreements, and adversarial correspondence between counsel. Contains specific case numbers, court jurisdictions, docket entries, and citations to statutes or precedent.",
 "Marketing-Advertisements": "Final paid-media creative content. Short-form ad copy, display banner creative briefs, video ad scripts, and search/social ad variants for Google Ads, LinkedIn, Meta, and programmatic platforms. Focuses on punchy headlines, pain points, and value propositions tied to a specific paid placement. Unlike Campaign Assets, this is the finished ad creative, not the surrounding campaign strategy or organic content.",
 "Marketing-Campaign Assets": "The broader creative and strategic layer of a marketing campaign. Email copy, drip-campaign sequences, social media content calendars, creative briefs, target persona definitions, and campaign messaging frameworks. Heavy on \"target personas,\" \"call to action\" (CTA), and funnel stages. Unlike Advertisements, this is the campaign concept and surrounding non-paid content, not the final paid ad creative.",
 "Marketing-Case Studies": "Customer success stories used as marketing evidence. Strictly structured as \"Challenge,\" \"Solution,\" and \"Impact / Results.\" Contains specific customer quotes, named logos, before/after metrics (e.g. \"40% faster, $X saved\"), and a deployment/timeline summary.",
 "Marketing-Event Material": "Logistics and promotional content for in-person, hybrid, or virtual events (trade shows, conferences, summits, user groups). Booth layout copy, speaker bios, session agendas, sponsorship decks, run-of-show documents, attendee communications, and post-event follow-up. Unlike Webinars, the focus is event logistics and on-site/promotional content rather than the presentation script itself.",
 "Marketing-Webinars": "Content tied to a live or recorded webinar presentation. Presenter scripts, slide-deck content, Q&A preparation, registration landing-page copy, and post-webinar follow-up emails. Unlike Event Material, the focus is the presentation content for a single virtual session, not the broader event logistics."
}


# ============================================================
# TEXT PREPROCESSING
# ============================================================
MAX_LENGTH = 600


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = str(text)

    # Normalize whitespace
    text = text.replace("\r", " ")
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = re.sub(r"\s+", " ", text)

    # Remove repeated decorative symbols
    text = re.sub(r"([^\w\s])\1{2,}", " ", text)

    # Remove standalone decorative symbol groups
    text = re.sub(
        r"(?<!\S)[^\w\s]{2,}(?!\S)",
        " ",
        text
    )

    # Remove separator lines
    text = re.sub(
        r"(?<!\w)(?:[-_=*#~]){3,}(?!\w)",
        " ",
        text
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip()

def truncate_text(text: str, max_length: int = MAX_LENGTH) -> str:
    encoding = _tokenizer.encode(text)

    if len(encoding.ids) <= max_length:
        return text

    return _tokenizer.decode(
        encoding.ids[:max_length]
    )


# ============================================================
# LOAD DATASET
# ============================================================

file_path = "new_file_classification_chunks_report.csv"

hf_dataset: Dataset = kagglehub.load_dataset(
    KaggleDatasetAdapter.HUGGING_FACE,
    "koushikchinta/tagging",
    file_path
)


# ============================================================
# FLATTEN DATASET
# ============================================================

def flatten(batch):
    new_batch = {
        SENTENCE1_COLUMN: [],
        SENTENCE2_COLUMN: [],
        SCORE: [],
    }

    for text, labels in zip(
        batch["original_chunk_text"],
        batch["classification_labels"],
    ):

        # ----------------------------------------------------
        # CLEAN TEXT BEFORE ADDING IT TO THE TRAINING DATASET
        # ----------------------------------------------------

        text = clean_text(text)
        text = truncate_text(text)

        _tags = []
        _scores = []

        if labels and labels.strip():

            labels = labels.strip()

            tag_names = labels.split(",")

            for tag_name in tag_names:

                _name, _score = tag_name.split(":", 1)

                _name = _name.strip()

                _tags.append(_name)
                _scores.append(float(_score))

            # Add negative/random examples until there are 5
            num_random = max(0, 5 - len(_tags))

            random_tags = random.choices(
                list(tags.keys()),
                k=num_random,
            )

            _tags.extend(random_tags)
            _scores.extend([0.0] * num_random)

        else:

            _tags = random.choices(
                list(tags.keys()),
                k=5,
            )

            _scores = [0.0] * 5

        # ----------------------------------------------------
        # CREATE TEXT/TAG PAIRS
        # ----------------------------------------------------

        for _tag, _score in zip(_tags, _scores):

            new_batch[SENTENCE1_COLUMN].append(text)

            new_batch[SENTENCE2_COLUMN].append(
                tags[_tag]
            )

            new_batch[SCORE].append(_score)

    return new_batch


# ============================================================
# APPLY PREPROCESSING + FLATTEN
# ============================================================

hf_dataset = (
    hf_dataset
    .select_columns([
        "original_chunk_text",
        "classification_labels",
    ])
    .map(
        flatten,
        batched=True,
        remove_columns=[
            "original_chunk_text",
            "classification_labels",
        ],
    )
)


# ============================================================
# CREATE TAG-TO-DESCRIPTION PAIRS
# ============================================================

tag_pairs = {
    SENTENCE1_COLUMN: [],
    SENTENCE2_COLUMN: [],
    SCORE: [],
}

tag_names = list(tags.keys())


for tag_name, description in tags.items():

    # Positive:
    # tag name -> its own description
    tag_pairs[SENTENCE1_COLUMN].append(tag_name)

    tag_pairs[SENTENCE2_COLUMN].append(description)

    tag_pairs[SCORE].append(1.0)

    # Negative:
    # tag name -> another tag's description
    random_tags = random.choices(
        [t for t in tag_names if t != tag_name],
        k=3,
    )

    for random_tag in random_tags:

        tag_pairs[SENTENCE1_COLUMN].append(tag_name)

        tag_pairs[SENTENCE2_COLUMN].append(
            tags[random_tag]
        )

        tag_pairs[SCORE].append(0.0)


df2 = Dataset.from_dict(tag_pairs)


# ============================================================
# COMBINE DATASETS
# ============================================================

kaggle_set = concatenate_datasets([
    df2,
    hf_dataset,
]).train_test_split(
    test_size=0.15
)


kaggle_train_set = kaggle_set["train"]


kaggle_sub_set = kaggle_set["test"].train_test_split(
    test_size=0.5
)


kaggle_test_set = kaggle_sub_set["train"]

kaggle_val_set = kaggle_sub_set["test"]


# ============================================================
# DATASET INFORMATION
# ============================================================

if __name__ == "__main__":

    print("\n===== DATASET =====")

    print("Train:")
    print(kaggle_train_set)

    print("\nTest:")
    print(kaggle_test_set)

    print("\nValidation:")
    print(kaggle_val_set)

    # --------------------------------------------------------
    # Show a few examples before/after cleaning
    # --------------------------------------------------------

    print("\n===== SAMPLE CLEANED TEXT =====")

    for i in range(min(5, len(kaggle_train_set))):

        print("\n--- Example", i, "---")

        print(
            kaggle_train_set[i][SENTENCE1_COLUMN]
        )

        print(
            "TAG:",
            kaggle_train_set[i][SENTENCE2_COLUMN]
        )

        print(
            "SCORE:",
            kaggle_train_set[i][SCORE]
        )



    lengths = [
        len(_tokenizer.encode(text).ids)
        for text in hf_dataset[SENTENCE1_COLUMN]
    ]

    import numpy as np

    print("Token statistics")
    print("---------------")
    print("Mean :", np.mean(lengths))
    print("P50  :", np.percentile(lengths, 50))
    print("P90  :", np.percentile(lengths, 90))
    print("P95  :", np.percentile(lengths, 95))
    print("P99  :", np.percentile(lengths, 99))
    print("Max  :", max(lengths))