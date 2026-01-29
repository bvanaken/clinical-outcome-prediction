import argparse
from typing import Dict, List, Tuple, Optional

import pandas as pd
import os
import re
import csv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mimic_dir', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--admission_only', default=False)
    parser.add_argument('--seed', default=123, type=int)

    return parser.parse_args()


def filter_notes(notes_df: pd.DataFrame, admissions_df: pd.DataFrame, admission_text_only=False) -> pd.DataFrame:
    """
    Keep only Discharge Summaries and filter out Newborn admissions. Replace duplicates and join reports with
    their addendums. If admission_text_only is True, filter all sections that are not known at admission time.
    """
    # filter out newborns
    adm_grownups = admissions_df[admissions_df.ADMISSION_TYPE != "NEWBORN"]
    notes_df = notes_df[notes_df.HADM_ID.isin(adm_grownups.HADM_ID)]

    # remove notes with no TEXT or HADM_ID
    notes_df = notes_df.dropna(subset=["TEXT", "HADM_ID"])

    # filter discharge summaries
    notes_df = notes_df[notes_df.CATEGORY == "Discharge summary"]

    # remove duplicates and keep the later ones
    notes_df = notes_df.sort_values(by=["CHARTDATE"])
    notes_df = notes_df.drop_duplicates(subset=["TEXT"], keep="last")

    # combine text of same admissions (those are usually addendums)
    combined_adm_texts = notes_df.groupby('HADM_ID')['TEXT'].apply(lambda x: '\n\n'.join(x)).reset_index()
    notes_df = notes_df[notes_df.DESCRIPTION == "Report"]
    notes_df = notes_df[["HADM_ID", "ROW_ID", "SUBJECT_ID", "CHARTDATE"]]
    notes_df = notes_df.drop_duplicates(subset=["HADM_ID"], keep="last")
    notes_df = pd.merge(combined_adm_texts, notes_df, on="HADM_ID", how="inner")

    # strip texts from leading and trailing and white spaces
    notes_df["TEXT"] = notes_df["TEXT"].str.strip()

    # remove entries without admission id, subject id or text
    notes_df = notes_df.dropna(subset=["HADM_ID", "SUBJECT_ID", "TEXT"])

    if admission_text_only:
        # reduce text to admission-only text
        notes_df = filter_admission_text(notes_df)

    return notes_df

def _norm_header(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s[:-1] if s.endswith(":") else s
    return s

def build_header_map(admission_sections: Dict[str, str],
                     extra_variants: Optional[Dict[str, List[str]]] = None) -> Dict[str, str]:
    """
    Returns map: normalized_header_text -> canonical_key
    """
    header_map = {}
    for canon_key, header in admission_sections.items():
        header_map[_norm_header(header)] = canon_key

    if extra_variants:
        for canon_key, variants in extra_variants.items():
            for v in variants:
                header_map[_norm_header(v)] = canon_key

    return header_map


def extract_sections_with_rules(note_text: str,
                                header_map: Dict[str, str],
                                require_blankline_for_unknown: bool = True) -> Dict[str, str]:
    """
    Strategy:
    - Find candidate header lines (line-anchored).
    - Accept a candidate as a boundary if:
        A) it maps to a target header (in header_map)  -> ALWAYS accept
        B) else accept only if it looks like a real boundary (e.g. preceded by blank line / 2 newlines)
           (configurable)
    - Slice sections from accepted boundary headers to next accepted boundary header.
    - If the same canonical header appears multiple times, concatenate.
    """

    # Candidate header = a line that is "Header:" or "Header" with nothing else on the line
    # (we do NOT match inline like "Chief Complaint: nausea" here; we’ll handle inline targets separately)
    header_line_re = re.compile(
        r"(?m)^(?P<h>[A-Za-z][A-Za-z0-9 /&\-\(\)\[\]]{0,60}?):\s*$"
    )

    candidates: List[Tuple[str, int, int]] = []  # (raw_header, line_start, line_end)
    for m in header_line_re.finditer(note_text):
        candidates.append((m.group("h"), m.start(), m.end()))

    # Helper: check if there are 2 newlines (i.e., blank line) right before the header line
    # We interpret "two newlines" as: somewhere immediately before line_start we have "\n\n"
    # ignoring spaces/tabs on the blank line.
    def preceded_by_blankline(line_start: int) -> bool:
        prefix = note_text[:line_start]
        # remove trailing spaces/tabs
        prefix = re.sub(r"[ \t]+$", "", prefix)
        return prefix.endswith("\n\n")

    # Accepted boundaries: (canon_key, header_line_start, header_line_end)
    boundaries: List[Tuple[str, int, int]] = []

    for raw, ls, le in candidates:
        canon = header_map.get(_norm_header(raw))

        if canon is not None:
            # target header: accept always (this saves inline-ish formatting like Attending\nChief Complaint)
            boundaries.append((canon, ls, le))
        else:
            # unknown header: accept only if structure strongly suggests a section boundary
            if not require_blankline_for_unknown:
                continue
            if preceded_by_blankline(ls):
                # We accept unknown boundaries only if they have a blank line before them
                # (prevents "Body ... \nFakeHeader:\nBody continued" from splitting)
                boundaries.append((_norm_header(raw), ls, le))  # keep normalized name if you want to store it
            else:
                # treat as body text, do nothing
                pass

    # ALSO handle inline target headers: "Chief Complaint: nausea, vomiting"
    # This catches cases where the header and its value are on the same line.
    inline_target_re = re.compile(r"(?m)^(?P<h>[^:\n]{2,60}):\s*(?P<v>.+)$")
    inline_hits: List[Tuple[str, int, int, str]] = []  # (canon, start, end, value)
    for m in inline_target_re.finditer(note_text):
        canon = header_map.get(_norm_header(m.group("h")))
        if canon:
            inline_hits.append((canon, m.start(), m.end(), m.group("v").strip()))

    # If we found inline hits, we want them to behave like boundaries too,
    # but we must be careful not to double-count if the same line was already a boundary.
    boundary_starts = {b[1] for b in boundaries}
    for canon, s, e, _ in inline_hits:
        if s not in boundary_starts:
            boundaries.append((canon, s, e))

    # Sort boundaries by occurrence
    boundaries.sort(key=lambda t: t[1])

    if not boundaries:
        return {}

    # Slice content between boundaries
    out: Dict[str, str] = {}
    for i, (canon, h_start, h_end) in enumerate(boundaries):
        content_start = h_end
        content_end = boundaries[i + 1][1] if i + 1 < len(boundaries) else len(note_text)
        content = note_text[content_start:content_end].strip()

        # If this boundary came from an inline target ("Header: value"),
        # the "content" might include that same value already; better to use parsed inline value if available.
        # We'll overwrite the first line if it matches.
        # (Simple approach: if inline matched at same start, use its value as prefix.)
        for c2, s2, e2, val in inline_hits:
            if c2 == canon and s2 == h_start:
                # Replace content with the remainder after that line, prefixed by inline value
                remainder = note_text[e2:content_end].strip()
                content = (val + ("\n" + remainder if remainder else "")).strip()
                break

        if canon in out and content:
            out[canon] = (out[canon].rstrip() + "\n\n" + content)
        elif content:
            out[canon] = content
        else:
            out.setdefault(canon, "")

    return out


def filter_admission_text(notes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract only the sections of the discharge summary that are known at admission time.
    """
    admission_sections = {
        "CHIEF_COMPLAINT": "Chief Complaint:",
        "PRESENT_ILLNESS": "History of Present Illness:",
        "MEDICAL_HISTORY": "Past Medical History:",
        "MEDICATION_ADM": "Medications on Admission:",
        "ALLERGIES": "Allergies:",
        "PHYSICAL_EXAM": "Physical Exam:",
        "FAMILY_HISTORY": "Family History:",
        "SOCIAL_HISTORY": "Social History:"
    }

    extra_variants = {
        "PRESENT_ILLNESS": ["Present Illness", "HPI", "History Present Illness"],
        "MEDICAL_HISTORY": ["PMH", "Medical History"],
        "MEDICATION_ADM": ["Home Medications", "Medications"],
    }

    header_map = build_header_map(admission_sections, extra_variants)

    for key in admission_sections.keys():
        notes_df[key] = ""

    for i, x in enumerate(notes_df["TEXT"]):
        sec = extract_sections_with_rules(x, header_map, require_blankline_for_unknown=True)
        for k, v in sec.items():
            if k in admission_sections.keys():
                notes_df.at[i, k] = v

    # filter notes with missing main information
    notes_df = notes_df[(notes_df.CHIEF_COMPLAINT != "") | (notes_df.PRESENT_ILLNESS != "") |
                        (notes_df.MEDICAL_HISTORY != "")]

    # add section headers and combine into TEXT_ADMISSION
    notes_df = notes_df.assign(TEXT="CHIEF COMPLAINT: " + notes_df.CHIEF_COMPLAINT.astype(str)
                                    + '\n\n' +
                                    "PRESENT ILLNESS: " + notes_df.PRESENT_ILLNESS.astype(str)
                                    + '\n\n' +
                                    "MEDICAL HISTORY: " + notes_df.MEDICAL_HISTORY.astype(str)
                                    + '\n\n' +
                                    "MEDICATION ON ADMISSION: " + notes_df.MEDICATION_ADM.astype(str)
                                    + '\n\n' +
                                    "ALLERGIES: " + notes_df.ALLERGIES.astype(str)
                                    + '\n\n' +
                                    "PHYSICAL EXAM: " + notes_df.PHYSICAL_EXAM.astype(str)
                                    + '\n\n' +
                                    "FAMILY HISTORY: " + notes_df.FAMILY_HISTORY.astype(str)
                                    + '\n\n' +
                                    "SOCIAL HISTORY: " + notes_df.SOCIAL_HISTORY.astype(str))

    return notes_df


def save_mimic_split_patient_wise(df, label_column, save_dir, task_name, seed, column_list=None):
    """
    Splits a MIMIC dataframe into 70/10/20 train, val, test with no patient occuring in more than one set.
    Uses ROW_ID as ID column and save to save_path.
    """
    if column_list is None:
        column_list = ["ID", "TEXT", label_column]

    # Load prebuilt MIMIC patient splits
    data_split = {"train": pd.read_csv("tasks/mimic_train.csv"),
                  "val": pd.read_csv("tasks/mimic_val.csv"),
                  "test": pd.read_csv("tasks/mimic_test.csv")}

    # Use row id as general id and cast to int
    df = df.rename(columns={'HADM_ID': 'ID'})
    df.ID = df.ID.astype(int)

    # Create path to task data
    os.makedirs(save_dir, exist_ok=True)

    # Save splits to data folder
    for split_name in ["train", "val", "test"]:
        split_set = df[df.SUBJECT_ID.isin(data_split[split_name].SUBJECT_ID)].sample(frac=1,
                                                                                     random_state=seed)[column_list]

        # lower case column names
        split_set.columns = map(str.lower, split_set.columns)

        split_set.to_csv(os.path.join(save_dir, "{}_{}.csv".format(task_name, split_name)),
                         index=False,
                         quoting=csv.QUOTE_ALL)
