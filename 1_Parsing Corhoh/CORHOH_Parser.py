import argparse
import csv
import os
import re
import sys
import zipfile
import xml.etree.ElementTree as ET


TEI_NS = "http://www.tei-c.org/ns/1.0"

def q(local: str) -> str:
    return f"{{{TEI_NS}}}{local}"


def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def extract_text(elem) -> str:
    """Return text content, including nested tags."""
    if elem is None:
        return ""
    return norm_space("".join(elem.itertext()))


def iter_blocks(xml_fileobj, mode: str):
    """
    Yield tuples as (doc_id, speaker, text, div_type).
    mode: answers | questions | both
    """
    # Track the current document id for nested text elements.
    doc_stack = []
    current_doc_id = None

    for event, elem in ET.iterparse(xml_fileobj, events=("start", "end")):
        if event == "start" and elem.tag == q("text") and "id" in elem.attrib:
            current_doc_id = elem.attrib["id"]
            doc_stack.append(current_doc_id)

        elif event == "end" and elem.tag == q("div"):
            div_type = elem.attrib.get("type", "").strip().lower()

            want = (
                (mode == "both" and div_type in {"answer", "question"})
                or (mode == "answers" and div_type == "answer")
                or (mode == "questions" and div_type == "question")
            )
            if want:
                speaker = extract_text(elem.find(q("speaker")))
                u_text = extract_text(elem.find(q("u")))
                yield (current_doc_id, speaker, u_text, div_type)

            # Clear processed elements to keep memory use down.
            elem.clear()

        elif event == "end" and elem.tag == q("text") and "id" in elem.attrib:
            if doc_stack:
                doc_stack.pop()
            current_doc_id = doc_stack[-1] if doc_stack else None
            elem.clear()


def open_input(path: str):
    """Return the XML file handle, directly or from a zip."""
    lower = path.lower()
    if lower.endswith(".zip"):
        zf = zipfile.ZipFile(path, "r")
        xml_names = [n for n in zf.namelist() if n.lower().endswith(".xml")]
        if not xml_names:
            raise FileNotFoundError("No .xml file found inside the zip.")
        xml_name = "CORHOH.xml" if "CORHOH.xml" in zf.namelist() else xml_names[0]
        return zf.open(xml_name, "r"), zf
    else:
        return open(path, "rb"), None


def main():
    ap = argparse.ArgumentParser(description="Parse CORHOH TEI XML and output BlockID + Text")
    ap.add_argument("--input", required=True, help="Path to CORHOH.xml or CORHOH.zip")
    ap.add_argument("--output", required=True, help="Output CSV path")
    ap.add_argument(
        "--mode",
        choices=["answers", "questions", "both"],
        default="answers",
        help="Which blocks to extract",
    )
    ap.add_argument(
        "--id_scheme",
        choices=["global", "local"],
        default="global",
        help="global = sequential A1..; local = DocID_Speaker",
    )
    ap.add_argument("--include_docid", action="store_true", help="Include DocID column")

    # Default values for running from IDLE.
    if len(sys.argv) == 1:
        default_input = r"C:??????\CORHOH.xml"
        default_output = os.path.join(os.path.dirname(default_input), "answers.csv")
        sys.argv.extend(["--input", default_input, "--output", default_output])

    args = ap.parse_args()

    in_f, zf = open_input(args.input)

    cols = ["GlobalBlockID", "Text"]
    if args.include_docid:
        cols = ["GlobalBlockID", "DocID", "Speaker", "Text"]

    # Separate counters for answer and question ids.
    a_count = 0
    q_count = 0
    b_count = 0

    with in_f:
        with open(args.output, "w", encoding="utf-8", newline="") as out:
            w = csv.writer(out)
            w.writerow(cols)

            for doc_id, speaker, text, div_type in iter_blocks(in_f, args.mode):
                if not text:
                    continue

                if args.id_scheme == "local":
                    block_id = f"{doc_id}_{speaker}" if doc_id and speaker else (speaker or doc_id or "")
                else:
                    if div_type == "answer":
                        a_count += 1
                        block_id = f"A{a_count}"
                    elif div_type == "question":
                        q_count += 1
                        block_id = f"Q{q_count}"
                    else:
                        b_count += 1
                        block_id = f"B{b_count}"

                if args.include_docid:
                    w.writerow([block_id, doc_id, speaker, text])
                else:
                    w.writerow([block_id, text])

    if zf is not None:
        zf.close()

    print("Wrote:", args.output)


if __name__ == "__main__":
    main()
