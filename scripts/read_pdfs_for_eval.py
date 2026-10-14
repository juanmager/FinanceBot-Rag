"""Temporary script to extract verbatim quotes from PDFs for eval dataset."""
import sys, os
import pypdf

TARGET = sys.argv[1] if len(sys.argv) > 1 else ""
START = int(sys.argv[2]) if len(sys.argv) > 2 else 0
END = int(sys.argv[3]) if len(sys.argv) > 3 else 6

TARGET = os.path.abspath(TARGET)
reader = pypdf.PdfReader(TARGET)
print(f"Total pages: {len(reader.pages)}")
for pg_num in range(START, min(END, len(reader.pages))):
    text = reader.pages[pg_num].extract_text() or ""
    print(f"\n--- PAGE {pg_num} ---")
    print(text[:3000])
