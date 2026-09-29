from pathlib import Path
from typing import Dict, List


class MarkdownExtractor:
    """Read Markdown and preserve heading-based sections."""

    def extract(self, path: str) -> Dict[str, object]:
        text = Path(path).read_text(encoding="utf-8")
        sections: List[Dict] = []
        heading = "Document"
        section_start = 0
        offset = 0
        for line in text.splitlines(keepends=True):
            if line.lstrip().startswith("#"):
                if offset > section_start:
                    sections.append({"name": heading, "start": section_start, "end": offset})
                heading = line.lstrip().lstrip("#").strip() or "Document"
                section_start = offset
            offset += len(line)
        if section_start < len(text):
            sections.append({"name": heading, "start": section_start, "end": len(text)})
        return {"text": text, "sections": sections}
