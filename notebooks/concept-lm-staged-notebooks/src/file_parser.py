"""
concept_lm/file_parser.py

Multi-format file ingestion for the concept LM chatbot.
Parses PDF, Python, Markdown, Word docs, and Jupyter notebooks into
structured text segments suitable for concept-level processing.
"""

import os
import re
import ast
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class TextSegment:
    text: str
    source: str          # filename
    segment_type: str    # 'paragraph', 'code_block', 'cell', 'heading', etc.
    position: int        # index within the document


class FileParser:
    """
    Unified file parser that routes to format-specific extractors and
    returns a list of TextSegment objects for downstream encoding.
    """
    SUPPORTED = {'.pdf', '.py', '.md', '.txt', '.docx', '.ipynb', '.rst'}

    def parse(self, filepath: str) -> List[TextSegment]:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"{filepath} not found")
        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED:
            raise ValueError(f"Unsupported file type: {suffix}. Supported: {self.SUPPORTED}")

        dispatch = {
            '.pdf': self._parse_pdf,
            '.py': self._parse_python,
            '.md': self._parse_markdown,
            '.txt': self._parse_plaintext,
            '.docx': self._parse_docx,
            '.ipynb': self._parse_notebook,
            '.rst': self._parse_plaintext,
        }
        return dispatch[suffix](filepath)

    def _parse_pdf(self, filepath: str) -> List[TextSegment]:
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("pip install pymupdf")

        doc = fitz.open(filepath)
        segments = []
        for page_num, page in enumerate(doc):
            blocks = page.get_text("blocks")
            for i, block in enumerate(blocks):
                text = block[4].strip()
                if len(text) < 10:
                    continue
                seg_type = 'heading' if (block[3] - block[1]) < 20 and len(text) < 100 else 'paragraph'
                segments.append(TextSegment(
                    text=text,
                    source=filepath,
                    segment_type=seg_type,
                    position=page_num * 1000 + i
                ))
        return segments

    def _parse_python(self, filepath: str) -> List[TextSegment]:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()

        segments = []
        try:
            tree = ast.parse(source)
        except SyntaxError:
            # Fall back to raw text blocks
            return self._split_by_blank_lines(source, filepath, 'code_block')

        lines = source.split('\n')
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = node.lineno - 1
                end = node.end_lineno
                block_text = '\n'.join(lines[start:end])
                seg_type = 'class_def' if isinstance(node, ast.ClassDef) else 'function_def'
                segments.append(TextSegment(
                    text=block_text,
                    source=filepath,
                    segment_type=seg_type,
                    position=start
                ))

        if not segments:
            return self._split_by_blank_lines(source, filepath, 'code_block')

        segments.sort(key=lambda s: s.position)
        return segments

    def _parse_markdown(self, filepath: str) -> List[TextSegment]:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        segments = []
        # Split on markdown headings
        parts = re.split(r'(#{1,6}\s+.+)', content)
        position = 0
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if re.match(r'#{1,6}\s+', part):
                seg_type = 'heading'
            elif part.startswith('```'):
                seg_type = 'code_block'
            else:
                seg_type = 'paragraph'
            segments.append(TextSegment(
                text=part, source=filepath, segment_type=seg_type, position=position
            ))
            position += 1
        return segments

    def _parse_plaintext(self, filepath: str) -> List[TextSegment]:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return self._split_by_blank_lines(content, filepath, 'paragraph')

    def _parse_docx(self, filepath: str) -> List[TextSegment]:
        try:
            from docx import Document
        except ImportError:
            raise ImportError("pip install python-docx")

        doc = Document(filepath)
        segments = []
        for i, para in enumerate(doc.paragraphs):
            text = para.text.strip()
            if not text:
                continue
            seg_type = 'heading' if para.style.name.startswith('Heading') else 'paragraph'
            segments.append(TextSegment(text=text, source=filepath, segment_type=seg_type, position=i))
        return segments

    def _parse_notebook(self, filepath: str) -> List[TextSegment]:
        try:
            import nbformat
        except ImportError:
            raise ImportError("pip install nbformat")

        with open(filepath, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)

        segments = []
        for i, cell in enumerate(nb.cells):
            source = cell['source'].strip()
            if not source:
                continue
            seg_type = 'markdown_cell' if cell['cell_type'] == 'markdown' else 'code_cell'
            segments.append(TextSegment(text=source, source=filepath, segment_type=seg_type, position=i))
        return segments

    def _split_by_blank_lines(self, text: str, source: str, seg_type: str) -> List[TextSegment]:
        blocks = re.split(r'\n\s*\n', text)
        segments = []
        for i, block in enumerate(blocks):
            block = block.strip()
            if len(block) > 5:
                segments.append(TextSegment(text=block, source=source, segment_type=seg_type, position=i))
        return segments


def segments_to_text(segments: List[TextSegment], max_chars: int = 100000) -> str:
    """Concatenate segments with double newlines up to max_chars."""
    parts = []
    total = 0
    for seg in segments:
        if total + len(seg.text) > max_chars:
            break
        parts.append(seg.text)
        total += len(seg.text)
    return '\n\n'.join(parts)
