import logging
from typing import Dict, List, Optional

import nltk
from nltk.tokenize import sent_tokenize


logger = logging.getLogger(__name__)


# Download sentence tokenizer data if necessary.
try:
    nltk.data.find(
        "tokenizers/punkt_tab"
    )
except LookupError:
    nltk.download(
        "punkt_tab",
        quiet=True,
    )


class TextChunker:
    """Intelligent text chunking for academic papers."""

    def __init__(
        self,
        target_chunk_size: int = 768,
        min_chunk_size: int = 256,
        max_chunk_size: int = 1024,
        overlap_size: int = 128,
    ):
        self.target_chunk_size = target_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size

    def chunk_paper(
        self,
        text: str,
        sections: List[Dict],
        preserve_sections: bool = True,
    ) -> List[Dict]:
        """
        Chunk paper text intelligently.

        Returns a list of chunks with metadata.
        """

        if not text.strip():
            return []

        if not preserve_sections or not sections:
            return self._chunk_text(text)

        chunks = []

        for section in sections:

            start = section["start"]
            end = section["end"]

            section_text = text[
                start:end
            ].strip()

            if not section_text:
                continue

            section_name = section.get(
                "name",
                "Unknown",
            )

            section_chunks = self._chunk_text(
                section_text,
                section_name=section_name,
            )

            chunks.extend(
                section_chunks
            )

        # Add global chunk indexes.
        for index, chunk in enumerate(
            chunks
        ):
            chunk["chunk_index"] = index

        return chunks

    def _chunk_text(
        self,
        text: str,
        section_name: Optional[str] = None,
    ) -> List[Dict]:
        """Chunk text using sentence boundaries."""

        sentences = sent_tokenize(
            text
        )

        chunks = []

        current_sentences = []
        current_tokens = 0

        for sentence in sentences:

            sentence_tokens = self._count_tokens(
                sentence
            )

            # Extremely long individual sentence.
            if sentence_tokens > self.max_chunk_size:

                if current_sentences:
                    chunks.append(
                        self._make_chunk(
                            current_sentences,
                            section_name,
                        )
                    )

                    current_sentences = []
                    current_tokens = 0

                chunks.append(
                    {
                        "text": sentence.strip(),
                        "token_count": sentence_tokens,
                        "section_name": section_name,
                    }
                )

                continue

            would_exceed = (
                current_tokens
                + sentence_tokens
                > self.max_chunk_size
            )

            if (
                would_exceed
                and current_sentences
            ):

                chunks.append(
                    self._make_chunk(
                        current_sentences,
                        section_name,
                    )
                )

                overlap = self._get_overlap_sentences(
                    current_sentences,
                    self.overlap_size,
                )

                current_sentences = overlap

                current_tokens = sum(
                    self._count_tokens(sentence)
                    for sentence in current_sentences
                )

            current_sentences.append(
                sentence
            )

            current_tokens += sentence_tokens

            # Once we're around the target,
            # allow the next sentence to decide
            # whether the chunk should close.
            if (
                current_tokens
                >= self.target_chunk_size
            ):
                continue

        if current_sentences:
            chunks.append(
                self._make_chunk(
                    current_sentences,
                    section_name,
                )
            )

        # Merge tiny trailing chunks.
        chunks = self._merge_small_chunks(
            chunks
        )

        return chunks

    def _get_overlap_sentences(
        self,
        sentences: List[str],
        target_tokens: int,
    ) -> List[str]:
        """Get sentences from the end for overlap."""

        overlap = []
        token_count = 0

        for sentence in reversed(
            sentences
        ):
            sentence_tokens = self._count_tokens(
                sentence
            )

            if (
                token_count
                + sentence_tokens
                > target_tokens
            ):
                break

            overlap.insert(
                0,
                sentence,
            )

            token_count += sentence_tokens

        return overlap

    def _make_chunk(
        self,
        sentences: List[str],
        section_name: Optional[str],
    ) -> Dict:
        """Create a chunk dictionary."""

        text = " ".join(
            sentence.strip()
            for sentence in sentences
        )

        return {
            "text": text,
            "token_count": self._count_tokens(
                text
            ),
            "section_name": section_name,
        }

    def _count_tokens(
        self,
        text: str,
    ) -> int:
        """
        Estimate token count.

        This is intentionally simple for now.
        The actual embedding tokenizer may use
        a somewhat different token count.
        """

        if not text.strip():
            return 0

        return len(
            text.split()
        )

    def _merge_small_chunks(
        self,
        chunks: List[Dict],
    ) -> List[Dict]:
        """Merge chunks that are too small."""

        if not chunks:
            return []

        result = []

        for chunk in chunks:

            if (
                result
                and chunk["token_count"]
                < self.min_chunk_size
                and result[-1]["token_count"]
                + chunk["token_count"]
                <= self.max_chunk_size
            ):
                previous = result[-1]

                previous["text"] = (
                    previous["text"]
                    + " "
                    + chunk["text"]
                )

                previous["token_count"] = (
                    self._count_tokens(
                        previous["text"]
                    )
                )

            else:
                result.append(chunk)

        return result