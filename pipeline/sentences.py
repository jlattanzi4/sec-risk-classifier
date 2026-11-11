"""Split a risk section into sentences, drop table debris, and normalize for matching.

The sections are plain text extracted from HTML. A newline usually ends a
paragraph, heading or list item, but about 2% of lines break in the middle of a
sentence, so lines are stitched back together before sentences are split.
Sentence boundaries are found with rules, not a model: the text is formal and
regular, the rules are fast enough for 3.6 GB, and each rule has a test.
docs/data-foundation.md records how this splitter compares with the alternatives.
"""

import hashlib
import re
import unicodedata
from typing import NamedTuple

import pandas as pd

LONG_SENTENCE_CHARS = 1500
HEADING_MAX_CHARS = 200
SENTENCE_COLUMNS = [
    "cik",
    "filing_year",
    "sent_idx",
    "para_idx",
    "text",
    "norm",
    "norm_hash",
    "n_chars",
    "is_long",
    "is_heading",
]

# Abbreviations that are always followed by more of the same sentence.
# Kept short on purpose: "SEC." and "art." end sentences all the time in a 10-K.
_NEVER_END = frozenset("mr mrs ms dr messrs prof vs v e.g i.e u.s.c c.f.r".split())
# Abbreviations that can end a sentence. A boundary after one of these counts
# only when the next word is a common sentence opener ("Beta Inc. These firms").
_MAYBE_END = frozenset("inc corp co ltd llc lp plc jr sr etc bros no nos st mt".split())
_SENTENCE_OPENERS = frozenset(
    "the we our this these that those in if as a an it its any there such for to under "
    "although while however further furthermore moreover additionally because since "
    "certain some many most each all should when where with from at on by due given "
    "neither both also accordingly consequently therefore thus unless until after "
    "before during despite like unlike similarly nevertheless nonetheless currently "
    "historically recently they he she you what who how whether one other others "
    "another even though once upon without within among between over except "
    "failure changes if".split()
)
# First words of the titles of a 10-K's items ("Item 7. Management's Discussion...").
_ITEM_TITLE_WORDS = frozenset(
    "business risk unresolved properties legal mine market selected management quantitative "
    "financial changes controls other directors executive security certain principal "
    "exhibits form reserved".split()
)
# Words a complete line never ends on; a line ending in one continues on the next.
_DANGLING = frozenset(
    "the of and or to in a an by for with as that our its their on at from is are be "
    "may will could would not any such which".split()
)

_BOUNDARY = re.compile(r"""([.!?]+["”’')\]]*)\s+(?=["“‘'(\[]*[A-Z])""")
_ENDS_SENTENCE = re.compile(r"""[.!?:;]["”’')\]]*$""")
_ENDS_LIST_ITEM = re.compile(r"[;,]\s*(?:and|or)$", re.IGNORECASE)
_BULLET = re.compile(r"^[•·●○■□▪▫◦‣⁃∙*]+\s*|^[-–—]\s+")
_BULLET_ONLY = re.compile(r"^(?:[•·●○■□▪▫◦‣⁃∙*o]|[-–—])$")
_ACRONYM = re.compile(r"^(?:[A-Za-z]\.)+[A-Za-z]$|^[A-Z]$")
_LIST_MARKER = re.compile(r"^\(?(?:\d{1,2}|[a-zA-Z]|[ivxIVX]{1,4})$")
_ITEM_REFERENCE = re.compile(r"\bItem \d{1,2}[A-C]?$", re.IGNORECASE)
_LIST_NUMBER = re.compile(r"^\(?\d{1,2}[.)]\s")
_NEXT_WORD = re.compile(r"""["“‘'(\[]*([A-Za-z]+)""")
_LAST_WORD = re.compile(r"([A-Za-z]+)$")
_DIGIT = re.compile(r"\d")
_NOT_NORM = re.compile(r"[^a-z0 ]+")
_SPACES = re.compile(r"\s+")


class Segment(NamedTuple):
    text: str
    para_idx: int
    is_heading: bool


def _paragraphs(text: str) -> list[tuple[str, bool]]:
    """Stitch lines into paragraphs: (text, is_heading) pairs.

    A bullet glyph on a line of its own marks the next line as a list item.
    A line that doesn't end a sentence is joined to the next when the next one
    plainly continues it; otherwise, if it's short, it's a heading.
    """
    lines: list[tuple[str, bool]] = []  # (text, is_list_item)
    pending_bullet = False
    for raw in text.split("\n"):
        line = _SPACES.sub(" ", raw).strip()
        if not line:
            continue
        if _BULLET_ONLY.match(line):
            pending_bullet = True
            continue
        stripped = _BULLET.sub("", line)
        is_item = pending_bullet or stripped != line
        pending_bullet = False
        if stripped:
            lines.append((stripped, is_item))

    paragraphs: list[tuple[str, bool]] = []
    i = 0
    while i < len(lines):
        current, _ = lines[i]
        while i + 1 < len(lines) and _continues(current, *lines[i + 1]):
            current = f"{current} {lines[i + 1][0]}"
            i += 1
        is_heading = (
            not lines[i][1]
            and len(current) <= HEADING_MAX_CHARS
            and not _ENDS_SENTENCE.search(current)
            and not _ENDS_LIST_ITEM.search(current)
            and not current[0].islower()
        )
        paragraphs.append((current, is_heading))
        i += 1
    return paragraphs


def _continues(current: str, following: str, following_is_item: bool) -> bool:
    """Whether `following` is the rest of a sentence that `current` left unfinished."""
    if following_is_item or _ENDS_SENTENCE.search(current) or _ENDS_LIST_ITEM.search(current):
        return False
    if following[0].islower():
        return True
    last_word = _LAST_WORD.search(current)
    if current.endswith(",") or (last_word and last_word.group(1).lower() in _DANGLING):
        return True
    # "...which was finalized in early" / "2017, went into effect": a wrapped
    # line that resumes on a number, as long as that number isn't a list marker.
    return (
        following[0].isdigit()
        and len(current) > HEADING_MAX_CHARS
        and not _LIST_NUMBER.match(following)
    )


def _is_boundary(paragraph: str, match: re.Match) -> bool:
    """Whether the punctuation at `match` ends a sentence."""
    if "!" in match.group(1) or "?" in match.group(1):
        return True
    token_start = paragraph.rfind(" ", 0, match.start()) + 1
    token = paragraph[token_start : match.start()].lstrip("\"“‘'([")
    lowered = token.lower()
    if lowered in _NEVER_END:
        return False
    if token_start == 0 and _LIST_MARKER.match(token):
        return False  # "1. Our ability to borrow": a list number, not a sentence
    if _ITEM_REFERENCE.search(paragraph, max(0, match.start() - 10), match.start()):
        # "see Item 7. Management's Discussion": a cross-reference runs into the
        # item's title. Any other word after "Item 7." starts a new sentence.
        following = _NEXT_WORD.match(paragraph, match.end())
        return not (following and following.group(1).lower() in _ITEM_TITLE_WORDS)
    if lowered in _MAYBE_END or _ACRONYM.match(token):
        following = _NEXT_WORD.match(paragraph, match.end())
        return bool(following) and following.group(1).lower() in _SENTENCE_OPENERS
    return True


def segment(text: str) -> list[Segment]:
    """Every sentence, heading and list item in a section, in order."""
    segments: list[Segment] = []
    for para_idx, (paragraph, is_heading) in enumerate(_paragraphs(text)):
        if is_heading:
            segments.append(Segment(paragraph, para_idx, True))
            continue
        start = 0
        for match in _BOUNDARY.finditer(paragraph):
            if _is_boundary(paragraph, match):
                segments.append(Segment(paragraph[start : match.end(1)], para_idx, False))
                start = match.end()
        if start < len(paragraph):
            segments.append(Segment(paragraph[start:], para_idx, False))
    return segments


def split_sentences(text: str) -> list[str]:
    return [s.text for s in segment(text)]


def is_debris(sentence: str) -> bool:
    """Table rows, page numbers and stray markers: under 15 letters, or mostly not letters."""
    letters = sum(c.isalpha() for c in sentence)
    non_space = sum(not c.isspace() for c in sentence)
    return letters < 15 or letters * 2 < non_space


def normalize(sentence: str) -> str:
    """The form sentences are compared in: NFKC, lowercase, every digit to 0, anything
    outside [a-z0-9 ] to a space, whitespace collapsed."""
    text = unicodedata.normalize("NFKC", sentence).lower()
    text = _NOT_NORM.sub(" ", _DIGIT.sub("0", text))
    return _SPACES.sub(" ", text).strip()


def norm_hash(norm: str) -> int:
    """A 64-bit key for exact matching, as a signed int so it fits a parquet int64."""
    digest = hashlib.blake2b(norm.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=True)


def filing_sentences(cik: int, filing_year: int, text: str) -> pd.DataFrame:
    """One row per kept sentence of a filing; debris is dropped before numbering."""
    return sentence_frame(cik, filing_year, [s for s in segment(text) if not is_debris(s.text)])


def sentence_frame(cik: int, filing_year: int, kept: list[Segment]) -> pd.DataFrame:
    """The sentence table for one filing, from segments that already passed the debris rule."""
    norms = [normalize(s.text) for s in kept]
    df = pd.DataFrame(
        {
            "cik": cik,
            "filing_year": filing_year,
            "sent_idx": range(len(kept)),
            "para_idx": [s.para_idx for s in kept],
            "text": [s.text for s in kept],
            "norm": norms,
            "norm_hash": [norm_hash(n) for n in norms],
            "n_chars": [len(s.text) for s in kept],
            "is_long": [len(s.text) > LONG_SENTENCE_CHARS for s in kept],
            "is_heading": [s.is_heading for s in kept],
        },
        columns=SENTENCE_COLUMNS,
    )
    return df.astype(
        {
            "cik": "int64",
            "filing_year": "int64",
            "sent_idx": "int64",
            "para_idx": "int64",
            "text": "str",
            "norm": "str",
            "norm_hash": "int64",
            "n_chars": "int64",
            "is_long": "bool",
            "is_heading": "bool",
        }
    )
