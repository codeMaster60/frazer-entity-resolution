"""Text normalization for business names and addresses.

Source 1 is clean; all noise lives in Source 2/3, so normalization only has to
move S2/S3 text toward the S1 form. Everything here is country-agnostic: the
vocabularies are unions over US / India / France patterns and are applied to
every record regardless of its country label, so the unlabelled French test
records get the same treatment as the trained-on ones.
"""
from __future__ import annotations

import re

from unidecode import unidecode

# --- legal forms, dropped from the name "core" ------------------------------
# Union of US, Indian and French forms plus common European ones. Word-order
# noise moves these to the front ("llc acme & sons marine"), so they
# are removed as a token set rather than as a trailing suffix.
LEGAL = frozenset("""
inc incorporated incorporation llc lc llp lllp lp ltd ltda limited limitee ltee
pvt pvtltd private corp corporation co company companies plc pc pa plp
sarl sarlu sas sasu sa eurl sci scic scop scm scs sca snc selarl selas eirl gie
gmbh ag bv nv srl spa oyj oy ab as aps kk pte sdn bhd opc kg ohg ug se
""".split())

# Injected boilerplate tokens seen added to noisy variants ("... LLC CENTER").
# Not removed — they can be genuine — but flagged so features can down-weight.
FILLER = frozenset("center centre service services commission".split())

# --- street types, canonicalized to one spelling ----------------------------
# US/Indian and French forms share this map; "bd"/"bld" (French boulevard) and
# "r"/"rue" collapse the same way "st"/"street" does.
STREET = {
    "street": "st", "streets": "st", "str": "st", "st": "st",
    "road": "rd", "roads": "rd", "rd": "rd",
    "avenue": "ave", "avenues": "ave", "aven": "ave", "av": "ave", "ave": "ave",
    "drive": "dr", "drives": "dr", "dr": "dr",
    "boulevard": "blvd", "boul": "blvd", "bd": "blvd", "bld": "blvd", "blvd": "blvd",
    "lane": "ln", "ln": "ln",
    "court": "ct", "ct": "ct",
    "place": "pl", "pl": "pl",
    "plaza": "plz", "plz": "plz",
    "circle": "cir", "cir": "cir",
    "highway": "hwy", "hwy": "hwy",
    "parkway": "pkwy", "pkwy": "pkwy", "pky": "pkwy",
    "terrace": "ter", "ter": "ter",
    "trail": "trl", "trl": "trl",
    "square": "sq", "sq": "sq",
    "crossing": "xing", "xing": "xing",
    "cove": "cv", "cv": "cv",
    "creek": "crk", "crk": "crk",
    "ridge": "rdg", "rdg": "rdg",
    "heights": "hts", "hts": "hts",
    "junction": "jct", "jct": "jct",
    "turnpike": "tpke", "tpke": "tpke",
    "expressway": "expy", "expy": "expy",
    "freeway": "fwy", "fwy": "fwy",
    "extension": "ext", "ext": "ext",
    # French
    "rue": "r", "r": "r",
    "route": "rte", "rte": "rte",
    "impasse": "imp", "imp": "imp",
    "allee": "all", "allees": "all", "all": "all",
    "chemin": "ch", "ch": "ch",
    "quai": "quai", "cours": "crs", "crs": "crs",
    "faubourg": "fbg", "fbg": "fbg",
    "passage": "pass", "villa": "villa", "cite": "cite",
    "residence": "res", "res": "res",
}

# --- address tokens carrying no matching signal -----------------------------
# Municipal/unit noise, the literal NULL placeholders, city-kind suffixes that
# appear on one side only ("Germantown" vs "Germantown Township"), French
# articles, and the bis/ter/quater house-number modifiers.
ADDR_STOP = frozenset("""
null nul none na nil unknown
door doors no nos num number h hno hn hse house bldg building
unit units apt apts apartment flat fl floor ste suite room rm shop office off
po pob pobox box gpo
near nr opp opposite behind beside adjacent adj above below front back
township twp city cities town village taluk tehsil tq dist district
the of and at in on
de du des la le les l d aux au sur sous
bis ter quater quinquies
plot survey sy
""".split())

# --- state / region canonicalization ---------------------------------------
# State is a high-frequency token, so it is excluded from blocking by the
# document-frequency cap anyway; canonicalizing it matters for the address
# overlap features. Indic-script state names are matched on their unidecode
# transliteration (unidecode("बिहार") == "bihara").
STATE = {
    # US full names
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar",
    "california": "ca", "colorado": "co", "connecticut": "ct", "delaware": "de",
    "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne",
    "nevada": "nv", "ohio": "oh", "oklahoma": "ok", "oregon": "or",
    "pennsylvania": "pa", "tennessee": "tn", "texas": "tx", "utah": "ut",
    "vermont": "vt", "virginia": "va", "washington": "wa", "wisconsin": "wi",
    "wyoming": "wy",
    # India, English spellings and common misspellings
    "andhra": "ap", "andhrapradesh": "ap", "arunachal": "ar", "assam": "as",
    "bihar": "br", "chhattisgarh": "cg", "chattisgarh": "cg", "goa": "ga",
    "gujarat": "gj", "haryana": "hr", "himachal": "hp", "jharkhand": "jh",
    "karnataka": "ka", "karnatak": "ka", "kerala": "kl", "madhya": "mp",
    "madhyapradesh": "mp", "maharashtra": "mh", "maharastra": "mh",
    "manipur": "mn", "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl",
    "odisha": "od", "orissa": "od", "punjab": "pb", "rajasthan": "rj",
    "sikkim": "sk", "tamilnadu": "tn", "telangana": "tg", "tripura": "tr",
    "uttarakhand": "uk", "uttaranchal": "uk", "bengal": "wb",
    "westbengal": "wb", "delhi": "dl", "puducherry": "py",
    "pondicherry": "py", "chandigarh": "ch",
    # India, Indic-script names as unidecode renders them. unidecode drops the
    # inherent vowel, so महाराष्ट्र becomes "mhaaraassttr" rather than
    # "maharashtra" — these forms are measured, not transliterated by ear.
    "bihaar": "br", "raajsthaan": "rj", "mhaaraassttr": "mh", "dillii": "dl",
    "krnaattk": "ka", "telngaanaa": "tg", "hriyaannaa": "hr", "gujraat": "gj",
    "tmilnaaddu": "tn", "tmilllnaattu": "tn", "kerl": "kl", "pnjaab": "pb",
    "oddishaa": "od", "jhaarkhndd": "jh", "asm": "as", "chttiisgddh": "cg",
    "govaa": "ga", "uttraakhndd": "uk", "bngaal": "wb", "pshcim": "wb",
    "prdesh": "", "uttr": "up", "mdhy": "mp", "aandhr": "ap",
}
# Two-letter state codes are NOT canonicalized here: several of them ("in",
# "la", "de", "or", "me", "co") are also ordinary address or French-article
# tokens, and mapping those would corrupt French addresses far more than the
# lost state token costs. States are high-frequency and excluded from blocking
# by the document-frequency cap regardless, so this only softens one feature.

# --- name junk --------------------------------------------------------------
JUNK_RE = re.compile(
    r"\(\s*id\s*:[^)]*\)"            # "(ID: 94444)"
    r"|\bd\s*/\s*b\s*/\s*a\b|\bdba\b"  # doing-business-as
    r"|\bt\s*/\s*a\b|\btrading\s+as\b"
    r"|\bm\s*/\s*s\b|\bmessrs\b"
    r"|\bwww\.|https?://"
    r"|\.com\b|\.co\.[a-z]{2}\b|\.in\b|\.net\b|\.org\b|\.fr\b|\.io\b|\.biz\b"
    r"|<\s*null\s*>",
    re.I,
)
# Honorifics only stripped in leading position, where they are added noise.
LEAD_RE = re.compile(
    r"^(?:(?:-{2,}|[@#]+|(?:mr|mrs|ms|miss|shri|sri|smt|the)\b)[\s.,]*)+", re.I
)
NONALNUM_RE = re.compile(r"[^a-z0-9]+")
WS_RE = re.compile(r"\s+")
DIGITS_RE = re.compile(r"\d")
INDIC_RE = re.compile(r"[ऀ-෿]")          # Devanagari .. Sinhala
NONASCII_RE = re.compile(r"[^\x00-\x7F]")
URLISH_RE = re.compile(r"(?:\.com|\.in\b|\.org|\.net|\.co\.|www\.|@)", re.I)
POSTAL_RE = re.compile(r"\b(\d{5,6})\b")


def _ascii(s: str) -> str:
    """unidecode only when needed — most records are already ASCII."""
    return s if s.isascii() else unidecode(s)


def norm_text(s: str) -> str:
    """Lowercase ASCII form with punctuation collapsed to single spaces."""
    s = _ascii(s).lower()
    s = JUNK_RE.sub(" ", s)
    s = NONALNUM_RE.sub(" ", s)
    return WS_RE.sub(" ", s).strip()


def name_fields(raw: str) -> tuple[str, str, int]:
    """Return ``(name_norm, name_core, flags)`` for a business name.

    ``name_norm`` keeps every token; ``name_core`` drops legal forms and
    deduplicates, so word-order and suffix noise collapse to the same string.
    ``flags`` is a bitmask: 1 = Indic script, 2 = non-ASCII, 4 = URL/handle-like.
    """
    flags = 0
    if INDIC_RE.search(raw):
        flags |= 1
    if NONASCII_RE.search(raw):
        flags |= 2
    if URLISH_RE.search(raw):
        flags |= 4

    stripped = LEAD_RE.sub("", raw.strip())
    norm = norm_text(stripped)
    toks = norm.split()
    core = sorted({t for t in toks if t not in LEGAL and len(t) > 1})
    if not core:  # name was nothing but a legal form
        core = sorted(set(toks))
    return norm, " ".join(core), flags


def addr_fields(raw: str) -> tuple[str, str, str]:
    """Return ``(addr_tok, addr_nums, postal)`` for an address.

    Addresses are reordered freely between sources, so the output is an
    order-independent token bag. ``addr_nums`` keeps the digit-bearing tokens
    (house numbers are the rarest and most discriminative address signal) and
    ``postal`` the 5/6-digit code when present (US ZIP, French CP, Indian PIN).
    """
    if not raw:
        return "", "", ""
    norm = norm_text(raw)
    postal = ""
    toks, nums = [], []
    seen = set()
    for t in norm.split():
        if DIGITS_RE.search(t):
            if not postal and len(t) in (5, 6) and t.isdigit():
                postal = t
            if t not in seen:
                nums.append(t)
        else:
            # Stopwords are tested on the raw token, before canonicalization:
            # "Indiana" -> "in" would otherwise be swallowed by the stoplist.
            if t in ADDR_STOP:
                continue
            if t in STATE:
                code = STATE[t]
                if not code:
                    continue
                t = "@" + code  # "@" can never collide with a normalized token
            else:
                t = STREET.get(t, t)
                if len(t) < 2:
                    continue
        if t not in seen:
            seen.add(t)
            toks.append(t)
    return " ".join(toks), " ".join(nums), postal
