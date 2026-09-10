"""
Canonical CWE label space for the detection task.

Why this exists
---------------
The raw corpus carries 145 distinct `raw_cwe` values over ~3.6k vulnerable
functions. That space is unusable as a classification target for three reasons:

1. ~25% of it is NVD *categories* (CWE-19 "Data Processing Errors",
   CWE-254 "Security Features", ...) and CWE-1000 *pillars* (CWE-664, CWE-693).
   Those are not weaknesses, and they co-exist with their own children, so the
   label set is not mutually exclusive.
2. The long tail has 57 classes with fewer than 10 examples.
3. Sibling weaknesses that differ only in abstraction level (CWE-119 / 125 / 787,
   CWE-22 / 23 / 36) are scored as hard errors by exact match.

`map_cwe` collapses the raw value onto a canonical CWE class and returns None
for anything that should be dropped. `family_of` then groups those classes
into six coarse families used as the v4 intermediate label (detect → family
→ CWE). The two layers are independent: changing a family assignment does
not change which raw ids survive.

The corpus is ~100% C/C++, so the families are chosen around memory safety,
resource handling and C-level logic errors. Web-oriented canonical classes
are kept under INJ for mapping completeness but are expected to stay small.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Canonical classes: id -> human-readable short name.
# The short name goes into the prompt so the model picks from a described list
# rather than from a bare number.
# ---------------------------------------------------------------------------
CANONICAL: dict[str, str] = {
    # Out-of-bounds read (CWE-125) and write (CWE-787) are folded in here.
    # Telling them apart from a single function body, with no crash context and no
    # caller, is the "right family, wrong abstraction level" problem: the corpus
    # only supports ~20 independent fixes for each. Splitting this class back out
    # is the natural second stage once the coarse classifier works.
    "CWE-119": "Memory buffer error: out-of-bounds read or write, overflow",
    "CWE-416": "Use after free / double free",
    "CWE-476": "NULL pointer dereference",
    "CWE-401": "Missing release of memory or resource after effective lifetime",
    "CWE-190": "Integer overflow or wraparound",
    "CWE-369": "Divide by zero",
    "CWE-362": "Race condition / concurrent execution using shared resource",
    "CWE-400": "Uncontrolled resource consumption",
    "CWE-835": "Loop with unreachable exit condition / uncontrolled recursion",
    "CWE-704": "Incorrect type conversion or cast",
    "CWE-134": "Externally controlled format string",
    "CWE-908": "Use of uninitialized resource",
    "CWE-754": "Improper check for unusual or exceptional conditions",
    "CWE-863": "Incorrect authorization / improper access control",
    "CWE-287": "Improper authentication",
    "CWE-200": "Exposure of sensitive information to an unauthorized actor",
    "CWE-327": "Use of broken or risky cryptographic algorithm or weak randomness",
    "CWE-522": "Insufficiently protected credentials",
    "CWE-345": "Insufficient verification of data authenticity",
    "CWE-22": "Path traversal",
    "CWE-78": "OS command injection",
    "CWE-94": "Code injection",
    "CWE-89": "SQL injection",
    "CWE-79": "Cross-site scripting",
    "CWE-611": "XML external entity reference",
    "CWE-502": "Deserialization of untrusted data",
    "CWE-918": "Server-side request forgery",
    "CWE-601": "Open redirect",
    "CWE-352": "Cross-site request forgery",
    "CWE-436": "Interpretation conflict / request smuggling",
}

# ---------------------------------------------------------------------------
# Coarse families for the two-stage classifier.
# Tokens are the completion labels (MEM, AUTH, ...). Membership is by
# canonical class, not by raw NVD id.
# ---------------------------------------------------------------------------
FAMILY_ORDER: tuple[str, ...] = ("MEM", "RES", "RACE", "INJ", "AUTH", "LOGIC")

FAMILY_NAMES: dict[str, str] = {
    "MEM": "Memory safety: buffer errors, use-after-free, NULL deref, integer overflow, uninitialized",
    "RES": "Resource lifetime: leaks, unbounded consumption, infinite loop / recursion",
    "RACE": "Race condition / concurrent execution on a shared resource",
    "INJ": "Injection and untrusted control: path, command, code, format string",
    "AUTH": "Authentication, authorization, secrets, crypto, authenticity",
    "LOGIC": "Type confusion / bad cast, missing error or exceptional-condition checks",
}

# canonical CWE -> family token
CWE_TO_FAMILY: dict[str, str] = {
    "CWE-119": "MEM",
    "CWE-416": "MEM",
    "CWE-476": "MEM",
    "CWE-190": "MEM",
    "CWE-908": "MEM",
    "CWE-401": "RES",
    "CWE-400": "RES",
    "CWE-835": "RES",
    "CWE-362": "RACE",
    "CWE-22": "INJ",
    "CWE-78": "INJ",
    "CWE-94": "INJ",
    "CWE-134": "INJ",
    "CWE-89": "INJ",
    "CWE-79": "INJ",
    "CWE-611": "INJ",
    "CWE-502": "INJ",
    "CWE-918": "INJ",
    "CWE-601": "INJ",
    "CWE-352": "INJ",
    "CWE-436": "INJ",
    "CWE-863": "AUTH",
    "CWE-287": "AUTH",
    "CWE-200": "AUTH",
    "CWE-327": "AUTH",
    "CWE-522": "AUTH",
    "CWE-345": "AUTH",
    "CWE-704": "LOGIC",
    "CWE-754": "LOGIC",
    "CWE-369": "LOGIC",
}

# ---------------------------------------------------------------------------
# Raw CWE -> canonical class.
# Only entries listed here survive; everything else is dropped by map_cwe.
# ---------------------------------------------------------------------------
_CANONICAL_MEMBERS: dict[str, tuple[str, ...]] = {
    # --- memory safety ------------------------------------------------------
    "CWE-119": ("119", "120", "121", "122", "123", "124", "125", "126", "127",
                "129", "131", "193", "466", "680", "786", "787", "788", "805",
                "806", "823", "825"),
    "CWE-416": ("416", "415", "672", "590", "761", "762", "763", "1341"),
    "CWE-476": ("476", "690"),
    "CWE-908": ("908", "909", "457", "665", "824", "1187"),
    # --- resources ----------------------------------------------------------
    "CWE-401": ("401", "772", "775", "404", "459", "771", "773"),
    "CWE-400": ("400", "770", "774", "789", "406", "769", "1050"),
    "CWE-835": ("835", "834", "674", "606"),
    # --- numeric / typing ---------------------------------------------------
    "CWE-190": ("190", "191", "192", "194", "195", "196", "197"),
    "CWE-369": ("369", "1339"),
    "CWE-704": ("704", "588", "681", "843"),
    # --- concurrency --------------------------------------------------------
    "CWE-362": ("362", "364", "365", "366", "367", "421", "543", "662", "667", "820", "833", "1223"),
    # --- error handling -----------------------------------------------------
    "CWE-754": ("754", "755", "252", "253", "390", "391", "394", "617", "670", "703"),
    # --- injection ----------------------------------------------------------
    "CWE-134": ("134",),
    "CWE-22": ("22", "23", "24", "36", "37", "38", "39", "41", "59", "61", "73", "706"),
    "CWE-78": ("78", "77", "88"),
    "CWE-94": ("94", "95", "96", "98", "426", "427", "428", "434", "470", "913", "1336"),
    "CWE-89": ("89", "90", "91", "943", "564"),
    "CWE-79": ("79", "80", "81", "83", "87"),
    "CWE-611": ("611", "776", "827"),
    "CWE-502": ("502", "915"),
    "CWE-918": ("918",),
    "CWE-601": ("601",),
    "CWE-352": ("352",),
    "CWE-436": ("436", "93", "113", "115", "444"),
    # --- authn / authz / secrets -------------------------------------------
    "CWE-863": ("863", "862", "285", "266", "269", "273", "276", "279", "281",
                "282", "283", "425", "552", "566", "638", "639", "668", "732"),
    "CWE-287": ("287", "288", "290", "294", "302", "303", "304", "305", "306",
                "307", "308", "309", "384", "613", "640", "289", "291", "293"),
    "CWE-200": ("200", "203", "209", "212", "215", "226", "244", "359", "402",
                "497", "526", "532", "538", "540"),
    "CWE-327": ("327", "326", "328", "329", "330", "331", "334", "335", "336",
                "337", "338", "340", "916", "759", "760", "1240"),
    "CWE-522": ("522", "256", "257", "259", "311", "312", "313", "316", "319",
                "321", "322", "323", "798", "549"),
    "CWE-345": ("345", "346", "347", "349", "353", "354", "358", "494",
                "295", "297", "565", "924", "940", "1021"),
}

RAW_TO_CANONICAL: dict[str, str] = {}
for _canon, _members in _CANONICAL_MEMBERS.items():
    for _m in _members:
        _raw = f"CWE-{_m}"
        _prev = RAW_TO_CANONICAL.get(_raw)
        if _prev is not None and _prev != _canon:
            raise ValueError(f"{_raw} claimed by both {_prev} and {_canon}")
        RAW_TO_CANONICAL[_raw] = _canon

assert set(_CANONICAL_MEMBERS) == set(CANONICAL), (
    f"canonical member/name mismatch: {set(_CANONICAL_MEMBERS) ^ set(CANONICAL)}"
)
assert set(CWE_TO_FAMILY) == set(CANONICAL), (
    f"family coverage mismatch: {set(CWE_TO_FAMILY) ^ set(CANONICAL)}"
)
assert set(FAMILY_NAMES) == set(FAMILY_ORDER), (
    f"family name/order mismatch: {set(FAMILY_NAMES) ^ set(FAMILY_ORDER)}"
)
_unknown_fam = set(CWE_TO_FAMILY.values()) - set(FAMILY_ORDER)
if _unknown_fam:
    raise ValueError(f"unknown family tokens: {sorted(_unknown_fam)}")

FAMILY_TO_CWES: dict[str, tuple[str, ...]] = {
    fam: tuple(cid for cid, f in CWE_TO_FAMILY.items() if f == fam)
    for fam in FAMILY_ORDER
}

# ---------------------------------------------------------------------------
# Explicitly dropped: NVD categories, CWE-1000 pillars and catch-alls.
# These are not weaknesses; several are ancestors of classes we keep, so leaving
# them in would make the label set non-mutually-exclusive.
# ---------------------------------------------------------------------------
DROPPED_CATEGORIES = {
    # NVD categories
    "CWE-16", "CWE-17", "CWE-18", "CWE-19", "CWE-21", "CWE-133", "CWE-171",
    "CWE-189", "CWE-199", "CWE-254", "CWE-255", "CWE-264", "CWE-275", "CWE-310",
    "CWE-320", "CWE-355", "CWE-356", "CWE-361", "CWE-371", "CWE-388", "CWE-399",
    "CWE-417", "CWE-429", "CWE-438", "CWE-442", "CWE-465", "CWE-499", "CWE-632",
    "CWE-635", "CWE-728", "CWE-840", "CWE-845", "CWE-846", "CWE-847", "CWE-848",
    "CWE-849", "CWE-850", "CWE-851", "CWE-852", "CWE-853", "CWE-854", "CWE-855",
    "CWE-856", "CWE-857", "CWE-858", "CWE-859", "CWE-860", "CWE-861",
    # CWE-1000 pillars
    "CWE-284", "CWE-435", "CWE-664", "CWE-682", "CWE-691", "CWE-693", "CWE-697",
    "CWE-703", "CWE-707", "CWE-710",
    # catch-alls with no discriminative content
    "CWE-20", "CWE-74", "CWE-116", "CWE-138", "CWE-172", "CWE-241", "CWE-707",
    "NVD-CWE-noinfo", "NVD-CWE-Other", "CWE-Other", "CWE-noinfo",
}

# CWE-703 is both an NVD-ish abstraction and the parent of CWE-754; it is listed
# in the CWE-754 family above, so remove it from the drop set to avoid ambiguity.
DROPPED_CATEGORIES.discard("CWE-703")

_overlap = set(RAW_TO_CANONICAL) & DROPPED_CATEGORIES
if _overlap:
    raise ValueError(f"ids are both mapped and dropped: {sorted(_overlap)}")


def map_cwe(raw: str | None) -> str | None:
    """Collapse a raw CWE id onto a canonical class, or None if it must be dropped."""
    if not raw:
        return None
    raw = str(raw).strip().upper()
    if not raw.startswith("CWE-"):
        return None
    if raw in DROPPED_CATEGORIES:
        return None
    return RAW_TO_CANONICAL.get(raw)


def family_of(cwe: str | None) -> str | None:
    """Canonical CWE -> coarse family token, or None if unlabeled / unknown."""
    if not cwe:
        return None
    return CWE_TO_FAMILY.get(str(cwe).strip().upper())


def map_family(raw: str | None) -> str | None:
    """Raw NVD id -> coarse family, or None if the id is dropped/unmapped."""
    return family_of(map_cwe(raw))


def cwes_in_family(family: str, classes: list[str] | None = None) -> list[str]:
    """Canonical CWEs that belong to `family`, optionally restricted to `classes`."""
    allowed = set(classes) if classes is not None else None
    out = []
    for cid in FAMILY_TO_CWES.get(family, ()):
        if allowed is None or cid in allowed:
            out.append(cid)
    return out


def label_menu(classes: list[str] | None = None) -> str:
    """Render the allowed CWE list for the prompt."""
    ids = classes if classes is not None else list(CANONICAL)
    return "\n".join(f"- {cid}: {CANONICAL[cid]}" for cid in ids)


def family_menu(families: list[str] | None = None) -> str:
    """Render the allowed family list for the prompt."""
    ids = families if families is not None else list(FAMILY_ORDER)
    return "\n".join(f"- {fid}: {FAMILY_NAMES[fid]}" for fid in ids)


__all__ = [
    "CANONICAL",
    "RAW_TO_CANONICAL",
    "DROPPED_CATEGORIES",
    "FAMILY_ORDER",
    "FAMILY_NAMES",
    "CWE_TO_FAMILY",
    "FAMILY_TO_CWES",
    "map_cwe",
    "family_of",
    "map_family",
    "cwes_in_family",
    "label_menu",
    "family_menu",
]
