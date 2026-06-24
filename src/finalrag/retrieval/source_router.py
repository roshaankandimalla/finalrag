import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SourceRule:
    file_name: str
    domains: tuple[str, ...]
    patterns: tuple[str, ...]


SOURCE_RULES = (
    SourceRule(
        file_name="ril_annual_report_2025",
        domains=("finance",),
        patterns=(
            r"\breliance\b",
            r"\bril\b",
            r"\bjio\b",
            r"\bo2c\b",
            r"\bspectrum\b",
            r"\b5g\b",
            r"\bretail\b",
            r"\bvalue added\b",
            r"\bannual report\b",
        ),
    ),
    SourceRule(
        file_name="jpm_proxy_2026",
        domains=("finance",),
        patterns=(
            r"\bjpmorgan\b",
            r"\bjpm\b",
            r"\bdimon\b",
            r"\bproxy\b",
            r"\bcompensation\b",
            r"\brotce\b",
            r"\bpeer\b",
            r"\bboard\b",
            r"\brisk committee\b",
            r"\bawm\b",
        ),
    ),
    SourceRule(
        file_name="msft_2025_10k",
        domains=("finance",),
        patterns=(
            r"\bmicrosoft\b",
            r"\bmsft\b",
            r"\bazure\b",
            r"\bintelligent cloud\b",
            r"\bdeferred income tax\b",
            r"\bshares purchased\b",
            r"\b10-k\b",
            r"\bcommercial cloud\b",
        ),
    ),
    SourceRule(
        file_name="dailymed_ozempic_prescribing_label",
        domains=("medical",),
        patterns=(
            r"\bozempic\b",
            r"\bsemaglutide\b",
            r"\bdailymed\b",
            r"\bhba1c\b",
            r"\bplasma glucose\b",
            r"\bsitagliptin\b",
            r"\bliraglutide\b",
            r"\bmonotherapy\b",
            r"\bprescribing\b",
            r"\bdosage\b",
        ),
    ),
    SourceRule(
        file_name="HCAHPS-Hospital",
        domains=("medical",),
        patterns=(
            r"\bhcahps\b",
            r"\bhospital\b",
            r"\bfacility\b",
            r"\bsurvey\b",
            r"\bstar rating\b",
            r"\bpatient experience\b",
            r"\bnurse communication\b",
            r"\bdischarge\b",
            r"\bsoutheast health\b",
        ),
    ),
    SourceRule(
        file_name="united_healthcare_policy",
        domains=("medical",),
        patterns=(
            r"\bunited ?healthcare\b",
            r"\bhedis\b",
            r"\bcbp\b",
            r"\bcpt\b",
            r"\bloinc\b",
            r"\bprenatal\b",
            r"\bpostpartum\b",
            r"\bcms part d\b",
            r"\bhos\b",
            r"\bquality measure\b",
        ),
    ),
    SourceRule(
        file_name="gdpr",
        domains=("legal",),
        patterns=(
            r"\bgdpr\b",
            r"\barticle 28\b",
            r"\brecital 81\b",
            r"\bcontroller\b",
            r"\bprocessor\b",
            r"\bdata protection\b",
            r"\bsupervisory authority\b",
        ),
    ),
    SourceRule(
        file_name="RBI",
        domains=("legal",),
        patterns=(
            r"\brbi\b",
            r"\bbasel\b",
            r"\bcet1\b",
            r"\btier 1\b",
            r"\bdta\b",
            r"\bdeferred tax\b",
            r"\bminority interest\b",
            r"\bcash flow hedge\b",
            r"\bcapital adequacy\b",
        ),
    ),
    SourceRule(
        file_name="sebi_circular_2026",
        domains=("legal",),
        patterns=(
            r"\bsebi\b",
            r"\bicdr\b",
            r"\bipo\b",
            r"\bscsb\b",
            r"\brights issue\b",
            r"\bmerchant banker\b",
            r"\baudiovisual\b",
            r"\boffer document\b",
        ),
    ),
)


def _score_rule(query: str, rule: SourceRule) -> int:
    return sum(
        1
        for pattern in rule.patterns
        if re.search(pattern, query, flags=re.IGNORECASE)
    )


def route_sources(
    query: str,
    domains: list[str],
    minimum_score: int = 1,
    max_sources: int = 2,
) -> dict:
    """Pick likely source documents inside the already-routed domains.

    This is deliberately conservative: no match means no file filter, and
    ties keep up to two sources so cross-source questions still work.
    """
    allowed_domains = set(domains)
    scored = [
        {
            "file_name": rule.file_name,
            "domains": list(rule.domains),
            "score": _score_rule(query, rule),
        }
        for rule in SOURCE_RULES
        if allowed_domains.intersection(rule.domains)
    ]
    scored = [item for item in scored if item["score"] >= minimum_score]
    scored.sort(key=lambda item: (-item["score"], item["file_name"]))
    selected = scored[:max_sources]
    if not selected:
        return {
            "file_names": [],
            "confidence": "no_source_filter",
            "source_scores": [],
        }
    confidence = "high_single_source" if len(selected) == 1 else "multi_source"
    return {
        "file_names": [item["file_name"] for item in selected],
        "confidence": confidence,
        "source_scores": scored,
    }
