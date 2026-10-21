"""Prompt taxonomy for glioma sub-region segmentation (WT / TC / ET).

Families (pre-registered in docs/연구방향_최종.md):
  short    : atomic noun phrases (SAM3-native style; MedSAM3 finding: <=3 words work best)
  radiol   : radiological-finding descriptions (imaging-appearance informed)
  hier     : hierarchy-explicit descriptions encoding the set relation ET ⊂ TC ⊂ WT
  ensemble : eval-time probability averaging over the 3 trainable families
"""

REGIONS = ["WT", "TC", "ET"]

PROMPT_FAMILIES: dict[str, dict[str, str]] = {
    "short": {
        "WT": "glioma",
        "TC": "tumor core",
        "ET": "enhancing tumor",
    },
    "radiol": {
        "WT": "whole glioma including peritumoral edema, hyperintense on FLAIR",
        "TC": "glioma core including necrotic and enhancing regions, excluding edema",
        "ET": "contrast-enhancing tumor, bright on post-contrast T1",
    },
    "hier": {
        "WT": "entire glioma region containing the tumor core and surrounding edema",
        "TC": "tumor core inside the whole glioma, excluding the surrounding edema",
        "ET": "enhancing tumor inside the tumor core, excluding necrosis and edema",
    },
}

# KiTS23 mirror (nested TUM ⊂ MASS ⊂ KID, mapped onto the WT/TC/ET slots)
PROMPT_FAMILIES_KITS: dict[str, dict[str, str]] = {
    "short": {
        "WT": "kidney",
        "TC": "kidney mass",
        "ET": "kidney tumor",
    },
    "radiol": {
        "WT": "entire kidney including any tumors and cysts, on contrast-enhanced CT",
        "TC": "kidney masses including tumor and cysts, distinct from normal parenchyma",
        "ET": "solid renal tumor, heterogeneously enhancing on CT",
    },
    "hier": {
        "WT": "entire kidney region containing all masses and normal parenchyma",
        "TC": "masses inside the kidney, excluding normal kidney parenchyma",
        "ET": "solid tumor inside the kidney masses, excluding cysts",
    },
}

# Paraphrase variants (defense batch): 4 per family, same register as the parent family.
PARAPHRASE_FAMILIES: dict[str, dict[str, str]] = {
    # --- short register: atomic noun phrases
    "short_p1": {"WT": "brain tumor", "TC": "core of the tumor", "ET": "enhancing lesion"},
    "short_p2": {"WT": "tumor", "TC": "central tumor core", "ET": "contrast-enhancing region"},
    "short_p3": {"WT": "glial tumor", "TC": "inner tumor core", "ET": "enhancing tumor tissue"},
    "short_p4": {"WT": "cerebral neoplasm", "TC": "solid tumor core", "ET": "avidly enhancing tumor"},
    # --- radiol register: imaging-appearance descriptions
    "radiol_p1": {
        "WT": "complete tumor extent with surrounding edema, appearing bright on FLAIR imaging",
        "TC": "central tumor including necrosis and enhancement, edema excluded",
        "ET": "tumor tissue showing avid enhancement on post-gadolinium T1-weighted images"},
    "radiol_p2": {
        "WT": "full lesion including vasogenic edema, FLAIR-hyperintense region",
        "TC": "necrotic and enhancing tumor portions without the edematous rim",
        "ET": "brightly enhancing tumor on contrast-enhanced T1"},
    "radiol_p3": {
        "WT": "entire abnormal signal region including peritumoral edema on FLAIR",
        "TC": "tumor core comprising necrotic center and enhancing rim, not edema",
        "ET": "region of contrast uptake on T1 post-contrast"},
    "radiol_p4": {
        "WT": "whole tumor with edema, hyperintense on T2-FLAIR sequences",
        "TC": "core tumor components excluding surrounding FLAIR hyperintensity",
        "ET": "gadolinium-enhancing tumor component"},
    # --- hier register: containment-explicit descriptions
    "hier_p1": {
        "WT": "outermost tumor region that contains both the core and the edema",
        "TC": "middle tumor region inside the whole tumor, without the edema",
        "ET": "innermost enhancing region contained within the tumor core"},
    "hier_p2": {
        "WT": "largest region encompassing tumor core and surrounding swelling",
        "TC": "subregion of the whole tumor made of necrosis and enhancement",
        "ET": "smallest subregion, the enhancing part inside the core"},
    "hier_p3": {
        "WT": "full glioma envelope including everything from core to edema",
        "TC": "core located inside the glioma envelope, edema not included",
        "ET": "enhancing tissue nested inside the tumor core"},
    "hier_p4": {
        "WT": "tumor plus edema, the superset of all tumor subregions",
        "TC": "subset of the whole tumor: necrotic plus enhancing tissue",
        "ET": "subset of the tumor core: only the enhancing part"},
}

TRAINABLE_FAMILIES = ["short", "radiol", "hier"]
EVAL_FAMILIES = TRAINABLE_FAMILIES + ["ensemble"]


def get_prompts(family: str, dataset: str = "brats") -> list[str]:
    """[region1, region2, region3] prompt strings for a family/dataset."""
    table = {"brats": {**PROMPT_FAMILIES, **PARAPHRASE_FAMILIES},
             "kits": PROMPT_FAMILIES_KITS}[dataset]
    if family not in table:
        raise KeyError(f"unknown/eval-only family: {family}")
    return [table[family][r] for r in REGIONS]
