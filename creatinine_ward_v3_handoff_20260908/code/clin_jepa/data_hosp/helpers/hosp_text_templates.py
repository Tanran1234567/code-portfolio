"""Text rendering for hospital-scope events.

The hospital pipeline reuses the mimic-code concept names (``chemistry``,
``complete_blood_count``, ``bg``, ``coagulation``, ``enzyme``, ``vitalsign``,
``gcs``, ``sofa``), so :func:`clin_jepa.data.helpers.text_templates.format_event`
already renders most of its events identically to the ICU pipeline — which
matters, because the encoder is warm-started from ICU-pretrained weights and
should not see a different surface form for the same measurement.

Only the sources with no ICU counterpart are handled here: OMR ward vitals,
care-setting transfers, microbiology, hospital-wide prescriptions, eMAR
administrations, and coded procedures.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from clin_jepa.data.helpers.event_ordering import _sort_key
from clin_jepa.data.helpers.text_templates import (
    format_event as _format_concept_event,
)
from clin_jepa.data.helpers.text_templates import format_numeric

# Sources whose rendering is defined in this module rather than inherited.
HOSP_SOURCES = frozenset({
    "omr", "careunit", "micro", "rx", "emar_admin", "proc_icd", "diagnoses",
    "admission_context", "prior_diagnoses",
})

HOSP_DISPLAY_NAMES: dict[tuple[str, str], str] = {
    ("omr", "sbp"): "Systolic BP (ward)",
    ("omr", "dbp"): "Diastolic BP (ward)",
    ("omr", "weight"): "Weight",
    ("omr", "bmi"): "BMI",
    ("omr", "height"): "Height",
}

# Where hospital-only variables sort within a bin. The ICU ordering puts
# vitals first (0-99) and interventions last (900+); ward vitals join the
# vitals block, care setting leads the bin because it frames everything
# else, and microbiology sits after labs.
HOSP_SORT_ORDER: dict[tuple[str, str], int] = {
    ("careunit", "care_unit"): -10,
    ("omr", "sbp"): 10,
    ("omr", "dbp"): 11,
    ("omr", "weight"): 90,
    ("omr", "bmi"): 91,
    ("omr", "height"): 92,
    ("micro", "culture"): 700,
    ("rx", "*"): 900,
    ("emar_admin", "*"): 910,
    ("proc_icd", "*"): 920,
}


def hosp_sort_key(source_table: str, variable_name: str) -> int:
    """Canonical within-bin sort priority, extending the ICU ordering."""
    key = HOSP_SORT_ORDER.get((source_table, variable_name))
    if key is not None:
        return key
    key = HOSP_SORT_ORDER.get((source_table, "*"))
    if key is not None:
        return key
    return _sort_key(source_table, variable_name)


def _has_numeric(value) -> bool:
    return value is not None and not (
        isinstance(value, float) and np.isnan(value)
    )


def _has_text(value) -> bool:
    return value is not None and str(value) not in ("", "nan", "None", "<NA>")


def format_hosp_event(
    source_table: str,
    variable_name: str,
    numeric_value: Optional[float],
    text_value: Optional[str],
    unit: str,
) -> Optional[str]:
    """Render one event as a text fragment, or None if it carries no value.

    Delegates every mimic-code concept source to the ICU formatter so that
    shared measurements keep an identical surface form.
    """
    if source_table not in HOSP_SOURCES:
        return _format_concept_event(
            source_table, variable_name, numeric_value, text_value, unit,
        )

    num_ok = _has_numeric(numeric_value)
    txt_ok = _has_text(text_value)
    if not num_ok and not txt_ok:
        return None

    if source_table == "careunit":
        return f"Care unit: {text_value}"

    if source_table == "admission_context":
        return f"Admission context: {text_value}"

    if source_table == "prior_diagnoses":
        label = str(text_value) if txt_ok else variable_name
        return f"Known prior diagnosis: {label} ({variable_name})"

    if source_table == "diagnoses":
        if txt_ok:
            return f"Diagnosis: {text_value} ({variable_name})"
        return f"Diagnosis: {variable_name}"

    if source_table == "micro":
        # "Culture (BLOOD CULTURE): STAPH AUREUS COAG +" / "... : no growth"
        spec = variable_name.replace("_", " ").title()
        organism = str(text_value) if txt_ok else "no growth"
        return f"Culture ({spec}): {organism}"

    if source_table == "omr":
        name = HOSP_DISPLAY_NAMES.get(
            (source_table, variable_name), variable_name.replace("_", " ").title(),
        )
        if num_ok:
            return f"{name}: {format_numeric(numeric_value)} {unit}".rstrip()
        return f"{name}: {text_value}"

    if source_table == "rx":
        # variable_name is the drug; text_value carries "dose unit route".
        parts = [variable_name]
        if txt_ok:
            parts.append(str(text_value))
        return f"Medication ordered: {' '.join(parts)}"

    if source_table == "emar_admin":
        if txt_ok and str(text_value) != "Administered":
            return f"Medication given: {variable_name} ({text_value})"
        return f"Medication given: {variable_name}"

    if source_table == "proc_icd":
        label = str(text_value) if txt_ok else variable_name
        return f"Procedure: {label} ({variable_name})"

    return None
