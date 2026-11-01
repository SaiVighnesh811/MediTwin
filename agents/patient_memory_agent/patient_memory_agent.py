"""
agents/patient_memory_agent/patient_memory_agent.py
Patient Memory Agent for MediTwin AI.

Stores longitudinal patient events (diagnoses, symptoms, medications, lab values,
risk assessments, imaging findings, documents).
Default backend: JSON files in data/patient_memory/<patient_id>.json.
Pluggable interface for future MongoDB/SQL backend.
"""

import os
import json
import datetime
from typing import Any, Dict, List, Optional, Union

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_STORAGE_DIR = os.path.join(PROJECT_ROOT, "data", "patient_memory")

VALID_EVENT_TYPES = {
    "diagnosis",
    "symptom",
    "medication",
    "lab_value",
    "risk_assessment",
    "imaging_finding",
    "document",
}


def _current_timestamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _normalize_iso(ts: Optional[Union[str, datetime.datetime]]) -> tuple[str, bool]:
    """Returns (iso_string, is_uncertain)."""
    if ts is None or not str(ts).strip():
        return _current_timestamp(), True
    if isinstance(ts, datetime.datetime):
        return ts.isoformat(), False
    ts_str = str(ts).strip()
    try:
        # validate parseable
        dt = datetime.datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return dt.isoformat(), False
    except Exception:
        # Fallback to current ingest timestamp with uncertain flag
        return _current_timestamp(), True


class PatientMemoryAgent:
    """Storage and querying interface for patient historical events."""

    def __init__(self, storage_dir: Optional[str] = None):
        self.storage_dir = storage_dir or DEFAULT_STORAGE_DIR
        os.makedirs(self.storage_dir, exist_ok=True)

    def _patient_path(self, patient_id: str) -> str:
        safe_id = "".join(c for c in patient_id if c.isalnum() or c in ("-", "_"))
        return os.path.join(self.storage_dir, f"{safe_id}.json")

    def _load_events(self, patient_id: str) -> List[Dict[str, Any]]:
        path = self._patient_path(patient_id)
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("events", []) if isinstance(data, dict) else []
        except Exception:
            return []

    def _save_events(self, patient_id: str, events: List[Dict[str, Any]]) -> None:
        path = self._patient_path(patient_id)
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump({"patient_id": patient_id, "events": events}, f, indent=2)
        os.replace(tmp_path, path)

    def add_event(
        self,
        patient_id: str,
        event_type: str,
        data: Dict[str, Any],
        timestamp: Optional[Union[str, datetime.datetime]] = None,
        source: str = "manual",
    ) -> Dict[str, Any]:
        """
        Append an event to patient timeline.
        Ignores exact duplicates (matching event_type, data, and normalized timestamp/source).
        """
        if event_type not in VALID_EVENT_TYPES:
            raise ValueError(f"Invalid event_type '{event_type}'. Must be one of {VALID_EVENT_TYPES}")

        iso_ts, is_uncertain = _normalize_iso(timestamp)
        payload = dict(data) if isinstance(data, dict) else {"value": data}

        event: Dict[str, Any] = {
            "event_type": event_type,
            "data": payload,
            "timestamp": iso_ts,
            "date_uncertain": is_uncertain,
            "source": source,
        }

        events = self._load_events(patient_id)

        # Exact duplicate detection: same event_type, source, data payload, and timestamp
        for existing in events:
            if (
                existing.get("event_type") == event_type
                and existing.get("source") == source
                and existing.get("data") == payload
                and existing.get("timestamp") == iso_ts
            ):
                return existing  # ignore exact duplicate, return existing

        events.append(event)
        self._save_events(patient_id, events)
        return event

    def get_timeline(
        self,
        patient_id: str,
        types: Optional[List[str]] = None,
        since: Optional[Union[str, datetime.datetime]] = None,
        until: Optional[Union[str, datetime.datetime]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Returns chronological events sorted ascending by timestamp.
        Can filter by list of event types and since/until date range.
        """
        events = self._load_events(patient_id)
        filtered = []

        since_str = since.isoformat() if isinstance(since, datetime.datetime) else (str(since) if since else None)
        until_str = until.isoformat() if isinstance(until, datetime.datetime) else (str(until) if until else None)

        for ev in events:
            if types and ev.get("event_type") not in types:
                continue
            ts = ev.get("timestamp", "")
            if since_str and ts < since_str:
                continue
            if until_str and ts > until_str:
                continue
            filtered.append(ev)

        # Chronological sort (stable sort on timestamp)
        filtered.sort(key=lambda x: x.get("timestamp", ""))
        return filtered

    def latest(self, patient_id: str, event_type: str) -> Optional[Dict[str, Any]]:
        """Returns the most recent event of the given type, or None."""
        timeline = self.get_timeline(patient_id, types=[event_type])
        return timeline[-1] if timeline else None

    def get_trend(self, patient_id: str, test_name: str) -> List[Dict[str, Any]]:
        """
        Returns chronological list of lab_value events matching test_name (case-insensitive).
        Each record has {timestamp, test, value, unit, reference}.
        """
        timeline = self.get_timeline(patient_id, types=["lab_value"])
        trends = []
        name_lower = test_name.strip().lower()
        for ev in timeline:
            d = ev.get("data", {})
            t_name = str(d.get("test", "")).strip().lower()
            if t_name == name_lower or name_lower in t_name:
                trends.append({
                    "timestamp": ev.get("timestamp"),
                    "date_uncertain": ev.get("date_uncertain", False),
                    "test": d.get("test"),
                    "value": d.get("value"),
                    "unit": d.get("unit"),
                    "reference": d.get("reference"),
                })
        return trends

    def summary(self, patient_id: str) -> Dict[str, Any]:
        """Provides an aggregated clinical overview of patient history."""
        timeline = self.get_timeline(patient_id)
        diagnoses = set()
        symptoms = set()
        medications = set()
        latest_labs: Dict[str, Any] = {}
        imaging_history: List[Dict[str, Any]] = []
        risk_assessments: List[Dict[str, Any]] = []

        for ev in timeline:
            etype = ev.get("event_type")
            d = ev.get("data", {})
            if etype == "diagnosis":
                name = d.get("diagnosis") or d.get("name")
                if name:
                    diagnoses.add(str(name))
            elif etype == "symptom":
                sym = d.get("symptom") or d.get("name")
                if sym:
                    symptoms.add(str(sym))
            elif etype == "medication":
                med = d.get("medicine") or d.get("name")
                if med:
                    medications.add(str(med))
            elif etype == "lab_value":
                tname = d.get("test")
                if tname:
                    latest_labs[str(tname)] = {
                        "value": d.get("value"),
                        "unit": d.get("unit"),
                        "timestamp": ev.get("timestamp"),
                    }
            elif etype == "imaging_finding":
                imaging_history.append({
                    "finding": d.get("finding") or d.get("predicted_class"),
                    "probability": d.get("probability") or d.get("pneumonia_probability"),
                    "timestamp": ev.get("timestamp"),
                })
            elif etype == "risk_assessment":
                risk_assessments.append({
                    "risk_level": d.get("risk_level") or d.get("predicted_class"),
                    "probabilities": d.get("probabilities"),
                    "timestamp": ev.get("timestamp"),
                })

        return {
            "patient_id": patient_id,
            "total_events": len(timeline),
            "diagnoses": sorted(list(diagnoses)),
            "symptoms": sorted(list(symptoms)),
            "medications": sorted(list(medications)),
            "latest_labs": latest_labs,
            "imaging_findings": imaging_history[-3:],  # last 3
            "latest_risk": risk_assessments[-1] if risk_assessments else None,
        }


# ==============================================================================
# Adapters from existing agents
# ==============================================================================

def adapt_ocr_result(
    patient_id: str,
    ocr_result: Dict[str, Any],
    timestamp: Optional[str] = None,
    source: str = "ocr_agent",
    memory_agent: Optional[PatientMemoryAgent] = None,
    raw_text: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Turns OCR extract_fields dictionary into timeline events.
    Missing/None fields are skipped and recorded as 'field_missing'.
    If medicine is None in a prescription, runs the NLP agent fallback on raw_text
    (or ocr_result.get('_raw_text')) to extract MEDICATION entities, recording source='nlp_fallback'.
    Unknown date defaults to ingest time with date_uncertain=True.
    """
    mem = memory_agent or PatientMemoryAgent()
    events_added = []
    doc_type = ocr_result.get("type")

    # Determine input text for fallback
    text_to_inspect = raw_text or ocr_result.get("_raw_text")

    if doc_type == "lab":
        # Required fields: test, value
        test_val = ocr_result.get("test")
        val = ocr_result.get("value")
        if not test_val:
            ev_miss = mem.add_event(
                patient_id,
                "document",
                {"status": "field_missing", "field": "test", "raw": ocr_result},
                timestamp=timestamp,
                source=source,
            )
            events_added.append(ev_miss)
        elif not val:
            ev_miss = mem.add_event(
                patient_id,
                "document",
                {"status": "field_missing", "field": "value", "test": test_val, "raw": ocr_result},
                timestamp=timestamp,
                source=source,
            )
            events_added.append(ev_miss)
        else:
            payload = {
                "test": test_val,
                "value": val,
                "unit": ocr_result.get("unit"),
                "reference": ocr_result.get("reference"),
            }
            ev = mem.add_event(patient_id, "lab_value", payload, timestamp=timestamp, source=source)
            events_added.append(ev)

    elif doc_type == "prescription":
        med = ocr_result.get("medicine")
        event_src = source
        # NLP Fallback if medicine is None and text is available
        if not med and text_to_inspect:
            try:
                from agents.nlp_agent.nlp_agent import extract_entities
                nlp_out = extract_entities(text_to_inspect)
                med_candidates = [
                    m["text"] for m in nlp_out.get("medications", [])
                    if m.get("text", "").strip().lower() not in {"tab", "cap", "inj", "syp", "tablet", "capsule", "food"}
                ]
                if med_candidates:
                    med = med_candidates[0]
                    event_src = "nlp_fallback"
            except Exception:
                pass

        if not med:
            ev_miss = mem.add_event(
                patient_id,
                "document",
                {"status": "field_missing", "field": "medicine", "raw": ocr_result},
                timestamp=timestamp,
                source=source,
            )
            events_added.append(ev_miss)
        else:
            payload = {
                "medicine": med,
                "dosage": ocr_result.get("dosage"),
                "frequency": ocr_result.get("frequency"),
                "duration": ocr_result.get("duration"),
            }
            ev = mem.add_event(patient_id, "medication", payload, timestamp=timestamp, source=event_src)
            events_added.append(ev)
    else:
        # Generic document
        ev = mem.add_event(
            patient_id,
            "document",
            {"status": "unrecognized_type", "raw": ocr_result},
            timestamp=timestamp,
            source=source,
        )
        events_added.append(ev)

    return events_added


def adapt_nlp_result(
    patient_id: str,
    nlp_result: Dict[str, Any],
    timestamp: Optional[str] = None,
    source: str = "nlp_agent",
    memory_agent: Optional[PatientMemoryAgent] = None,
) -> List[Dict[str, Any]]:
    """
    Turns NLP extract_entities dictionary into diagnosis, symptom, medication,
    and procedure timeline events.
    """
    mem = memory_agent or PatientMemoryAgent()
    events_added = []

    for disease in nlp_result.get("diseases", []):
        ev = mem.add_event(patient_id, "diagnosis", {"diagnosis": disease}, timestamp=timestamp, source=source)
        events_added.append(ev)

    for symptom in nlp_result.get("symptoms", []):
        ev = mem.add_event(patient_id, "symptom", {"symptom": symptom}, timestamp=timestamp, source=source)
        events_added.append(ev)

    for medication in nlp_result.get("medications", []):
        ev = mem.add_event(patient_id, "medication", {"medicine": medication}, timestamp=timestamp, source=source)
        events_added.append(ev)

    for procedure in nlp_result.get("procedures", []):
        ev = mem.add_event(patient_id, "document", {"procedure": procedure}, timestamp=timestamp, source=source)
        events_added.append(ev)

    return events_added


def adapt_risk_result(
    patient_id: str,
    risk_result: Dict[str, Any],
    timestamp: Optional[str] = None,
    source: str = "risk_agent",
    memory_agent: Optional[PatientMemoryAgent] = None,
) -> Dict[str, Any]:
    """Turns Risk Prediction output into a risk_assessment timeline event."""
    mem = memory_agent or PatientMemoryAgent()
    payload = {
        "risk_level": risk_result.get("risk_level") or risk_result.get("predicted_class"),
        "probabilities": risk_result.get("probabilities") or risk_result.get("risk_probabilities"),
        "features_used": risk_result.get("features_used", 21),
    }
    return mem.add_event(patient_id, "risk_assessment", payload, timestamp=timestamp, source=source)


def adapt_medical_image_result(
    patient_id: str,
    image_result: Dict[str, Any],
    timestamp: Optional[str] = None,
    source: str = "medical_image_agent",
    memory_agent: Optional[PatientMemoryAgent] = None,
) -> Dict[str, Any]:
    """Turns Medical Image Agent output into an imaging_finding timeline event."""
    mem = memory_agent or PatientMemoryAgent()
    payload = {
        "finding": image_result.get("predicted_class"),
        "pneumonia_probability": image_result.get("pneumonia_probability") or image_result.get("probability"),
        "normal_probability": image_result.get("normal_probability"),
        "heatmap_path": image_result.get("heatmap_path"),
        "disclaimer": image_result.get("disclaimer"),
    }
    return mem.add_event(patient_id, "imaging_finding", payload, timestamp=timestamp, source=source)

