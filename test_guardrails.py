"""Unit tests for the deterministic guardrails in llm_manager.

These exercise the % math and N/A gating using the exact failing examples from the
Programme Office feedback, with no LLM call. Run: python -m pytest test_guardrails.py
or simply: python test_guardrails.py
"""
from llm_manager import _apply_guardrails, _parse_components, _classify_component

REQUIREMENTS = open("requirements_english.txt", encoding="utf-8").read()


def _row(cid, status="Not Met", quote="x"):
    return {"criterion_id": cid, "criterion": cid, "status": status, "quote": quote}


def _status(data, cid):
    for r in data["requirements_compliance_table"]:
        if r.get("criterion_id") == cid:
            return r["status"]
    raise AssertionError(f"row {cid} missing")


def run(data, text, ctype=None):
    if ctype:
        data["course_type"] = ctype
    return _apply_guardrails(data, text, REQUIREMENTS)


def test_component_classification():
    # Supervised in-class / short tests are CONTROLLED, not uncontrolled.
    assert _classify_component("continuous assessment in the form of short tests (10%)")[0] == "controlled"
    assert _classify_component("in-class test (10%)")[0] == "controlled"
    assert _classify_component("final individual exam (80%)")[0] == "controlled"
    # Tutorial/lab quizzes and assignments are UNCONTROLLED.
    assert _classify_component("quizzes to be completed during pc lab/tutorial (15%)")[0] == "uncontrolled"
    assert _classify_component("programming assignments (30%)")[0] == "uncontrolled"
    # Group detection.
    assert _classify_component("the final grade is based 100% on a group case")[1] is True


def test_crit6_stochastic_all_controlled():
    # Stochastic Models: short tests 10% + in-class test 10% + final exam 80% = 0% uncontrolled.
    text = ("Continuous assessment in the form of short tests (10%); "
            "In-class test (10%); Final individual exam (80%).")
    data = {"requirements_compliance_table": [
        _row("uncontrolled_conditions_maximum"),
        _row("individual_performance_minimum"),
    ]}
    run(data, text, "exam_only")
    assert _status(data, "uncontrolled_conditions_maximum") == "N/A"
    assert _status(data, "individual_performance_minimum") == "Met"


def test_crit8_sustainable_finance_100pct_controlled():
    # Sustainable Finance: 100% individual digital on-site exam → conditional criteria N/A.
    text = "The final grade is based 100% on an individual digital on-site exam."
    data = {"requirements_compliance_table": [
        _row("ai_checklist_and_partial_questioning"),
        _row("critical_questioning_requirement"),
        _row("uncontrolled_conditions_maximum"),
    ]}
    run(data, text, "exam_only")
    assert _status(data, "ai_checklist_and_partial_questioning") == "N/A"
    assert _status(data, "critical_questioning_requirement") == "N/A"
    assert _status(data, "uncontrolled_conditions_maximum") == "N/A"


def test_crit7_conditional_na_when_under_25():
    # 15% uncontrolled quizzes (<=25%) → critical questioning N/A, 25% cap Met.
    text = ("Individual written exam, theory 85%; "
            "quizzes to be completed during PC lab/tutorial (15%).")
    data = {"requirements_compliance_table": [
        _row("critical_questioning_requirement"),
        _row("uncontrolled_conditions_maximum"),
    ]}
    run(data, text, "mixed")
    assert _status(data, "critical_questioning_requirement") == "N/A"
    assert _status(data, "uncontrolled_conditions_maximum") == "Met"


def test_bonus_na_when_absent():
    text = "Final exam 100%, two hours, individual, closed book."
    data = {"requirements_compliance_table": [
        _row("bonus_scheme_condition"),
        _row("bonus_scheme_maximum"),
    ]}
    run(data, text, "exam_only")
    assert _status(data, "bonus_scheme_condition") == "N/A"
    assert _status(data, "bonus_scheme_maximum") == "N/A"


def test_bonus_kept_when_present():
    text = "A bonus of up to 0.5 points may be earned. Final exam 100%."
    data = {"requirements_compliance_table": [_row("bonus_scheme_condition", "Not Met")]}
    run(data, text, "exam_only")
    assert _status(data, "bonus_scheme_condition") == "Not Met"  # not forced to N/A


def test_attendance_na_when_absent():
    text = "Final exam 100%."
    data = {"requirements_compliance_table": [_row("attendance_no_points")]}
    run(data, text, "exam_only")
    assert _status(data, "attendance_no_points") == "N/A"


def test_validity_na_single_100pct():
    # Oriëntatie Fiscale Economie: single 100% exam + resit → no partial results → N/A.
    text = "Digitaal tentamen (100%) van 2 uur, individueel, open vragen, gesloten boek."
    data = {"requirements_compliance_table": [_row("validity_period_partial_results")]}
    run(data, text, "exam_only")
    assert _status(data, "validity_period_partial_results") == "N/A"


def test_thesis_auto_na():
    # Thesis course → exam-centric criteria auto-N/A, critical questioning stays.
    text = "Master thesis, individually written and assessed."
    data = {"requirements_compliance_table": [
        _row("duration_exam"),
        _row("bonus_scheme_maximum"),
        _row("critical_questioning_requirement", "Met", "students subjected to critical questioning"),
    ]}
    run(data, text, "thesis")
    assert _status(data, "duration_exam") == "N/A"
    assert _status(data, "bonus_scheme_maximum") == "N/A"
    assert _status(data, "critical_questioning_requirement") == "Met"  # not in course_type_na


def test_no_override_when_breakdown_unparseable():
    # Weights don't sum to ~100% → weight-based overrides skipped, model verdict kept.
    text = "Some exam worth 40% and another component."
    data = {"requirements_compliance_table": [_row("uncontrolled_conditions_maximum", "Partial")]}
    run(data, text, "mixed")
    assert _status(data, "uncontrolled_conditions_maximum") == "Partial"


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
