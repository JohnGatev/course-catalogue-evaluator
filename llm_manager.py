import json
import re
from openai import OpenAI

class LLMManager:
    def __init__(self, api_key: str, base_url: str = None, model: str = "gpt-oss-120b"):
        self.api_key = api_key
        self.base_url = base_url if base_url and str(base_url).strip() else None
        self.model = model

        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        self.client = OpenAI(**client_kwargs)

    def evaluate_course(self, system_prompt: str, requirements_text: str, course_text: str):
        json_instructions = """
CRITICAL OVERRIDE: Output ONLY valid JSON. No markdown fences, no prose outside the JSON object.

The REQUIREMENTS are supplied as JSON. Each requirement carries machine-usable fields you MUST honour:
  • "id" / "number" — the stable identifier; echo "id" back as "criterion_id" in every row.
  • "met_when" — the exact condition under which the requirement is Met.
  • "na_when" — a trigger; when it holds, the requirement is N/A (NOT Not Met).
  • "applies_when" — the requirement is only evaluated when this holds; otherwise it is N/A.
  • "depends_on" — another requirement whose verdict feeds this one.
  • "course_type_na" — course types for which this requirement is automatically N/A.
There is also a top-level "componentControlMatrix" (controlled vs uncontrolled keyword groups) — use it for every weight calculation.

═══ STEP 1 — CLASSIFY THE COURSE TYPE ═══
Decide one of: exam_only | mixed | thesis | writing | research_project.
  • thesis / writing / research_project: the main deliverable is a thesis, paper, essay, research report or writing portfolio (little or no timed exam). For these, every requirement whose "course_type_na" lists the detected type is N/A. In practice usually only critical_questioning_requirement (and academic-integrity statements) stay relevant.
  • exam_only: ALL graded components are supervised exams/tests (written, oral, midterm, final, in-class test, on-site digital exam).
  • mixed: at least one uncontrolled component (take-home, assignment, paper, project, portfolio, presentation, tutorial quiz).
State the detected type in overall_evaluation.

═══ STEP 2 — CLASSIFY EACH GRADED COMPONENT ═══
List every graded component that has an explicit percentage weight. Ignore components mentioned without a stated weight. For each, decide CONTROLLED or UNCONTROLLED using componentControlMatrix:
  • CONTROLLED: written/oral exams, midterms, finals, in-class tests, short tests taken in class, supervised continuous assessment, on-site/digital on-site exams, closed-book invigilated tests. In-class / short / continuous tests taken under supervision are CONTROLLED — never count them as uncontrolled.
  • UNCONTROLLED: take-home assignments, homework, papers, projects, portfolios, presentations, case studies, group work done outside supervision, quizzes during lectures/tutorials/labs.
  • Default: an exam/test/midterm/final is individual and controlled unless the syllabus explicitly says otherwise.
Compute: uncontrolled_total = sum of UNCONTROLLED weights; individual_total = sum of weights NOT explicitly labelled group.

═══ STEP 3 — GRADE EACH REQUIREMENT (use its met_when / na_when / applies_when) ═══

GENERAL N/A RULE: If a requirement's na_when trigger holds, or its applies_when condition is false, output status "N/A" with an empty quote. Never output "Not Met" in those cases.

exam_assessment_method: Met when method + question type (for exam components) + weighting of partial and final exams are stated. Do NOT demand a question type for assignments/projects.

resit_assessment_method: Met when the resit method+weighting are described. If the text says the resit has the SAME format as the original/regular exam, that is sufficient → Met. If exam_assessment_method is Met and the resit mirrors it → Met. N/A when there is no resittable exam/partial component.

duration_exam: Met when a duration is stated for the timed exam component(s). N/A when there is no timed/invigilated exam (thesis, writing course, continuous-assessment-only or assignment-only course).

individual_or_group_work: Met when each graded component is identifiable as individual or group. A component explicitly labelled "group ..." (e.g. "group case", "100% group case") or "individual ..." satisfies this directly. Exam-type components default to individual.

individual_performance_minimum: Met when individual_total ≥ 60%.

uncontrolled_conditions_maximum (the 25% cap): N/A when uncontrolled_total = 0%. Met when 0% < uncontrolled_total ≤ 25%. Not Met when uncontrolled_total > 25% (quote the breakdown showing the excess).

critical_questioning_requirement AND ai_checklist_and_partial_questioning: these apply ONLY when uncontrolled_total > 25%. When uncontrolled_total ≤ 25% (including exam-only / 100% controlled), mark N/A. When > 25%, evaluate normally.

attendance_no_points: N/A when the syllabus does not mention any attendance requirement, participation grade, or compulsory presence — this requirement is a prohibition, not text that must be present. Only evaluate (Met/Partial/Not Met) when attendance IS mentioned.

minimum_requirement_final_exam: Met when an explicit minimum grade threshold (5.00 / 5.50) is stated for the relevant component(s) or the weighted average. Do NOT require the literal phrase "final exam"; a stated 5.5 minimum on assignments or on the weighted average satisfies this.

validity_period_partial_results (TER 4.9): ONE row only. N/A when the course has no partial components (single 100% assessment, or a thesis with no partial results). Otherwise: Met when a validity statement matches "up to and including the first final exam offered", OR the partial exams test other skills than the final exam AND the resittable part is ≥ 70% (then "within the same academic year"/"up to and including the resit" is reasonable → Met). Mark Partial only when a validity statement exists but is genuinely non-compliant.

bonus_scheme_condition AND bonus_scheme_maximum: N/A when NO bonus scheme / bonus points are mentioned anywhere. Only evaluate when a bonus scheme is present.

═══ DEDUPLICATION (STRICT) ═══
Each requirement appears EXACTLY ONCE, keyed by its id. If the requirements text repeats a criterion due to formatting, merge into one row using the most complete text. Never truncate a criterion name.

═══ OUTPUT STRUCTURE ═══
{
  "course_name": "Extracted course name, or 'Unknown Course' if not found.",
  "coordinator_name": "Full name of the course coordinator/lecturer as it appears in the document. If not found, use 'Lecturer'.",
  "instructor_email": "Extracted instructor email address. If not found, use ''.",
  "course_type": "One of: exam_only, mixed, thesis, writing, research_project.",
  "overall_evaluation": "A paragraph summarizing the evaluation. State the detected course type explicitly and the uncontrolled assessment percentage. Summarise which criteria are Met, Partial, Not Met, and N/A, and why any are N/A.",
  "requirements_compliance_table": [
    {
      "criterion_id": "The requirement id from the requirements JSON (e.g. 'duration_exam').",
      "criterion": "Full, untruncated name of the requirement criterion. Never abbreviate or cut off the criterion text.",
      "status": "One of: Met, Not Met, Partial, N/A. Use N/A whenever the requirement's na_when holds or its applies_when is false (e.g. no bonus scheme, no attendance requirement, no timed exam, no partial results, uncontrolled ≤ 25% for the conditional criteria). Use PARTIAL when required information is partially present but incomplete. Use Not Met only when applicable information is entirely absent or directly contradicts the requirement.",
      "quote": "REQUIRED for all rows except N/A. Copy characters exactly as they appear in the syllabus — no ellipses (...), no paraphrasing, no concatenating sentences from different locations. Pick the SINGLE most relevant sentence or continuous passage. Met → EXACT verbatim proof. Partial → EXACT verbatim text showing incomplete compliance. Not Met (absent) → literal string 'Information not present'. Not Met (wrong/insufficient) → EXACT verbatim deviating text. N/A → empty string."
    }
  ],
  "email_english": "Professional email to the coordinator. MUST open with 'Dear [actual coordinator name],' — never a placeholder. For each Not Met or Partial criterion: state what is missing or wrong, and provide the exact required wording or policy that must be added. Do not mention N/A or Met criteria. DO NOT use markdown formatting. Write the email in plain text only.",
  "email_dutch": "Same email in Dutch. MUST open with 'Beste [actual coordinator name],'. DO NOT use markdown formatting. Write in plain text only."
}
        """

        messages = [
            {"role": "system", "content": system_prompt + json_instructions},
            {"role": "user", "content": f"REQUIREMENTS:\n{requirements_text}\n\nCOURSE SYLLABUS:\n{course_text}\n\nEvaluate the syllabus against all requirements. Return JSON only."}
        ]

        call_kwargs = dict(
            model=self.model,
            messages=messages,
            temperature=0.1,
            top_p=0.1,
        )

        try:
            try:
                response = self.client.chat.completions.create(
                    **call_kwargs,
                    response_format={"type": "json_object"}
                )
                if response.choices[0].message.content is None:
                    raise ValueError("API returned None content for JSON mode")
            except Exception:
                response = self.client.chat.completions.create(**call_kwargs)

            content = response.choices[0].message.content
            if content is None:
                return {"error": "The API returned an empty response (content is None)."}

            # Strip markdown fences if present
            content = re.sub(r'^```(?:json)?\s*', '', content.strip(), flags=re.IGNORECASE)
            content = re.sub(r'\s*```$', '', content.strip())

            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                content = json_match.group(0)

            data = json.loads(content)

            # Deterministic guardrails: correct math/N-A gating the LLM is unreliable at.
            try:
                data = _apply_guardrails(data, course_text, requirements_text)
            except Exception:
                pass  # never let a guardrail bug break a successful evaluation

            exact_quotes = [str(row.get("quote", "")) for row in data.get("requirements_compliance_table", []) if row.get("quote") and str(row.get("quote")).strip()]
            data["exact_quotes"] = exact_quotes

            return data

        except Exception as e:
            return {"error": str(e)}

    def generate_emails(self, course_name: str, coordinator_name: str, selected_criteria: list):
        json_instructions = """
CRITICAL OVERRIDE: Output ONLY valid JSON.
{
  "email_english": "Professional email to the coordinator. MUST open with 'Dear [coordinator name],' — never a placeholder. For each criterion provided: state what is missing or wrong, and provide the exact required wording or policy that must be added. DO NOT use markdown formatting (no bold, no italics, no bullet points using asterisks). Write the email in plain text only.",
  "email_dutch": "Same email in Dutch. MUST open with 'Beste [coordinator name],'. DO NOT use markdown formatting. Write in plain text only."
}
        """
        
        criteria_text = json.dumps(selected_criteria, indent=2)
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant writing emails to course coordinators." + json_instructions},
            {"role": "user", "content": f"Course: {course_name}\\nCoordinator: {coordinator_name}\\n\\nCriteria to address in the email:\\n{criteria_text}\\n\\nWrite the emails based ONLY on the provided criteria. If the list is empty, write a positive email stating that all requirements are met."}
        ]
        
        call_kwargs = dict(
            model=self.model,
            messages=messages,
            temperature=0.1,
            top_p=0.1,
        )

        try:
            try:
                response = self.client.chat.completions.create(
                    **call_kwargs,
                    response_format={"type": "json_object"}
                )
                if response.choices[0].message.content is None:
                    raise ValueError("API returned None content for JSON mode")
            except Exception:
                response = self.client.chat.completions.create(**call_kwargs)

            content = response.choices[0].message.content
            if content is None:
                return {"error": "The API returned an empty response."}

            content = re.sub(r'^```(?:json)?\s*', '', content.strip(), flags=re.IGNORECASE)
            content = re.sub(r'\s*```$', '', content.strip())

            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                content = json_match.group(0)

            data = json.loads(content)
            return data

        except Exception as e:
            return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic guardrails
#
# The LLM is unreliable at (a) summing component weights and computing the
# controlled/uncontrolled split, and (b) consistently choosing N/A over Not Met.
# These functions correct exactly those cases from the parsed syllabus. They only
# ever relax a false "Not Met" to Met/N/A per the documented rules, or fix a
# miscomputed percentage — they never invent compliance. When the assessment
# breakdown cannot be parsed confidently, the LLM verdict is left untouched.
# ─────────────────────────────────────────────────────────────────────────────

_STRONG_CONTROLLED = [
    "in-class", "in class", "short test", "on-site", "on site", "closed book",
    "written exam", "written individual exam", "oral exam", "midterm", "mid-term",
    "end-term", "endterm", "final exam", "final examination", "digital on-site",
    "digitaal tentamen", "tentamen", "invigilat", "proctor",
]
_STRONG_UNCONTROLLED = [
    "take-home", "take home", "homework", "assignment", "programming", "paper",
    "essay", "project", "portfolio", "presentation", "case study", "case studies",
    "group case", "group work", "group project", "quiz during", "tutorial quiz",
    "quizzes during", "quizzes to be completed", "lab quiz", "pc lab",
]
_BONUS_KEYWORDS = ["bonus", "bonuspunt", "bonus point"]
_ATTENDANCE_KEYWORDS = ["attendance", "aanwezigheid", "compulsory presence",
                        "mandatory attendance", "participation grade", "verplichte aanwezigheid"]


def _classify_component(ctx):
    """Return (control, is_group) for a percentage's surrounding text window."""
    is_group = "group" in ctx or "groep" in ctx
    for k in _STRONG_CONTROLLED:
        if k in ctx:
            return "controlled", is_group
    for k in _STRONG_UNCONTROLLED:
        if k in ctx:
            return "uncontrolled", is_group
    if any(k in ctx for k in ("exam", "examination", "test", "tentamen", "quiz")):
        return "controlled", is_group
    return "unknown", is_group


def _parse_components(course_text):
    """Parse weighted graded components. Returns list of (weight, control, is_group)."""
    comps = []
    for m in re.finditer(r'(\d+(?:[.,]\d+)?)\s*%', course_text):
        weight = float(m.group(1).replace(",", "."))
        s = max(0, m.start() - 120)
        back = course_text[s:m.start()]
        # Keep only this component's own clause — cut at the last clause boundary
        # so a neighbouring component's keywords don't bleed in.
        boundary = max(back.rfind(d) for d in (";", "\n", "•", "·", "–", "—", ". "))
        if boundary != -1:
            back = back[boundary + 1:]
        fwd = course_text[m.end():min(len(course_text), m.end() + 30)]
        fwd = re.split(r'[;\n•·–—]| \. ', fwd)[0]
        ctx = (back + course_text[m.start():m.end()] + fwd).lower()
        control, is_group = _classify_component(ctx)
        comps.append((weight, control, is_group))
    return comps


def _course_type_na_map(requirements_text):
    """id -> list of course types for which the requirement is auto-N/A."""
    out = {}
    try:
        req = json.loads(requirements_text)

        def walk(items):
            for r in items:
                if isinstance(r, dict) and r.get("id"):
                    out[r["id"]] = r.get("course_type_na", []) or []
                    if r.get("subpoints"):
                        walk(r["subpoints"])
        walk(req.get("requirements", []))
    except Exception:
        pass
    return out


def _set_row(row, status, clear_quote_on_na=True):
    row["status"] = status
    if status == "N/A" and clear_quote_on_na:
        row["quote"] = ""


def _apply_guardrails(data, course_text, requirements_text):
    table = data.get("requirements_compliance_table", [])
    if not isinstance(table, list) or not table:
        return data

    text_lower = (course_text or "").lower()
    by_id = {row.get("criterion_id"): row for row in table if row.get("criterion_id")}
    course_type = (data.get("course_type") or "").strip().lower()

    # 1. Course-type auto-N/A (thesis / writing / research_project)
    if course_type:
        for cid, na_types in _course_type_na_map(requirements_text).items():
            if course_type in na_types and cid in by_id:
                _set_row(by_id[cid], "N/A")

    # 2. Bonus criteria → N/A when no bonus scheme mentioned anywhere
    if not any(k in text_lower for k in _BONUS_KEYWORDS):
        for cid in ("bonus_scheme_condition", "bonus_scheme_maximum"):
            if cid in by_id:
                _set_row(by_id[cid], "N/A")

    # 3. Attendance → N/A when attendance/participation never mentioned
    if not any(k in text_lower for k in _ATTENDANCE_KEYWORDS):
        if "attendance_no_points" in by_id:
            _set_row(by_id["attendance_no_points"], "N/A")

    # 4. Weight-based overrides — only when the breakdown parses to ~100%
    comps = _parse_components(course_text or "")
    total = sum(w for w, _, _ in comps)
    if comps and 90.0 <= total <= 110.0:
        uncontrolled_total = sum(w for w, c, _ in comps if c == "uncontrolled")
        individual_total = sum(w for w, _, g in comps if not g)

        # 25% cap on uncontrolled assessment
        if "uncontrolled_conditions_maximum" in by_id:
            row = by_id["uncontrolled_conditions_maximum"]
            if uncontrolled_total == 0:
                _set_row(row, "N/A")
            elif uncontrolled_total <= 25:
                _set_row(row, "Met", clear_quote_on_na=False)
            else:
                _set_row(row, "Not Met", clear_quote_on_na=False)

        # Conditional criteria apply only when uncontrolled > 25%
        for cid in ("critical_questioning_requirement", "ai_checklist_and_partial_questioning"):
            if cid in by_id and uncontrolled_total <= 25:
                _set_row(by_id[cid], "N/A")

        # 60% individually-assessed minimum
        if "individual_performance_minimum" in by_id:
            _set_row(by_id["individual_performance_minimum"],
                     "Met" if individual_total >= 60 else "Not Met",
                     clear_quote_on_na=False)

    # 5. Validity of partial results → N/A when there are no partial components
    #    (single 100% assessment). Conservative: only when exactly one weight parsed.
    if "validity_period_partial_results" in by_id:
        single = len(comps) <= 1 or (comps and max(w for w, _, _ in comps) >= 100)
        if single:
            _set_row(by_id["validity_period_partial_results"], "N/A")

    return data
