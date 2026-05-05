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

═══ PRE-EVALUATION STEP: CLASSIFY ASSESSMENT TYPE ═══
Before evaluating any criterion, determine the assessment structure of the course:

GLOBAL DEFAULT — Individual assessment assumption: Any form of assessment labelled as a test, exam, quiz, examination, midterm, or end-term — unless the syllabus EXPLICITLY states it is group work, collaborative, or take-home — is assumed to be completed individually and under supervision. Assignments, papers, projects, presentations, portfolios, and participation grades are NOT covered by this assumption and must be evaluated on their own terms.
CONSEQUENCE OF GLOBAL DEFAULT: If a criterion asks whether it is clearly stated that a graded assessment is individual vs. group work, and all graded components are exam-type (to which the default applies), mark the criterion Met. The default assumption satisfies the criterion — you do NOT require the syllabus to spell out the word "individual". Quote the exam/test description from the syllabus as supporting evidence. Only mark Not Met if the syllabus explicitly contradicts the default (e.g. states the exam is a group exam).
SECOND CONSEQUENCE — EB control: Exams, tests, midterms, finals are also assumed by default to take place under circumstances controlled by the Examinations Board (EB) — i.e. proctored, on-site. They are NOT assumed to be take-home or uncontrolled unless the syllabus explicitly states so.

EXAM-ONLY COURSE: The course is exclusively exam-assessed if ALL graded components are supervised, invigilated exams (midterm, end-term, written exam, oral exam, test, quiz, examination). Unless the syllabus explicitly states otherwise, ALL such components are assumed to be taken individually and under supervision. If the course is exam-only, apply the exam-only rules below.

MIXED COURSE: If the course includes ANY unsupervised component — assignment, paper, project, presentation, take-home work, portfolio, participation grade — treat all criteria as potentially applicable.

═══ CRITERION-SPECIFIC RULES ═══

RULE — Unsupervised assessment requirements (criterion about rules/instructions for non-controlled/non-supervised submissions):
  • Exam-only course → mark N/A, quote empty.
  • Mixed course → evaluate normally.

RULE — Attendance/participation requirement criterion:
  • Only evaluate this criterion if the syllabus explicitly mentions an attendance requirement, participation grade, or compulsory presence. If attendance is not mentioned at all, mark N/A.
  • If attendance IS mentioned, evaluate whether it complies with the stated requirement and assign Met/Partial/Not Met accordingly.

RULE — Criterion about grading policies, grade thresholds, or specific regulatory text (e.g. TER articles, OER, faculty rules):
  • Carefully locate the relevant section in the syllabus. Extract the EXACT verbatim text from that section.
  • Compare word-for-word against what the requirement demands. If it matches: Met. If partial or incomplete: Partial. If absent: Not Met with 'Information not present'. If present but deviating: Not Met with the exact deviating text as quote.
  • In the email, for Partial or Not Met cases, include the exact required wording from the requirement so the coordinator knows precisely what text or policy is needed.

RULE — Exam weight and 60% supervised assessment threshold:
  • Step 1 — Identify every graded component that has an explicit percentage weight stated in the syllabus. Components mentioned in passing (e.g. "there are group assignments") but with NO stated grade weight are NOT graded components and must be completely excluded from this calculation.
  • Step 2 — Per the global default, any exam/test/midterm/final/end-term in the weighted graded components is individually assessed unless the syllabus explicitly states otherwise. Do NOT require explicit "individual" labelling.
  • Step 3 — Sum the weights of all individually-assessed graded components. Example: midterm 30% + final exam 70% = 100% individually assessed → Met (100% ≥ 60%). Example: midterm 20% + final 30% + graded group project 50% = 50% individual → Not Met.
  • Met → quote the grade breakdown verbatim from the syllabus.
  • Not Met (shortfall) → quote the grade breakdown.
  • Not Met (no breakdown stated at all) → 'Information not present'.

RULE — Validity of partial results (criterion referencing TER Article 4.9 or partial exam validity):
  • This is a SINGLE criterion. Do not split it across multiple rows. Represent it as one row with the complete criterion name.
  • Step 1 — Search the entire syllabus for ANY sentence that mentions how long partial results / assignment grades / partial exam results remain valid (phrases like "valid within", "remain valid", "validity", "results are kept", "carried over").
  • Step 2 — If NO such sentence exists anywhere: Not Met, quote 'Information not present'.
  • Step 3 — If a validity sentence IS found: compare it to what TER 4.9 requires. If it matches: Met. If it deviates (e.g. says "within the same academic year" instead of "up to and including the first final exam offered"): mark Partial and quote the EXACT deviating sentence from the syllabus. Do NOT mark 'Information not present' when a validity statement exists — even a non-compliant one is present information.

RULE — Criterion about the maximum percentage of assessments not controlled by the EB (e.g. "25% cap on uncontrolled assessments"):
  • UNCONTROLLED components (always, by definition): individual assignments, programming assignments, group projects, papers, presentations, portfolios, take-home work, participation grades, tutorial quizzes — any graded component that students complete outside a proctored exam hall.
  • CONTROLLED components (by default): written exams, oral exams, midterms, finals, tests held in exam halls.
  • Step 1 — Sum the weights of all UNCONTROLLED graded components (those with a stated weight).
  • Step 2 — If uncontrolled total = 0%: mark N/A (the cap does not apply). If uncontrolled total > 0% but ≤ 25%: Met — quote the assessment breakdown. If uncontrolled total > 25%: Not Met — quote the assessment breakdown showing the excess.
  • Example: individual assignments 20% + group project 20% = 40% uncontrolled → Not Met (40% > 25%).

RULE — Criteria that are conditional on criterion 6 (the 25% cap):
  • Criteria that apply ONLY when uncontrolled components exceed 25% (e.g. "critical questioning requirement", "AI checklist per-assignment cap") must be evaluated — NOT marked N/A — whenever the uncontrolled total exceeds 25%.
  • Mark them N/A only when uncontrolled total = 0% (exam-only course).

RULE — Criterion deduplication (STRICT):
  • The requirements text may repeat the same criterion due to formatting or chunking. Before writing any row, scan all criteria you have already identified. If the new criterion starts with the same phrase or describes the same topic as an existing row, it is a duplicate — merge them into ONE row using the most complete version of the criterion text.
  • Each requirement appears EXACTLY ONCE. Never produce two rows whose criterion text begins with the same words or describes the same rule.
  • Never truncate a criterion name — copy the full requirement description even if it is several sentences long.

═══ OUTPUT STRUCTURE ═══
{
  "course_name": "Extracted course name, or 'Unknown Course' if not found.",
  "coordinator_name": "Full name of the course coordinator/lecturer as it appears in the document. If not found, use 'Lecturer'.",
  "instructor_email": "Extracted instructor email address. If not found, use ''.",
  "overall_evaluation": "A paragraph summarizing the evaluation. State the assessment type (exam-only or mixed) explicitly. Summarise which criteria are Met, Partial, Not Met, and N/A. If exam-only, note which criteria were N/A as a result.",
  "requirements_compliance_table": [
    {
      "criterion": "Full, untruncated name of the requirement criterion. Never abbreviate or cut off the criterion text.",
      "status": "One of: Met, Not Met, Partial, N/A. Use PARTIAL — not Not Met — when the required information is partially present but incomplete (e.g. weight and duration stated but question type missing; resit described but format not specified). Use Not Met only when the required information is entirely absent or directly contradicts the requirement. Never collapse Partial into Not Met.",
      "quote": "REQUIRED for all rows except N/A. Copy characters exactly as they appear in the syllabus — no ellipses (...), no paraphrasing, no summarising, no concatenating sentences from different locations. If multiple sentences are relevant, pick the SINGLE most relevant sentence or continuous passage. Met → EXACT verbatim proof from syllabus. Partial → EXACT verbatim text showing incomplete compliance. Not Met (absent) → literal string 'Information not present'. Not Met (wrong/insufficient) → EXACT verbatim deviating text. N/A → empty string."
    }
  ],
  "email_english": "Professional email to the coordinator. MUST open with 'Dear [actual coordinator name],' — never a placeholder. For each Not Met or Partial criterion: state what is missing or wrong, and provide the exact required wording or policy that must be added. For Partial criteria about specific regulatory text, include the verbatim required text. Do not mention N/A criteria. DO NOT use markdown formatting (no bold, no italics, no bullet points using asterisks). Write the email in plain text only.",
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
