import streamlit as st
import streamlit.components.v1 as components
import base64
import html as html_lib
import re as re_lib
import os
import json
import tempfile
import pandas as pd
from datetime import datetime
from document_processor import extract_text, extract_url_text, highlight_pdf
from rag_engine import (embed_requirements, get_full_requirements_text,
                        embed_course_content, get_full_course_text,
                        generate_course_id, query_course_content)
from llm_manager import LLMManager
from storage import save_evaluation, get_all_evaluations, delete_evaluation, get_setting, save_setting, init_db

def _find_quote(quote, text):
    """Return first (start, end) for quote in text, or (-1, -1). Used for has_q checks."""
    results = _find_all_quotes(quote, text)
    return results[0] if results else (-1, -1)


def _find_all_quotes(quote, text):
    """Return list of (start, end) for every occurrence of quote in text.
    Tries exact → whitespace-normalised → fuzzy (rapidfuzz)."""
    results = []
    q = quote.strip()
    if not q:
        return results

    # 1. All exact matches
    start = 0
    while True:
        pos = text.find(q, start)
        if pos == -1:
            break
        results.append((pos, pos + len(q)))
        start = pos + 1
    if results:
        return results

    # 2. All whitespace-normalised matches
    pattern = re_lib.sub(r'\s+', r'\\s+', re_lib.escape(q))
    for m in re_lib.finditer(pattern, text):
        results.append((m.start(), m.end()))
    if results:
        return results

    # 3. Best fuzzy match (single result — rapidfuzz partial_ratio_alignment)
    try:
        from rapidfuzz import fuzz
        alignment = fuzz.partial_ratio_alignment(q, text)
        if alignment.score >= 78:
            results.append((alignment.dest_start, alignment.dest_end))
    except Exception:
        pass

    if results:
        return results

    # 4. Ellipsis-split fallback: LLM used '...' to summarise multi-location evidence.
    #    Split on ellipses, search each non-trivial fragment independently.
    if '...' in q:
        fragments = re_lib.split(r'\.\.\.', q)
        # Strip leading/trailing punctuation/whitespace that bleeds in from the split
        fragments = [re_lib.sub(r'^[\s;,.()\[\]]+|[\s;,.()\[\]]+$', '', f) for f in fragments]
        fragments = [f for f in fragments if len(f) > 8]
        seen = set()
        for frag in fragments:
            for span in _find_all_quotes(frag, text):
                if span not in seen:
                    seen.add(span)
                    results.append(span)

    return results


def build_interactive_viewer(table_data, text):
    """Render side-by-side requirements table + text viewer with hover-to-highlight."""
    BADGE_BG  = {"Met": "#1e8a3a", "Not Met": "#cc2222", "Partial": "#cc8800", "N/A": "#888"}
    BADGE_FG  = {"Met": "white",   "Not Met": "white",   "Partial": "white",   "N/A": "white"}
    ABSENT    = "Information not present"

    # Locate ALL occurrences of every quote (handles overlapping / nested / repeated quotes)
    all_regions = []  # (start, end, row_idx)
    for i, row in enumerate(table_data):
        q = row.get("quote", "").strip()
        if not q or q == ABSENT:
            continue
        for start, end in _find_all_quotes(q, text):
            all_regions.append((start, end, i))
    matched_rows = set(r[2] for r in all_regions)

    # Segment-based HTML: split text at every region boundary, assign multi-class spans
    # This correctly handles nested, overlapping, and repeated quote regions.
    boundaries = sorted(set([0, len(text)] + [r[0] for r in all_regions] + [r[1] for r in all_regions]))
    parts = []
    for j in range(len(boundaries) - 1):
        seg_start = boundaries[j]
        seg_end   = boundaries[j + 1]
        if seg_start >= seg_end:
            continue
        covering = sorted(set(r[2] for r in all_regions if r[0] <= seg_start and r[1] >= seg_end))
        seg_text = html_lib.escape(text[seg_start:seg_end])
        if covering:
            cls = "q " + " ".join(f"q{idx}" for idx in covering)
            parts.append(f'<span class="{cls}">{seg_text}</span>')
        else:
            parts.append(seg_text)
    text_html = "".join(parts)

    # Build table rows
    rows_html = []
    for i, row in enumerate(table_data):
        num    = i + 1
        crit   = html_lib.escape(row.get("criterion", ""))
        status = row.get("status", "")
        quote  = row.get("quote", "").strip()
        bb = BADGE_BG.get(status, "#6c757d")
        bf = BADGE_FG.get(status, "white")

        is_absent  = (quote == ABSENT)
        has_q      = (not is_absent) and bool(quote) and i in matched_rows
        cursor_css = "pointer" if has_q else "default"

        if is_absent:
            preview_html = '<span style="color:#cc2222;font-style:italic;font-size:11px;">Information not present</span>'
        elif quote:
            preview_html = f'<span style="font-style:italic;font-size:11px;color:#555;">{html_lib.escape(quote[:95])}{"…" if len(quote) > 95 else ""}</span>'
        else:
            preview_html = '<span style="color:#aaa;font-size:11px;">—</span>'

        rows_html.append(
            f'<tr class="rr" data-idx="{i}" onmouseenter="hl({i})" '
            f'style="cursor:{cursor_css}; border-bottom:1px solid #eee;">'
            f'<td style="padding:7px 6px;text-align:center;vertical-align:top;font-size:12px;color:#888;font-weight:600;width:28px;">{num}</td>'
            f'<td style="padding:7px 10px;vertical-align:top;font-size:13px;">{crit}</td>'
            f'<td style="padding:7px 6px;text-align:center;vertical-align:top;white-space:nowrap;">'
            f'<span style="background:{bb};color:{bf};padding:2px 8px;border-radius:10px;font-size:11px;">{status}</span></td>'
            f'<td style="padding:7px 10px;vertical-align:top;">{preview_html}</td>'
            f'</tr>'
        )

    legend = (
        '<div style="display:flex;gap:8px;padding:5px 12px;font-size:11px;align-items:center;border-bottom:1px solid #ddd;background:#fafafa;flex-shrink:0;">'
        '<span style="background:#1e8a3a;color:white;padding:1px 8px;border-radius:3px;">Met</span>'
        '<span style="background:#cc2222;color:white;padding:1px 8px;border-radius:3px;">Not Met</span>'
        '<span style="background:#cc8800;color:white;padding:1px 8px;border-radius:3px;">Partial</span>'
        '<span style="background:#888;color:white;padding:1px 8px;border-radius:3px;">N/A</span>'
        '<span style="color:#888;margin-left:6px;">Hover row → scroll to quote</span>'
        '</div>'
    )

    return f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><style>
*{{box-sizing:border-box;margin:0;padding:0}}
html,body{{height:100%;overflow:hidden;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#fff}}
.wrap{{display:flex;gap:10px;height:100%;}}
.pane{{display:flex;flex-direction:column;border:1px solid #ddd;border-radius:6px;overflow:hidden}}
.pane-l{{flex:0 0 46%}}.pane-r{{flex:1}}
.ph{{padding:8px 12px;background:#f5f5f5;border-bottom:1px solid #ddd;font-weight:600;font-size:13px;color:#444;flex-shrink:0}}
.pb{{flex:1;overflow-y:scroll;min-height:0}}
table{{width:100%;border-collapse:collapse}}
thead tr{{background:#f8f8f8;position:sticky;top:0;z-index:5}}
th{{padding:7px 10px;text-align:left;font-size:12px;color:#666;border-bottom:2px solid #ddd;white-space:nowrap}}
.rr:hover{{background:#f0f4ff}}.rr.ar{{background:#e8f0fe}}
.q{{border-radius:2px}}.q.hl{{outline:3px solid #FF6B00;outline-offset:2px;position:relative;z-index:1}}
#txt{{padding:12px 14px;white-space:pre-wrap;font-family:'Courier New',monospace;font-size:12px;line-height:1.65;color:#333}}
</style></head><body>
<div class="wrap">
  <div class="pane pane-l">
    <div class="ph">Requirements Compliance</div>
    {legend}
    <div class="pb"><table>
      <thead><tr>
        <th style="width:28px;text-align:center;">#</th>
        <th style="width:35%">Criterion</th>
        <th style="width:13%">Status</th>
        <th>Supporting Quote</th>
      </tr></thead>
      <tbody>{"".join(rows_html)}</tbody>
    </table></div>
  </div>
  <div class="pane pane-r">
    <div class="ph">Course Document</div>
    <div class="pb" id="sb"><div id="txt">{text_html}</div></div>
  </div>
</div>
<script>
function hl(i){{
  document.querySelectorAll('.rr').forEach(r=>r.classList.remove('ar'));
  document.querySelectorAll('.q').forEach(s=>s.classList.remove('hl'));
  var row=document.querySelector('.rr[data-idx="'+i+'"]');
  if(row) row.classList.add('ar');
  var spans=document.querySelectorAll('.q'+i);
  spans.forEach(s=>s.classList.add('hl'));
  if(spans.length>0){{
    var sb=document.getElementById('sb');
    var top=spans[0].getBoundingClientRect().top - sb.getBoundingClientRect().top + sb.scrollTop - (sb.clientHeight/2);
    sb.scrollTo({{top:top, behavior:'smooth'}});
  }}
}}
function clr(){{
  document.querySelectorAll('.rr').forEach(r=>r.classList.remove('ar'));
  document.querySelectorAll('.q').forEach(s=>s.classList.remove('hl'));
}}
// Single mouseleave on tbody — avoids per-row gap flicker
document.querySelector('tbody').addEventListener('mouseleave', clr);
</script></body></html>"""


# Layout config
st.set_page_config(layout="wide", page_title="Course Catalog Evaluator")

# Initialize DB on load
init_db()

# Session state initialization
if 'courses' not in st.session_state:
    st.session_state.courses = []
if 'current_idx' not in st.session_state:
    st.session_state.current_idx = 0

def _read_text_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def initialize_defaults():
    if not get_setting("system_prompt"):
        if os.path.exists("system_prompt_english.txt"):
            save_setting("system_prompt", _read_text_file("system_prompt_english.txt"))
        elif os.path.exists("system_prompt_english.pdf"):
            save_setting("system_prompt", extract_text("system_prompt_english.pdf"))
        else:
            save_setting("system_prompt", "You are an expert evaluator for university courses.")

    if not get_setting("requirements_text"):
        if os.path.exists("requirements_english.txt"):
            text = _read_text_file("requirements_english.txt")
            save_setting("requirements_text", text)
            embed_requirements(text)
        elif os.path.exists("requirements_english.pdf"):
            text = extract_text("requirements_english.pdf")
            save_setting("requirements_text", text)
            embed_requirements(text)
        else:
            save_setting("requirements_text", "1. Default requirement")
            embed_requirements("1. Default requirement")

initialize_defaults()

# --- SIDEBAR: Settings & Uploads ---
with st.sidebar:
    st.header("Configuration")
    
    # API Settings
    api_key = st.text_input("API Key (gpt-oss-120b or OpenAI)", type="password")
    base_url = st.text_input("Base URL (Optional)", value="https://ai-research-proxy.azurewebsites.net")
    model_name = st.text_input("Model Name", value="gpt-oss-120b")
    
    st.markdown("---")
    
    # Editable Prompts
    st.subheader("Edit System Prompts")
    with st.expander("System Prompt"):
        sys_prompt_editable = st.text_area("System Prompt", value=get_setting("system_prompt"), height=200)
        if st.button("Save System Prompt"):
            save_setting("system_prompt", sys_prompt_editable)
            st.success("Saved!")
            
    with st.expander("Requirements Text (RAG Context)"):
        req_text_editable = st.text_area("Requirements Text", value=get_setting("requirements_text"), height=200)
        if st.button("Save & Embed Requirements"):
            save_setting("requirements_text", req_text_editable)
            embed_requirements(req_text_editable)
            st.success("Saved and Embedded!")
            
    st.markdown("---")
    
    # Inputs
    st.subheader("Upload Course Syllabi")
    uploaded_files = st.file_uploader("Upload PDF/TXT (Batch allowed)", type=['pdf', 'txt'], accept_multiple_files=True)
    
    manual_input = st.text_area("Or Paste Syllabus Text")
    
    url_input = st.text_input("Or scrape URL(s) — separate multiple with spaces")
    
    if st.button("Analyze & Evaluate", type="primary"):
        if not api_key:
            st.error("Please provide an API Key first.")
        else:
            with st.spinner("Processing..."):
                llm = LLMManager(api_key=api_key, base_url=base_url, model=model_name)
                req_text = get_setting("requirements_text")
                sys_prompt = get_setting("system_prompt")
                
                # We will process each item and append to session state
                results = []
                
                def process_and_run(c_name_fallback, text_val, content_type, content_path=None):
                    # Embed course content first (uses content hash → stable, skips if already done)
                    cid = generate_course_id(text_val)
                    embed_course_content(cid, text_val)

                    # Requirements: always use raw stored text — ChromaDB chunks have overlaps
                    # that cause long criteria to appear twice and confuse the LLM.
                    req_context = req_text
                    # Course content: RAG for large docs, full text for small ones
                    RAG_THRESHOLD = 5000
                    if len(text_val) > RAG_THRESHOLD:
                        retrieved = query_course_content(cid, req_context, n_results=12)
                        course_context = "\n\n---\n\n".join(retrieved) if retrieved else text_val
                    else:
                        course_context = text_val

                    eval_data = llm.evaluate_course(sys_prompt, req_context, course_context)

                    if "error" in eval_data:
                        st.error(f"Error for {c_name_fallback}: {eval_data['error']}")
                        return None

                    c_name = eval_data.get("course_name", c_name_fallback)
                    instructor_email = eval_data.get("instructor_email", "")

                    eval_id = save_evaluation(
                        c_name,
                        eval_data.get("overall_evaluation", ""),
                        eval_data.get("exact_quotes", []),
                        eval_data.get("email_english", ""),
                        eval_data.get("email_dutch", ""),
                        content_type,
                        content_path or "",
                        instructor_email,
                        requirements_table=eval_data.get("requirements_compliance_table", []),
                        text_content=text_val,
                    )

                    return {
                        "id": eval_id,
                        "course_name": c_name,
                        "overall_evaluation": eval_data.get("overall_evaluation", ""),
                        "requirements_compliance_table": eval_data.get("requirements_compliance_table", []),
                        "exact_quotes": eval_data.get("exact_quotes", []),
                        "email_english": eval_data.get("email_english", ""),
                        "email_dutch": eval_data.get("email_dutch", ""),
                        "instructor_email": instructor_email,
                        "content_type": content_type,
                        "content_path": content_path,
                        "text_val": text_val
                    }

                # 1. Handle Uploaded Files — processed in upload order, one evaluation each
                for pf in uploaded_files:
                    suffix = os.path.splitext(pf.name)[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(pf.getbuffer())
                        tmp_path = tmp.name
                    txt = extract_text(tmp_path)
                    res = process_and_run(pf.name, txt, "file", tmp_path)
                    if res: results.append(res)

                # 2. Handle pasted text
                if manual_input.strip():
                    res = process_and_run("Pasted Text", manual_input, "text")
                    if res: results.append(res)

                # 3. Handle URLs — space-separated, each processed as a separate evaluation
                for u in url_input.split():
                    u = u.strip()
                    if not u:
                        continue
                    txt = extract_url_text(u)
                    if txt.startswith("Error"):
                        st.error(f"Error scraping {u}: {txt}")
                    else:
                        res = process_and_run(u, txt, "url", u)
                        if res: results.append(res)
                
                if results:
                    st.session_state.courses = results
                    st.session_state.current_idx = 0
                    st.rerun()

    st.markdown("---")
    st.subheader("Past Evaluations")
    past_evals = get_all_evaluations()
    if past_evals:
        options = [f"{e['course_name']}  [{e['timestamp'][:10]}]" for e in past_evals]
        sel_idx = st.selectbox("Select to load", range(len(options)),
                               format_func=lambda i: options[i], label_visibility="collapsed")
        btn_col, del_col = st.columns([3, 1])
        if btn_col.button("Load", use_container_width=True):
            ev = past_evals[sel_idx]
            loaded = {
                "id": ev["id"],
                "course_name": ev["course_name"],
                "overall_evaluation": ev.get("evaluation_summary", ""),
                "requirements_compliance_table": json.loads(ev.get("requirements_table_json") or "[]"),
                "exact_quotes": json.loads(ev.get("exact_quotes_json") or "[]"),
                "email_english": ev.get("email_english", ""),
                "email_dutch": ev.get("email_dutch", ""),
                "instructor_email": ev.get("instructor_email", ""),
                "content_type": ev.get("original_content_type", ""),
                "content_path": ev.get("original_content_path", ""),
                "text_val": ev.get("text_content") or "",
            }
            st.session_state.courses = [loaded]
            st.session_state.current_idx = 0
            st.rerun()
        if del_col.button("🗑", use_container_width=True, help="Delete this evaluation"):
            delete_evaluation(past_evals[sel_idx]["id"])
            st.rerun()

# --- MAIN AREA ---

if st.session_state.courses:
    num_courses = len(st.session_state.courses)
    
    col_nav, col_download = st.columns([8, 2])
    with col_nav:
        c1, c2, c3 = st.columns([1, 8, 1])
        if c1.button("◀ Prev") and st.session_state.current_idx > 0:
            st.session_state.current_idx -= 1
            st.rerun()
            
        c2.markdown(f"<h3 style='text-align: center;'>Course {st.session_state.current_idx + 1} of {num_courses}: {st.session_state.courses[st.session_state.current_idx]['course_name']}</h3>", unsafe_allow_html=True)
            
        if c3.button("Next ▶") and st.session_state.current_idx < num_courses - 1:
            st.session_state.current_idx += 1
            st.rerun()
            
    with col_download:
        # Download summary across all courses
        csv_data = []
        for crs in st.session_state.courses:
            csv_data.append({
                "Instructor Email": crs.get("instructor_email", ""),
                "Overall Evaluation": crs.get("overall_evaluation", ""),
                "Email To Send": crs.get("email_english", "")
            })
        df_export = pd.DataFrame(csv_data)
        st.download_button(
            label="Download Summary Table",
            data=df_export.to_csv(index=False).encode('utf-8'),
            file_name="course_evaluations_summary.csv",
            mime="text/csv"
        )
            
    st.markdown("---")
    
    current_course = st.session_state.courses[st.session_state.current_idx]
    
    st.subheader("Output & Evaluation")
    st.markdown(current_course["overall_evaluation"])

    st.markdown("---")
    table_data = current_course.get("requirements_compliance_table", [])
    c_text     = current_course.get("text_val", "")
    components.html(build_interactive_viewer(table_data, c_text), height=680, scrolling=False)

    st.markdown("---")
    st.markdown("### Instructor Emails")
    
    st.markdown("#### Select Criteria for Email")
    failed_criteria = [row for row in table_data if row.get("status") in ["Not Met", "Partial"]]
    selected_criteria = []
    
    if failed_criteria:
        for i, row in enumerate(failed_criteria):
            if st.checkbox(f"[{row['status']}] {row['criterion']}", value=True, key=f"crit_{st.session_state.current_idx}_{i}"):
                selected_criteria.append(row)
    else:
        st.write("All requirements are met.")

    if st.button("Regenerate Emails Based on Selection"):
        with st.spinner("Regenerating emails..."):
            llm = LLMManager(api_key=api_key, base_url=base_url, model=model_name)
            new_emails = llm.generate_emails(current_course["course_name"], "Course Coordinator", selected_criteria)
            
            if "error" in new_emails:
                st.error(new_emails["error"])
            else:
                st.session_state.courses[st.session_state.current_idx]["email_english"] = new_emails.get("email_english", "")
                st.session_state.courses[st.session_state.current_idx]["email_dutch"] = new_emails.get("email_dutch", "")
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("English Email", expanded=True):
        st.text_area("Email (English)", value=current_course["email_english"], height=220,
                     key=f"email_en_{st.session_state.current_idx}")

    with st.expander("Dutch Email"):
        st.text_area("Email (Dutch)", value=current_course["email_dutch"], height=220,
                     key=f"email_nl_{st.session_state.current_idx}")
else:
    st.info("Upload course syllabus file(s), paste text, or enter a URL — then press 'Analyze & Evaluate'.")
