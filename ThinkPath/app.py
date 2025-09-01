
import json
import os
import re
from typing import List, Dict, Any
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from llm_providers import LLMClient

# =============================
# Bilingual strings (FA/EN)
# =============================
T = {
    "en": {
        "title": "ThinkPath (Python) — Guided LLM Chat",
        "subtitle": "Strategic Thinking Assistant · Bilingual (FA/EN) · OpenAI / Ollama",
        "sidebar_title": "Settings",
        "provider": "Provider",
        "provider_openai": "OpenAI (Chat Completions)",
        "provider_ollama": "Ollama (Local)",
        "model": "Model",
        "api_key": "OpenAI API Key",
        "base_url": "Ollama Base URL",
        "language": "UI Language",
        "question": "Your Question",
        "generate_paths": "Generate Thinking Paths",
        "paths": "Thinking Paths",
        "run_to_step": "Run up to Step {n}",
        "clear_history": "Clear History",
        "history": "Conversation",
        "error_parse": "Couldn't parse paths as JSON. Try again or adjust your model/temperature.",
        "select_language": "Answer Language",
        "answer_lang_auto": "Match UI language",
        "answer_lang_fa": "Persian (Farsi)",
        "answer_lang_en": "English",
        "about": "About",
        "footer": "Built with Streamlit · This app runs either on local Ollama or OpenAI API.",
        "no_paths_yet": "No paths yet. Ask a question and click “Generate Thinking Paths”.",
        "executed": "Executed: {title} — Steps 1..{n}",
        "temperature": "Temperature",
        "system_note": "Use the selected language for answers. Keep structure clean, with short paragraphs and bullet points."
    },
    "fa": {
        "title": "ThinkPath (نسخه پایتون) — گفت‌وگوی هدایت‌شده",
        "subtitle": "دستیار تفکر راهبردی · دو‌زبانه (فارسی/انگلیسی) · OpenAI / Ollama",
        "sidebar_title": "تنظیمات",
        "provider": "ارائه‌دهنده",
        "provider_openai": "OpenAI (Chat Completions)",
        "provider_ollama": "Ollama (لوکال)",
        "model": "نام مدل",
        "api_key": "کلید OpenAI",
        "base_url": "آدرس Ollama",
        "language": "زبان رابط کاربری",
        "question": "سؤال شما",
        "generate_paths": "تولید مسیرهای تفکر",
        "paths": "مسیرهای تفکر",
        "run_to_step": "اجرا تا گام {n}",
        "clear_history": "پاک‌کردن گفت‌وگو",
        "history": "گفت‌وگو",
        "error_parse": "تحلیل مسیرها به JSON ناموفق بود. دوباره تلاش کنید یا مدل/دما را تغییر دهید.",
        "select_language": "زبان پاسخ",
        "answer_lang_auto": "مطابق زبان رابط",
        "answer_lang_fa": "فارسی",
        "answer_lang_en": "انگلیسی",
        "about": "درباره",
        "footer": "ساخته‌شده با Streamlit · این برنامه با Ollama محلی یا OpenAI کار می‌کند.",
        "no_paths_yet": "هنوز مسیری تولید نشده است. سؤال خود را وارد کرده و «تولید مسیرهای تفکر» را بزنید.",
        "executed": "اجرا شد: {title} — گام‌های ۱ تا {n}",
        "temperature": "Temperature (دمای تولید)",
        "system_note": "پاسخ را با زبان انتخاب‌شده ارائه بده. ساختار را تمیز نگه‌دار با پاراگراف کوتاه و بولت."
    },
}

def i18n(lang: str, key: str, **fmt):
    txt = T.get(lang, T["en"]).get(key, key)
    if fmt:
        txt = txt.format(**fmt)
    return txt

# =============================
# Streamlit page config
# =============================
st.set_page_config(page_title="ThinkPath (Python)", page_icon="🧠", layout="wide")

# Session State
if "paths" not in st.session_state:
    st.session_state.paths = []
if "history" not in st.session_state:
    st.session_state.history = []
if "last_question" not in st.session_state:
    st.session_state.last_question = ""

# Sidebar
with st.sidebar:
    ui_lang = st.selectbox("Language / زبان رابط", options=["fa", "en"], index=0, format_func=lambda x: "فارسی" if x=="fa" else "English")
    st.markdown(f"### {i18n(ui_lang,'sidebar_title')}")
    provider = st.selectbox(i18n(ui_lang, "provider"), options=["openai", "ollama"], index=1, format_func=lambda x: i18n(ui_lang, "provider_openai") if x=="openai" else i18n(ui_lang, "provider_ollama"))
    default_model = "gpt-4o-mini" if provider == "openai" else "llama3.1:8b"
    model = st.text_input(i18n(ui_lang, "model"), value=default_model)
    temperature = st.slider(i18n(ui_lang, "temperature"), min_value=0.0, max_value=1.5, value=0.3, step=0.1)
    if provider == "openai":
        api_key = st.text_input(i18n(ui_lang, "api_key"), type="password", value=os.getenv("OPENAI_API_KEY", ""))
        base_url = None
    else:
        api_key = None
        base_url = st.text_input(i18n(ui_lang, "base_url"), value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    ans_lang_choice = st.selectbox(i18n(ui_lang,"select_language"), options=["auto","fa","en"], index=0, format_func=lambda x: i18n(ui_lang, f"answer_lang_{x}") if x!="auto" else i18n(ui_lang, "answer_lang_auto"))
    st.divider()
    st.markdown(f"**{i18n(ui_lang,'about')}**")
    st.caption(i18n(ui_lang, "footer"))

st.title(i18n(ui_lang, "title"))
st.caption(i18n(ui_lang, "subtitle"))

question = st.text_area(i18n(ui_lang, "question"), height=120, value=st.session_state.last_question)

colA, colB = st.columns([1,1])
with colA:
    go = st.button(i18n(ui_lang, "generate_paths"))
with colB:
    if st.button(i18n(ui_lang, "clear_history")):
        st.session_state.history = []
        st.session_state.paths = []
        st.session_state.last_question = ""

def pick_answer_language():
    if ans_lang_choice == "auto":
        return "fa" if ui_lang=="fa" else "en"
    return ans_lang_choice

def extract_json_block(text: str) -> str:
    m = re.search(r'(\{.*\})', text, flags=re.DOTALL)
    return m.group(1) if m else text

def ensure_paths_structure(raw: str) -> List[Dict[str, Any]]:
    try:
        data = json.loads(raw)
    except Exception:
        try:
            data = json.loads(extract_json_block(raw))
        except Exception:
            return []
    if isinstance(data, dict) and "paths" in data:
        paths = data["paths"]
    elif isinstance(data, list):
        paths = data
    else:
        return []
    norm = []
    for p in paths:
        title = p.get("title") or p.get("name") or "Path"
        steps = p.get("steps") or []
        if not isinstance(steps, list):
            continue
        steps = [str(s) for s in steps][:3]
        norm.append({"title": title, "steps": steps})
    return norm[:4]

def build_client() -> LLMClient:
    return LLMClient(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature
    )

if go and question.strip():
    st.session_state.last_question = question.strip()
    client = build_client()
    answer_lang = pick_answer_language()

    system_prompt = (
        "You are a Strategic Thinking Assistant that produces GUIDED thinking paths. "
        "Return STRICT JSON with 4 objects under key 'paths'. Each path has a 'title' and exactly 3 concise 'steps'. "
        "Write the path titles and steps in the target language indicated."
    )
    if answer_lang == "fa":
        system_prompt += " زبان خروجی: فارسی."
    else:
        system_prompt += " Output language: English."

    user_prompt = (
        f"Question: {st.session_state.last_question}\n\n"
        "Generate four distinct thinking approaches (e.g., Analytical, Creative, Practical, Comprehensive). "
        "Each approach must include exactly three actionable steps tailored to the question.\n\n"
        "Return JSON ONLY in the following shape:\n"
        '{\n  "paths": [\n    {"title": "...", "steps": ["...", "...", "..."]},\n    {"title": "...", "steps": ["...", "...", "..."]},\n    {"title": "...", "steps": ["...", "...", "..."]},\n    {"title": "...", "steps": ["...", "...", "..."]}\n  ]\n}\n'
        "Do NOT add any extra commentary."
    )

    with st.spinner("Generating paths…"):
        resp = client.chat([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])
    paths = ensure_paths_structure(resp)
    st.session_state.paths = paths
    if not paths:
        st.error(i18n(ui_lang, "error_parse"))
    else:
        st.success(i18n(ui_lang, "paths"))

st.subheader(i18n(ui_lang, "paths"))
if not st.session_state.paths:
    st.info(i18n(ui_lang, "no_paths_yet"))
else:
    cols = st.columns(2)
    for i, path in enumerate(st.session_state.paths):
        col = cols[i % 2]
        with col:
            with st.expander(f"🧭 {path['title']}", expanded=True):
                for step_idx, step in enumerate(path["steps"], start=1):
                    st.markdown(f"**{step_idx}. {step}**")
                st.markdown("---")
                c1, c2, c3 = st.columns(3)
                for step_idx, c in enumerate([c1, c2, c3], start=1):
                    if c.button(i18n(ui_lang, "run_to_step", n=step_idx), key=f"run_{i}_{step_idx}"):
                        client = build_client()
                        answer_lang = pick_answer_language()
                        exec_system = (
                            "You are a stepwise reasoning assistant. "
                            "Given a thinking path with 3 steps, produce a structured answer that uses ONLY the first N steps (cumulative). "
                            "Use clear headings, bold key terms, and short bullet points. "
                            "End with a brief 'Next Steps' section.\n"
                        )
                        if answer_lang == "fa":
                            exec_system += " زبان پاسخ: فارسی."
                            title_line = f"**{path['title']}** — اجرای گام‌های ۱ تا {step_idx}"
                        else:
                            exec_system += " Answer language: English."
                            title_line = f"**{path['title']}** — Executed steps 1..{step_idx}"

                        exec_user = {
                            "question": st.session_state.last_question,
                            "path": path,
                            "execute_upto": step_idx,
                        }
                        with st.spinner("Thinking…"):
                            ans = client.chat([
                                {"role": "system", "content": exec_system},
                                {"role": "user", "content": json.dumps(exec_user, ensure_ascii=False)},
                            ])
                        st.session_state.history.append({"role": "assistant", "content": f"{title_line}\n\n{ans}"})
                        st.toast(i18n(ui_lang, "executed", title=path["title"], n=step_idx))

st.subheader(i18n(ui_lang, "history"))
if st.session_state.history:
    for msg in st.session_state.history:
        st.markdown(msg["content"])
else:
    st.caption("—")
