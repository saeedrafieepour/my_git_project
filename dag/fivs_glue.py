import streamlit as st
import json
import os
from datetime import datetime
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from typing import Dict, List
from PIL import Image, ImageFilter
import numpy as np
import io
import csv

# Set OpenAI API key from Streamlit secrets
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Page settings
st.set_page_config(
    page_title="Health Literacy Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏥 Health Literacy Assistant")
st.markdown("""
**Privacy-First Health Learning Tool**  
Upload medical documents, images, and vitals CSV to explore health literacy concepts and non-diagnostic insights.
""")

# Initialize session state
if "uploaded_documents" not in st.session_state:
    st.session_state.uploaded_documents = {}
if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = {}
if "vitals_history" not in st.session_state:
    st.session_state.vitals_history = []
if "csv_vitals_history" not in st.session_state:
    st.session_state.csv_vitals_history = []
if "uploaded_images" not in st.session_state:
    st.session_state.uploaded_images = []
if "image_analysis" not in st.session_state:
    st.session_state.image_analysis = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "health_goals" not in st.session_state:
    st.session_state.health_goals = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "last_processed_docs" not in st.session_state:
    st.session_state.last_processed_docs = set()

# Safety guardrails
HEALTH_GUARDRAILS = """
You are a health education assistant, NOT a medical professional.
IMPORTANT GUARDRAILS:
- Provide EDUCATIONAL explanations only—help users understand health concepts
- NEVER diagnose conditions or suggest treatment
- ALWAYS encourage consultation with healthcare professionals
"""

# Utility functions

def analyze_medical_image(image: Image.Image, exam_type: str) -> Dict[str, str]:
    gray = image.convert("L")
    arr = np.array(gray)
    mean = float(np.mean(arr))
    std = float(np.std(arr))

    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_arr = np.array(edges)
    edge_strength = float(np.mean(edge_arr))

    # Pattern heuristic tags (educational only, not diagnosis)
    sign_flag = "No strong pattern detected."
    if exam_type == "chest":
        if mean < 80 and std > 55 and edge_strength > 35:
            sign_flag = "Pattern suggests this image could have pneumonia-like opacity characteristics (educational only)."
        elif mean > 170 and std < 30:
            sign_flag = "Pattern suggests overexposed image; may hide detail in lung fields."
        else:
            sign_flag = "Pattern within a moderate histogram range for chest scans."

    elif exam_type == "brain":
        if std > 70 and edge_strength > 40:
            sign_flag = "Pattern suggests high anatomical edge contrast typical of complex structures, potentially requiring expert imaging review."
        else:
            sign_flag = "Pattern appears moderate for brain scan contrast/texture."

    analysis = {
        "exam_type": exam_type,
        "mean_intensity": f"{mean:.1f}",
        "contrast_score": f"{std:.1f}",
        "edge_score": f"{edge_strength:.1f}",
        "pattern_flag": sign_flag
    }

    # Educational pattern suggestions (non-diagnostic)
    if exam_type == "chest":
        if mean < 80:
            text = (
                "The image is relatively dark, which can happen in underexposed X-rays. "
                "In educational examples of viral pneumonia (like COVID-19), areas of opacity may appear as cloudy/whitish patches. "
                "A radiologist would interpret this in context with symptoms and clinical tests."
            )
        elif mean > 170:
            text = (
                "The image is quite bright, indicating possible overexposure. "
                "Proper X-ray technique and calibrated equipment are crucial for medical image interpretation."
            )
        else:
            text = (
                "Image brightness is within the normal range for basic chest scans. "
                "This is a neutral educational estimate—not a diagnosis."
            )
        text += "\n\nAlways consult qualified imaging professionals for medical interpretation."

    elif exam_type == "brain":
        if std > 65:
            text = (
                "High contrast and edge structure can be seen in scans with complex anatomy or brain lesions. "
                "In training data, mass-like regions may generate strong edges, but this is only a rough indicator. "
                "Actual clinical evaluation requires radiologist expertise."
            )
        else:
            text = (
                "Brain scan contrast appears moderate. This educational screen does not replace medical interpretation."
            )
        text += "\n\nAlways seek professional medical imaging review for any concerns."

    else:
        text = "General imaging analysis. No medical judgment provided. Always consult a healthcare provider."

    analysis["educational_insight"] = text
    return analysis


def parse_vitals_csv(csv_bytes: bytes):
    text = csv_bytes.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    rows = []
    for row in reader:
        try:
            parsed = {
                "timestamp": row.get("timestamp") or row.get("time") or datetime.now().isoformat(),
                "pulse": int(row.get("pulse", row.get("heart_rate", 0) or 0)),
                "bp": row.get("bp", f"{row.get('systolic','')}/{row.get('diastolic','')}").strip(),
                "spo2": float(row.get("spo2", row.get("o2", 0) or 0)),
                "temperature": float(row.get("temp", row.get("temperature", 0) or 0))
            }
            rows.append(parsed)
        except Exception:
            continue
    return rows


def generate_medical_history_insights(extracted_text: Dict[str, str]) -> Dict[str, str]:
    text_blob = "\n\n".join(extracted_text.values()).lower()
    history_flags = []

    risk_terms = {
        "hypertension": "Hypertension history mentioned",
        "high blood pressure": "High blood pressure history mentioned",
        "diabetes": "Diabetes history mentioned",
        "asthma": "Asthma history mentioned",
        "heart disease": "Heart disease history mentioned",
        "covid": "COVID-19 history or exposure mentioned",
        "pneumonia": "Pneumonia history mentioned"
    }

    for term, label in risk_terms.items():
        if term in text_blob:
            history_flags.append(label)

    if not history_flags:
        history_flags.append("No explicit chronic condition keywords found in uploaded medical history text.")

    return {
        "history_flags": history_flags,
        "raw_text_snippet": text_blob[:800] + "..." if len(text_blob) > 800 else text_blob
    }


def generate_insight_recommendations(latest_vitals: dict, history_insights: dict) -> List[str]:
    recs = []

    if not latest_vitals:
        recs.append("No vital signs available yet. Log vitals manually or via CSV to get personalized insights.")
        return recs

    try:
        sys = float(latest_vitals.get('bp', '0/0').split('/')[0]) if '/' in latest_vitals.get('bp','') else 0
        dia = float(latest_vitals.get('bp', '0/0').split('/')[1]) if '/' in latest_vitals.get('bp','') else 0
        pulse = float(latest_vitals.get('pulse', 0))
        spo2 = float(latest_vitals.get('spo2', 0))
        temp = float(latest_vitals.get('temperature', 0))
    except Exception:
        recs.append("Unable to parse vitals cleanly for recommendation. Check data format.")
        return recs

    # Blood pressure insights
    if sys >= 140 or dia >= 90:
        recs.append("Educational: BP readings are in the hypertensive range; consider consulting a healthcare provider for long-term management.")
    elif sys >= 130 or dia >= 80:
        recs.append("Educational: BP readings are elevated. Lifestyle changes like reducing sodium and exercising can be beneficial.")
    else:
        recs.append("Educational: BP readings are in normal range. Maintain your current cardiovascular wellness habits.")

    # Heart rate insights
    if pulse < 60:
        recs.append("Educational: Resting pulse is low. If you're active or athletic, this can be normal; otherwise discuss with a provider.")
    elif pulse > 100:
        recs.append("Educational: Resting pulse is elevated. Monitor for symptoms and discuss with a clinician.")
    else:
        recs.append("Educational: Resting pulse is in typical adult range.")

    # SpO2/temperature quick check
    if spo2 < 95:
        recs.append("Educational: Blood oxygen is low (<95%). In respiratory conditions, this can be a concern; seek medical guidance.")
    if temp >= 100.4:
        recs.append("Educational: Fever reported. If you have history of infection (e.g., COVID or pneumonia), consult a health provider.")

    # history-aware suggestions
    for flag in history_insights.get('history_flags', []):
        if 'hypertension' in flag.lower() and sys >= 130:
            recs.append("History-aware: Because hypertension is mentioned, tracking blood pressure regularly is especially important.")
        if 'diabetes' in flag.lower():
            recs.append("History-aware: If diabetes is discussed in your records, keep watch on blood pressure and cardiovascular warning signs.")
        if 'covid' in flag.lower() and spo2 < 96:
            recs.append("History-aware: Past COVID mention plus low SpO2 may suggest additional respiratory follow-up is educational to pursue.")

    if not history_insights.get('history_flags'):
        recs.append("No medical history flags found; continue building patient history in uploaded documents.")

    return recs


# Sidebar upload
with st.sidebar:
    st.header("📥 Upload Medical Data")
    uploaded_files = st.file_uploader(
        "Upload PDF, image (brain/chest), or CSV vitals",
        type=["pdf", "png", "jpg", "jpeg", "csv"],
        accept_multiple_files=True
    )

    if uploaded_files:
        with st.spinner("Processing uploads..."):
            for f in uploaded_files:
                name = f.name
                if name in st.session_state.last_processed_docs:
                    continue

                if name.lower().endswith(".pdf"):
                    if name not in st.session_state.uploaded_documents:
                        reader = PdfReader(f)
                        text = ""
                        for pno, page in enumerate(reader.pages):
                            part = page.extract_text() or ""
                            text += f"\n--- Page {pno + 1} ---\n" + part
                        st.session_state.uploaded_documents[name] = {
                            "uploaded_at": datetime.now().isoformat(),
                            "num_pages": len(reader.pages),
                            "raw_text": text
                        }
                        st.session_state.extracted_text[name] = text

                elif name.lower().endswith((".png", ".jpg", ".jpeg")):
                    bio = io.BytesIO(f.read())
                    img = Image.open(bio).convert("RGB")
                    exam_type = st.selectbox(f"Identify image type for {name}", ["chest", "brain"], key=f"type_{name}")
                    analysis = analyze_medical_image(img, exam_type)
                    st.session_state.uploaded_images.append({"name": name, "image": img, "exam_type": exam_type})
                    st.session_state.image_analysis.append({"name": name, **analysis})

                elif name.lower().endswith(".csv"):
                    vlist = parse_vitals_csv(f.read())
                    if vlist:
                        st.session_state.csv_vitals_history.extend(vlist)

                st.session_state.last_processed_docs.add(name)

        st.success("Upload complete. Explore tabs for details.")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📤 Review Text", "💊 Vitals Tracker", "🤖 Chat Assistant", "📋 Health Insights", "ℹ️ About"])

with tab1:
    st.header("Review & Correct Extracted Text")
    if st.session_state.extracted_text:
        selected_doc = st.selectbox("Select document:", list(st.session_state.extracted_text.keys()))
        text_val = st.text_area("Extracted text", value=st.session_state.extracted_text[selected_doc], height=300)
        if text_val != st.session_state.extracted_text[selected_doc]:
            st.session_state.extracted_text[selected_doc] = text_val
            st.success("Text updated")
    else:
        st.info("Upload at least one PDF to review text")

with tab2:
    st.header("Vitals Tracker")
    st.markdown("Use manual entry or upload CSV for vitals history")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        pulse = st.number_input("Pulse (bpm)", 30, 200, 72)
    with col2:
        systolic = st.number_input("Systolic BP", 60, 250, 120)
    with col3:
        diastolic = st.number_input("Diastolic BP", 40, 150, 80)
    with col4:
        spo2 = st.number_input("SpO₂ (%)", 50.0, 100.0, 98.0, step=0.1)
    temp = st.number_input("Temperature (°F)", 92.0, 108.0, 98.6, step=0.1)
    if st.button("Log vitals"):
        st.session_state.vitals_history.append({
            "timestamp": datetime.now().isoformat(),
            "pulse": pulse,
            "bp": f"{systolic}/{diastolic}",
            "spo2": spo2,
            "temperature": temp
        })
        st.success("Logged")

    if st.session_state.csv_vitals_history:
        st.markdown("### Vitals from CSV")
        for r in st.session_state.csv_vitals_history[-10:]:
            st.write(f"{r['timestamp']} | Pulse {r['pulse']} | BP {r['bp']} | Temp {r['temperature']}°F | O₂ {r['spo2']}%")

    if st.session_state.vitals_history:
        st.markdown("### Personal Vitals History")
        for r in st.session_state.vitals_history[-10:]:
            st.write(f"{r['timestamp']} | Pulse {r['pulse']} | BP {r['bp']} | Temp {r['temperature']}°F | O₂ {r['spo2']}%")

with tab3:
    st.header("Health Education Chat")
    question = st.text_area("Ask a health question", key="question", height=120)
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("OpenAI key missing. Add to .streamlit/secrets.toml to enable chat.")
    else:
        if st.button("Ask"):
            if not question.strip():
                st.warning("Please type a question before clicking Ask.")
            else:
                with st.spinner("Generating educational response..."):
                    try:
                        # Prepare context from uploaded documents, vitals, and image analysis
                        context_text = "\n\n".join(
                            [f"Document: {name}\n{text}" for name, text in st.session_state.extracted_text.items()]
                        ) or "No document context uploaded."

                        latest_vitals = st.session_state.vitals_history[-1] if st.session_state.vitals_history else None
                        if not latest_vitals and st.session_state.csv_vitals_history:
                            latest_vitals = st.session_state.csv_vitals_history[-1]

                        vitals_context = (
                            f"Pulse: {latest_vitals['pulse']} bpm, BP: {latest_vitals['bp']}, "
                            f"Temp: {latest_vitals['temperature']}°F, SpO2: {latest_vitals['spo2']}%"
                        ) if latest_vitals else "No vitals logged yet."

                        # Include image analysis context
                        image_context = ""
                        if st.session_state.image_analysis:
                            image_context = "\n\nImage Analysis Results:\n" + "\n".join([
                                f"- {item['name']} ({item['exam_type']}): Mean intensity {item['mean_intensity']}, "
                                f"Contrast {item['contrast_score']}, Edge score {item['edge_score']}. "
                                f"Educational insight: {item['educational_insight']}"
                                for item in st.session_state.image_analysis
                            ])
                        else:
                            image_context = "\n\nNo medical images have been analyzed yet."

                        prompt = f"""{HEALTH_GUARDRAILS}

Context from documents:
{context_text}

Current vitals:
{vitals_context}

{image_context}

User question:
{question}

Please provide an educational, non-diagnostic response with clear guidance about seeking professional healthcare advice. Reference the image analysis results when relevant to the question.
"""

                        chat_response = None
                        try:
                            # new OpenAI 1.x client
                            from openai import OpenAI as OpenAIClient
                            client = OpenAIClient()
                            response = client.chat.completions.create(
                                model="gpt-4o",
                                messages=[
                                    {"role": "system", "content": "You are an educational health assistant. " + HEALTH_GUARDRAILS + " When discussing images, reference the provided analysis results and emphasize that you cannot diagnose conditions. Focus on educational patterns and always recommend professional medical interpretation."},
                                    {"role": "user", "content": prompt}
                                ],
                                temperature=0.2,
                                max_tokens=650
                            )

                            if hasattr(response, "choices") and response.choices:
                                choice = response.choices[0]
                                # choice.message may be an object with .content
                                if hasattr(choice, "message"):
                                    msg = choice.message
                                    chat_response = getattr(msg, "content", None) or (msg.get("content") if hasattr(msg, "get") else None)
                                elif isinstance(choice, dict) and "message" in choice:
                                    chat_response = choice["message"].get("content") if isinstance(choice["message"], dict) else getattr(choice["message"], "content", None)

                        except Exception as openai_exc:
                            # If OpenAI API call fails, try LangChain fallback
                            try:
                                llm = ChatOpenAI(model_name="gpt-4o", temperature=0.2)
                                if hasattr(llm, "predict"):
                                    chat_response = llm.predict(prompt)
                                elif hasattr(llm, "generate"):
                                    out = llm.generate([prompt])
                                    if isinstance(out, str):
                                        chat_response = out
                                    elif hasattr(out, "generations") and out.generations:
                                        chat_response = out.generations[0][0].text
                                    else:
                                        chat_response = str(out)
                                elif callable(llm):
                                    out = llm(prompt)
                                    if isinstance(out, str):
                                        chat_response = out
                                    elif hasattr(out, "content"):
                                        chat_response = out.content
                                    elif hasattr(out, "generations") and out.generations:
                                        chat_response = out.generations[0][0].text
                                    else:
                                        chat_response = str(out)
                            except Exception as chain_exc:
                                raise Exception(f"OpenAI 1.x call failed ({openai_exc}); LangChain fallback failed ({chain_exc})")

                        if chat_response is None:
                            raise Exception("Could not generate a response with any configured LLM client.")

                        st.markdown("### 📚 Educational Response")
                        st.info(chat_response)

                    except Exception as final_exc:
                        st.error(f"Chat failed: {final_exc}")
                        st.warning("Make sure your OpenAI package and API key are configured, and that you have internet access.")

                        st.session_state.chat_history.append({
                            "question": question,
                            "response": chat_response,
                            "timestamp": datetime.now().isoformat()
                        })
                    except Exception as e:
                        st.error(f"Chat failed: {e}")
                        st.warning("Make sure your OpenAI API key is set and valid.")

with tab4:
    st.header("Health Insights & Educational Notes")

    if st.session_state.image_analysis:
        st.subheader("🖼️ Image Analysis (Educational)")
        for item in st.session_state.image_analysis:
            st.markdown(f"**{item['name']}** ({item['exam_type']})")

            # Display uploaded image visual
            stored_img = next((img['image'] for img in st.session_state.uploaded_images if img['name'] == item['name']), None)
            if stored_img is not None:
                st.image(stored_img, caption=f"{item['exam_type'].title()} scan: {item['name']}", use_column_width=True)

            st.metric("Mean Intensity", item['mean_intensity'])
            st.metric("Contrast Score", item['contrast_score'])
            st.metric("Edge Score", item['edge_score'])

            # Add illustrative text markers for educational scenario
            marker = "Medium" if item['exam_type'] == "chest" else "Moderate"
            try:
                mi = float(item['mean_intensity'])
                if item['exam_type'] == "chest" and mi < 75:
                    marker = "Low brightness; may indicate underexposure or denser tissue areas"
                elif item['exam_type'] == "chest" and mi > 180:
                    marker = "High brightness; may indicate overexposure"
                elif item['exam_type'] == "brain" and float(item['contrast_score']) > 65:
                    marker = "High contrast edges; some anatomical complexity visible"
            except Exception:
                pass

            st.markdown(f"**Educational Note:** {marker}")
            st.markdown(f"**Pattern Indicator:** {item.get('pattern_flag', 'No pattern flag available')}  ")
            st.info(item['educational_insight'])

    plus_history = st.session_state.vitals_history + st.session_state.csv_vitals_history
    # Deduplicate by timestamp+pulse+bp+spo2+temperature
    seen = set()
    unique_history = []
    for entry in plus_history:
        key = (
            entry.get('timestamp'),
            entry.get('pulse'),
            entry.get('bp'),
            entry.get('spo2'),
            entry.get('temperature')
        )
        if key not in seen:
            seen.add(key)
            unique_history.append(entry)

    if unique_history:
        st.subheader("📊 Combined Vitals Insights")
        bp_numbers = [float(x['bp'].split('/')[0]) for x in unique_history if "/" in x['bp']]
        avg_bp = sum(bp_numbers) / len(bp_numbers) if bp_numbers else 0
        st.metric("Average Systolic (educational)", f"{avg_bp:.1f} mmHg")
        st.info("This is for learning and trend awareness, not a medical diagnosis.")
        st.markdown("### Recent Combined Vitals")
        for r in unique_history[-10:]:
            st.write(f"{r['timestamp']} | Pulse {r['pulse']} | BP {r['bp']} | Temp {r['temperature']}°F | O₂ {r['spo2']}%")
    else:
        st.info("No vitals data available yet.")

    # Medical history + vitals insights
    history_insights = generate_medical_history_insights(st.session_state.extracted_text)
    st.subheader("🩺 Medical History Insights")
    for flag in history_insights.get('history_flags', []):
        st.markdown(f"- {flag}")

    latest_vitals = unique_history[-1] if unique_history else None
    recs = generate_insight_recommendations(latest_vitals, history_insights)

    st.subheader("💡 Combined Recommendation Summary")
    for rec in recs:
        st.info(rec)

    st.subheader("🎯 Health Learning Goals")
    new_goal = st.text_input("Add goal", "")
    if st.button("Add goal") and new_goal:
        st.session_state.health_goals.append({"goal": new_goal, "completed": False})
    for i, goal in enumerate(st.session_state.health_goals):
        col_a, col_b, col_c = st.columns([0.1, 0.8, 0.1])
        done = col_a.checkbox("", key=f"geo_{i}", value=goal.get('completed', False))
        if done != goal.get('completed', False):
            st.session_state.health_goals[i]['completed'] = done
        col_b.markdown(f"{ '✅' if done else '⏳' } {goal['goal']}")
        if col_c.button("Delete", key=f"del_{i}"):
            st.session_state.health_goals.pop(i)
            st.experimental_rerun()

    st.markdown("---")
    st.warning("⚠️ This tool provides educational insights only. Not a medical diagnosis.")

with tab5:
    st.header("About This Enhanced Tool")
    st.markdown("'''")
    st.write("Preserves the original app functionality and adds image + CSV vitals data paths with strict no-diagnosis guardrails.")
    st.markdown("'''")
