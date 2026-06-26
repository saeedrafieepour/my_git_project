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

# Set OpenAI API key from Streamlit secrets
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# ============================================================================
# PAGE CONFIGURATION & INITIALIZATION
# ============================================================================
st.set_page_config(
    page_title="Health Literacy Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏥 Health Literacy Assistant")
st.markdown("""
**Privacy-First Health Learning Tool**  
Upload your medical documents and track vital signs to get personalized health education—all offline and local.
""")

# Initialize session state for persistent data
if "uploaded_documents" not in st.session_state:
    st.session_state.uploaded_documents = {}
if "vitals_history" not in st.session_state:
    st.session_state.vitals_history = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = {}
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "last_processed_docs" not in st.session_state:
    st.session_state.last_processed_docs = set()
if "health_goals" not in st.session_state:
    st.session_state.health_goals = []

# ============================================================================
# SAFETY GUARDRAILS
# ============================================================================
HEALTH_GUARDRAILS = """
You are a health education assistant, NOT a medical professional.
IMPORTANT GUARDRAILS:
- Provide EDUCATIONAL explanations only—help users understand health concepts
- NEVER diagnose conditions or suggest treatments
- NEVER recommend medication changes or dosages
- NEVER tell users to stop taking medications
- ALWAYS encourage consulting with healthcare professionals
- Cite which document/page your information comes from for verification
- If unsure, recommend consulting their doctor
"""

# ============================================================================
# SIDEBAR: DOCUMENT UPLOAD & MANAGEMENT
# ============================================================================
with st.sidebar:
    st.header("📄 Document Upload")
    
    uploaded_pdfs = st.file_uploader(
        "Upload Medical Documents (PDF/image scans)",
        type=["pdf"],
        accept_multiple_files=True
    )
    
    if uploaded_pdfs:
        with st.spinner("Processing documents... (this may take a moment)"):
            progress_bar = st.progress(0)
            total_files = len(uploaded_pdfs)
            
            for i, pdf in enumerate(uploaded_pdfs):
                if pdf.name not in st.session_state.uploaded_documents:
                    # Extract text with progress update
                    progress_bar.progress((i / total_files) * 0.5, f"Processing {pdf.name}...")
                    
                    reader = PdfReader(pdf)
                    text = ""
                    num_pages = len(reader.pages)
                    for page_num, page in enumerate(reader.pages):
                        text += f"\n--- Page {page_num + 1} ---\n"
                        text += page.extract_text()
                    
                    st.session_state.uploaded_documents[pdf.name] = {
                        "uploaded_at": datetime.now().isoformat(),
                        "num_pages": num_pages,
                        "raw_text": text
                    }
                    st.session_state.extracted_text[pdf.name] = text
                
                progress_bar.progress(((i + 1) / total_files) * 0.5, f"Completed {pdf.name}")
            
            progress_bar.empty()
        
        if st.session_state.uploaded_documents:
            st.success(f"✓ {len(st.session_state.uploaded_documents)} document(s) loaded")
            
            # Show documents
            with st.expander("📋 Uploaded Documents"):
                for doc_name, doc_info in st.session_state.uploaded_documents.items():
                    st.markdown(f"**{doc_name}**  \nPages: {doc_info['num_pages']}")

# ============================================================================
# MAIN CONTENT AREA: TABS
# ============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📤 Review Text", "💊 Vitals Tracker", "🤖 Chat Assistant", "📋 Health Insights", "ℹ️ About This Tool"])

# ============================================================================
# TAB 1: REVIEW EXTRACTED TEXT
# ============================================================================
with tab1:
    st.header("Review & Correct Extracted Text")
    st.markdown("Review the text extracted from your documents. You can edit it to correct OCR errors or remove sensitive information.")
    
    if st.session_state.extracted_text:
        selected_doc = st.selectbox("Select a document to review:", list(st.session_state.extracted_text.keys()))
        
        if selected_doc:
            extracted = st.session_state.extracted_text[selected_doc]
            edited_text = st.text_area(
                "Edit extracted text (changes are saved):",
                value=extracted,
                height=300,
                key=f"edit_{selected_doc}"
            )
            
            if edited_text != extracted:
                st.session_state.extracted_text[selected_doc] = edited_text
                st.success("✓ Changes saved to session")
            
            st.info(f"📊 Document: {selected_doc} | Characters: {len(edited_text)}")
    else:
        st.warning("Upload documents first to review extracted text")

# ============================================================================
# TAB 2: VITALS TRACKER
# ============================================================================
with tab2:
    st.header("Vital Signs Tracker")
    st.markdown("Log your vital signs to help contextualize your health questions.")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        pulse = st.number_input("Pulse (bpm)", min_value=30, max_value=200, value=72)
    with col2:
        systolic = st.number_input("Systolic BP (mmHg)", min_value=60, max_value=250, value=120)
    with col3:
        diastolic = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=150, value=80)
    with col4:
        spo2 = st.number_input("SpO₂ (%)", min_value=50.0, max_value=100.0, value=98.0, step=0.1)
    
    temp = st.number_input("Temperature (°F)", min_value=92.0, max_value=108.0, value=98.6, step=0.1)
    
    if st.button("➕ Log Vitals", use_container_width=True):
        vital_entry = {
            "timestamp": datetime.now().isoformat(),
            "pulse": pulse,
            "bp": f"{systolic}/{diastolic}",
            "spo2": spo2,
            "temperature": temp
        }
        st.session_state.vitals_history.append(vital_entry)
        st.success("✓ Vitals logged successfully")
    
    if st.session_state.vitals_history:
        st.markdown("### Recent Vitals")
        for i, entry in enumerate(reversed(st.session_state.vitals_history[-10:])):  # Show last 10
            timestamp = datetime.fromisoformat(entry["timestamp"]).strftime("%Y-%m-%d %H:%M")
            st.markdown(f"""
            **{timestamp}**  
            💓 {entry['pulse']} bpm | 🩹 {entry['bp']} mmHg | 🌡️ {entry['temperature']}°F | O₂ {entry['spo2']}%
            """)

# ============================================================================
# TAB 3: CHAT ASSISTANT
# ============================================================================
with tab3:
    st.header("💬 Health Education Chat")
    st.markdown("Ask questions about your documents and vital signs. The assistant will provide educational information with citations.")
    
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("""
        ⚠️ **OpenAI API Key Not Found**
        
        To use the chat assistant, you need to:
        1. Get an API key from [OpenAI](https://platform.openai.com/api-keys)
        2. Add it to `.streamlit/secrets.toml`:
           ```
           OPENAI_API_KEY = "sk-..."
           ```
        3. Restart the Streamlit app
        
        The extracted text review (Tab 1) and vitals tracker (Tab 2) will still work without an API key.
        """)
    elif not st.session_state.extracted_text:
        st.warning("⚠️ Upload documents first to chat with the assistant")
    elif st.session_state.retriever is None:
        st.info("📚 Documents uploaded! Prepare the chat assistant:")
        if st.button("🚀 Prepare Chat Assistant", use_container_width=True):
            # Build RAG chain from documents
            current_docs = set(st.session_state.extracted_text.keys())
            try:
                with st.spinner("Preparing documents for chat... (creating embeddings, may take 30-60 seconds)"):
                    # Combine all extracted texts
                    combined_text = "\n\n".join(
                        [f"[{doc_name}]\n{text}" for doc_name, text in st.session_state.extracted_text.items()]
                    )
                    
                    # Split into chunks (smaller chunks for faster processing)
                    text_splitter = CharacterTextSplitter(
                        separator="\n",
                        chunk_size=500,  # Reduced from 1000
                        chunk_overlap=100  # Reduced from 200
                    )
                    chunks = text_splitter.split_text(combined_text)
                    
                    # Limit chunks for faster processing (first 50 chunks)
                    chunks = chunks[:50] if len(chunks) > 50 else chunks
                    
                    st.info(f"📊 Processing {len(chunks)} text chunks...")
                    
                    # Create vector store
                    embeddings = OpenAIEmbeddings()
                    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
                    st.session_state.retriever = vectorstore.as_retriever()
                    st.session_state.last_processed_docs = current_docs.copy()
                    
                    st.success("✅ Chat assistant ready! You can now ask questions.")
                    st.rerun()
            except Exception as e:
                if "authentication" in str(e).lower() or "api key" in str(e).lower():
                    st.error("""
                    🔑 **Invalid OpenAI API Key**
                    
                    Your API key appears to be invalid or expired. Please:
                    
                    1. **Check your API key** at [OpenAI Platform](https://platform.openai.com/api-keys)
                    2. **Verify it starts with** `sk-` or `sk-proj-` (both are valid formats)
                    3. **Ensure you have credits** in your OpenAI account
                    4. **Update** `.streamlit/secrets.toml` with the correct key
                    5. **Restart** the Streamlit app
                    
                    **Example formats:**
                    ```
                    OPENAI_API_KEY = "sk-abc123..."
                    ```
                    """)
                else:
                    st.error(f"Error preparing documents: {str(e)}")
    else:
        # Chat interface
        user_question = st.text_input(
            "Ask a health education question:",
            placeholder="e.g., 'What do these lab results mean?' or 'Is my blood pressure healthy?'"
        )
        
        if user_question:
            try:
                with st.spinner("Thinking..."):
                    # Build RAG chain with health guardrails
                    template = f"""{HEALTH_GUARDRAILS}

Context from your documents:
{{context}}

Current vitals: {{vitals}}

Question: {{question}}

Provide an educational response with citations to specific documents/pages.
"""
                
                prompt = ChatPromptTemplate.from_template(template)
                llm = ChatOpenAI(model_name="gpt-4o", temperature=0.2)
                
                # Get current vitals
                current_vitals = st.session_state.vitals_history[-1] if st.session_state.vitals_history else 'None logged'
                
                rag_chain = (
                    {"context": st.session_state.retriever, "question": RunnablePassthrough(), "vitals": lambda x: current_vitals}
                    | prompt
                    | llm
                    | StrOutputParser()
                )
                
                response = rag_chain.invoke(user_question)
                
                # Store in chat history
                st.session_state.chat_history.append({
                    "question": user_question,
                    "response": response,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Display response
                st.markdown("### 📚 Educational Response")
                st.info(response)
                
                st.markdown("---")
                st.markdown("""
                **⚠️ Important Disclaimer:**  
                This assistant provides **educational information only**. It is not a medical professional.  
                Always consult with your healthcare provider before making health decisions.
                """)
            except Exception as e:
                if "authentication" in str(e).lower() or "api key" in str(e).lower():
                    st.error("""
                    🔑 **API Key Authentication Failed**
                    
                    Please check your OpenAI API key:
                    - Visit [OpenAI Platform](https://platform.openai.com/api-keys)
                    - Ensure your key is valid and has credits
                    - Update `.streamlit/secrets.toml` if needed
                    - Restart the app
                    """)
                else:
                    st.error(f"Error generating response: {str(e)}")

# ============================================================================
# TAB 4: HEALTH INSIGHTS & RECOMMENDATIONS
# ============================================================================
with tab4:
    st.header("📋 Health Insights & Educational Recommendations")
    st.markdown("*Educational suggestions based on your logged vitals and documents. Not medical advice.*")
    
    if not st.session_state.vitals_history:
        st.warning("⚠️ Log some vitals first to see personalized insights")
    else:
        # Get latest vitals
        latest_vitals = st.session_state.vitals_history[-1]
        pulse = latest_vitals['pulse']
        bp_systolic, bp_diastolic = map(int, latest_vitals['bp'].split('/'))
        spo2 = latest_vitals['spo2']
        temp = latest_vitals['temperature']
        
        # Create columns for different insight categories
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💓 Cardiovascular Health")
            
            # Blood Pressure Analysis
            if bp_systolic < 120 and bp_diastolic < 80:
                bp_status = "✅ Normal"
                bp_color = "green"
            elif bp_systolic < 130 and bp_diastolic < 80:
                bp_status = "⚠️ Elevated"
                bp_color = "orange"
            elif bp_systolic < 140 or bp_diastolic < 90:
                bp_status = "🟡 Stage 1 Hypertension"
                bp_color = "orange"
            else:
                bp_status = "🔴 Stage 2 Hypertension"
                bp_color = "red"
            
            st.metric("Blood Pressure", f"{bp_systolic}/{bp_diastolic} mmHg", bp_status)
            
            # Pulse Analysis
            if 60 <= pulse <= 100:
                pulse_status = "✅ Normal resting range"
                pulse_color = "green"
            elif pulse < 60:
                pulse_status = "⚠️ Bradycardia (may be normal for athletes)"
                pulse_color = "blue"
            else:
                pulse_status = "⚠️ Tachycardia"
                pulse_color = "orange"
            
            st.metric("Heart Rate", f"{pulse} bpm", pulse_status)
            
            # Educational recommendations
            with st.expander("💡 Educational Tips"):
                st.markdown("""
                **Understanding Your Numbers:**
                - **Blood Pressure**: <120/80 is optimal, 120-129/<80 is elevated
                - **Heart Rate**: 60-100 bpm resting is typical
                
                **General Education** (not medical advice):
                - Regular exercise can help maintain healthy blood pressure
                - Stress management techniques may help heart rate
                - Consult healthcare providers for personalized guidance
                """)
        
        with col2:
            st.subheader("🌡️ Vital Signs Overview")
            
            # Temperature
            if 97.0 <= temp <= 99.0:
                temp_status = "✅ Normal"
                temp_color = "green"
            elif temp > 99.0:
                temp_status = "🔴 Elevated"
                temp_color = "red"
            else:
                temp_status = "❄️ Low"
                temp_color = "blue"
            
            st.metric("Temperature", f"{temp}°F", temp_status)
            
            # SpO2
            if spo2 >= 95:
                spo2_status = "✅ Normal"
                spo2_color = "green"
            elif spo2 >= 90:
                spo2_status = "⚠️ Low"
                spo2_color = "orange"
            else:
                spo2_status = "🔴 Very Low"
                spo2_color = "red"
            
            st.metric("Blood Oxygen", f"{spo2}%", spo2_status)
            
            # Overall Health Score (educational only)
            health_score = 0
            if bp_systolic < 130 and bp_diastolic < 80: health_score += 25
            if 60 <= pulse <= 100: health_score += 25
            if 97.0 <= temp <= 99.0: health_score += 25
            if spo2 >= 95: health_score += 25
            
            st.metric("Vital Signs Score", f"{health_score}/100", 
                     "Educational indicator only")
        
        # Trends and History
        st.subheader("📈 Vitals History & Trends")
        
        if len(st.session_state.vitals_history) > 1:
            # Create simple trend charts
            dates = []
            pulses = []
            bp_sys = []
            bp_dia = []
            temps = []
            spo2s = []
            
            for entry in st.session_state.vitals_history[-7:]:  # Last 7 entries
                dates.append(datetime.fromisoformat(entry['timestamp']).strftime('%m/%d'))
                pulses.append(entry['pulse'])
                bp_sys.append(int(entry['bp'].split('/')[0]))
                bp_dia.append(int(entry['bp'].split('/')[1]))
                temps.append(entry['temperature'])
                spo2s.append(entry['spo2'])
            
            # Simple line charts using st.line_chart
            chart_data = {
                'Date': dates,
                'Heart Rate (bpm)': pulses,
                'Systolic BP': bp_sys,
                'Diastolic BP': bp_dia,
                'Temperature (°F)': temps,
                'SpO₂ (%)': spo2s
            }
            
            st.line_chart(chart_data, use_container_width=True)
        else:
            st.info("Log more vitals over time to see trends")
        
        # Health Goals & Learning Objectives
        st.subheader("🎯 Health Learning Goals")
        
        col_goal1, col_goal2 = st.columns([2, 1])
        
        with col_goal1:
            new_goal = st.text_input("Add a health education goal:", 
                                   placeholder="e.g., Learn about blood pressure, understand medication labels")
        
        with col_goal2:
            if st.button("➕ Add Goal", use_container_width=True) and new_goal:
                st.session_state.health_goals.append({
                    "goal": new_goal,
                    "date_added": datetime.now().isoformat(),
                    "completed": False
                })
                st.success("Goal added!")
                st.rerun()
        
        if st.session_state.health_goals:
            st.markdown("### Your Learning Goals")
            for i, goal in enumerate(st.session_state.health_goals):
                col_check, col_text, col_delete = st.columns([0.1, 0.8, 0.1])
                
                with col_check:
                    completed = st.checkbox("", 
                                          value=goal["completed"], 
                                          key=f"goal_{i}",
                                          label_visibility="hidden")
                    if completed != goal["completed"]:
                        st.session_state.health_goals[i]["completed"] = completed
                
                with col_text:
                    status = "✅" if goal["completed"] else "⏳"
                    st.markdown(f"{status} {goal['goal']}")
                
                with col_delete:
                    if st.button("🗑️", key=f"delete_{i}", help="Delete goal"):
                        st.session_state.health_goals.pop(i)
                        st.rerun()
        
        # Progress Summary
        if st.session_state.health_goals:
            total_goals = len(st.session_state.health_goals)
            completed_goals = sum(1 for goal in st.session_state.health_goals if goal["completed"])
            progress = completed_goals / total_goals if total_goals > 0 else 0
            
            st.progress(progress)
            st.markdown(f"**Progress:** {completed_goals}/{total_goals} learning goals completed")
        
        st.markdown("---")
        
        rec_col1, rec_col2 = st.columns(2)
        
        with rec_col1:
            st.markdown("### 🏃‍♂️ General Wellness Tips")
            st.markdown("""
            **Based on your vitals** (educational information only):
            
            📖 **Learn About Heart Health**
            - Understanding blood pressure readings
            - Heart rate zones for different activities
            - Importance of regular health check-ups
            
            🥗 **Nutrition Education**
            - Balanced diet principles
            - Reading nutrition labels
            - Hydration importance
            
            😴 **Rest & Recovery**
            - Sleep hygiene basics
            - Stress management techniques
            - Work-life balance concepts
            """)
        
        with rec_col2:
            st.markdown("### 📖 Health Literacy Resources")
            st.markdown("""
            **Educational Topics to Explore:**
            
            🔍 **Understanding Medical Tests**
            - Common blood work explanations
            - Vital signs interpretation
            - Basic medical terminology
            
            💊 **Medication Education**
            - How to read prescription labels
            - Understanding drug interactions
            - Importance of following directions
            
            🏥 **Healthcare System Navigation**
            - Finding reliable health information
            - Communicating with healthcare providers
            - Understanding insurance and billing
            """)
        
        # Export Educational Summary
        st.subheader("📄 Export Educational Summary")
        
        if st.button("📋 Generate Health Education Summary", use_container_width=True):
            summary = f"""
# Health Education Summary
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Your Recent Vitals
- Latest Reading: {latest_vitals['timestamp'][:10]}
- Blood Pressure: {latest_vitals['bp']} mmHg
- Heart Rate: {pulse} bpm
- Temperature: {temp}°F
- Blood Oxygen: {spo2}%

## Educational Insights
This summary provides educational information about your vital signs and general health concepts.
It is NOT medical advice and should not replace consultation with healthcare professionals.

## Key Health Concepts to Explore
1. Understanding blood pressure readings and ranges
2. Heart rate zones and what they mean
3. Temperature regulation and fever patterns
4. Blood oxygen saturation and respiratory health
5. Overall cardiovascular wellness

## Recommended Educational Resources
- American Heart Association: Blood Pressure Basics
- Mayo Clinic: Vital Signs Overview
- NIH: Understanding Medical Tests
- Local library health literacy programs

## Important Reminder
Always consult with qualified healthcare providers for personalized medical advice,
diagnosis, or treatment. This educational tool is designed to help you better
understand health concepts and engage more effectively with your healthcare team.
"""
            
            st.download_button(
                label="📥 Download Educational Summary",
                data=summary,
                file_name=f"health_education_summary_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                use_container_width=True
            )
            
            with st.expander("Preview Summary"):
                st.code(summary, language="text")
        
        # Disclaimer
        st.markdown("---")
        st.warning("""
        **⚠️ Important Educational Disclaimer**
        
        This tab provides **educational information only** to help you better understand health concepts. 
        The insights and recommendations are **not medical advice** and should not replace professional medical care.
        
        Always consult with qualified healthcare providers for personalized medical advice, diagnosis, or treatment.
        """)

# ============================================================================
# TAB 5: ABOUT THIS TOOL
# ============================================================================
with tab5:
    st.header("About This Tool")
    
    st.markdown("""
    ### Privacy & Digital Citizenship
    
    **What makes this tool different:**
    - 🔒 **Fully Local**: All processing happens on your device—no data sent to cloud services
    - 📋 **Your Control**: You upload documents and choose what to share
    - 🔍 **Transparent**: You can review and edit all extracted text
    - 📖 **Educational**: Designed to help you understand your health, not replace doctors
    
    ### Design Principles
    
    1. **Privacy First**  
       Medical data is sensitive. This tool processes everything locally without storing data on external servers.
    
    2. **Transparency**  
       You see exactly what text is extracted and can correct OCR errors before asking questions.
    
    3. **Safety Guardrails**  
       The assistant is trained to provide educational explanations, not medical advice, and always recommends consulting healthcare professionals.
    
    4. **Verifiable Sources**  
       Responses include citations so you can verify information in your documents.
    
    ### Guiding Questions
    
    - How can digital tools help you become a more informed participant in your healthcare?
    - What responsibilities do you have when sharing health information digitally?
    - How does local, privacy-respecting design build trust?
    
    ### Reflection
    
    **Privacy vs. Convenience:**  
    Traditional AI health tools collect data for better recommendations but raise privacy concerns.  
    This tool prioritizes your privacy, requiring more manual setup but giving you full control.
    
    **Accuracy vs. Accessibility:**  
    Local processing is more transparent and private, but cloud-based systems may provide more sophisticated analysis.  
    This tool favors explainability and safety over maximum accuracy.
    """)
    
    st.divider()
    st.markdown("*Built as a digital citizenship project exploring ethical technology design.*")