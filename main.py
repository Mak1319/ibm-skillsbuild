
import streamlit as st
import plotly.graph_objects as ugo
from tempfile import NamedTemporaryFile
import os
from agent import app , AgentState 
from utils.document import extract_text 


st.set_page_config(page_title="AI Resume Reviewer", layout="wide")

st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        .icon-header { font-size: 24px; margin-right: 10px; vertical-align: middle; }
        .positive { color: #2ecc71; }
        .negative { color: #e74c3c; }
        .info { color: #3498db; }
        .warning { color: #f39c12; }
    </style>
""", unsafe_allow_html=True)

def create_gauge(title, value, min_val, max_val, color="#1f2c56"):
    """Creates a professional Odometer-style gauge with fixed color logic"""
    range_size = max_val - min_val
    step1 = min_val + (range_size * 0.4)
    step2 = min_val + (range_size * 0.7)
    
    fig = ugo.Figure(ugo.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 16}},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickwidth': 1},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "white",
            'steps': [
                {'range': [min_val, step1], 'color': "#ffefef"},
                {'range': [step1, step2], 'color': "#fff9e6"},
                {'range': [step2, max_val], 'color': "#e6fffa"}
            ],
        }
    ))
    fig.update_layout(height=180, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='rgba(0,0,0,0)')
    return fig

st.markdown("<h1><i class='fa-solid fa-microchip info'></i> AI Agent: Resume & Candidate Analyzer</h1>", unsafe_allow_html=True)


uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx", "ppt", "pptx"])

if uploaded_file:
    with NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    if 'analysis_results' not in st.session_state:
        with st.status("🚀 Agent is analyzing...", expanded=True) as status:
            resume_content = extract_text(tmp_path)
            inputs = AgentState(resume=resume_content)
            config = {"configurable": {"thread_id": "current_user"}}

            for output in app.stream(inputs, config=config):
                for node_name, _ in output.items():
                    st.write(f"✔️ {node_name.replace('_', ' ').title()} finished.")
            
            final_snapshot = app.get_state(config)
            st.session_state.analysis_results = final_snapshot.values
            status.update(label="Analysis Complete!", state="complete", expanded=False)
    
    os.remove(tmp_path)

    state_data = st.session_state.analysis_results
    st.divider()
    
    # --- Step 2: Qualitative Analysis with Icons ---
    st.markdown("<h3><i class='fa-solid fa-magnifying-glass'></i> Qualitative Analysis</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container(border=True):
            st.markdown("<h4><i class='fa-solid fa-user-check positive'></i> Candidate: Strengths</h4>", unsafe_allow_html=True)
            for p in state_data.get('positive_points', []): st.markdown(f"- {p}")
        with st.container(border=True):
            st.markdown("<h4><i class='fa-solid fa-user-xmark negative'></i> Candidate: Risks</h4>", unsafe_allow_html=True)
            for p in state_data.get('negative_points', []): st.markdown(f"- {p}")

    with col2:
        with st.container(border=True):
            st.markdown("<h4><i class='fa-solid fa-file-circle-check info'></i> Resume: Highlights</h4>", unsafe_allow_html=True)
            for p in state_data.get('positive_points_resume', []): st.markdown(f"- {p}")
        with st.container(border=True):
            st.markdown("<h4><i class='fa-solid fa-file-circle-exclamation warning'></i> Resume: Flaws</h4>", unsafe_allow_html=True)
            for p in state_data.get('negative_points_resume', []): st.markdown(f"- {p}")

    # --- Step 3: Odometer Gauges ---
    st.divider()
    st.markdown("<h3><i class='fa-solid fa-chart-line'></i> Scoring Dashboard</h3>", unsafe_allow_html=True)
    
    full_state_obj = AgentState(**state_data)
    raw_scores = full_state_obj.normalize_avg()

    s_cand_neg, s_cand_pos = raw_scores[0], raw_scores[1]
    s_res_neg, s_res_pos = raw_scores[2], raw_scores[3]

    g1, g2, g3, g4 = st.columns(4)
    with g1:
        st.markdown("<p style='text-align:center'><i class='fa-solid fa-skull-crossbones negative'></i> Cand. Neg</p>", unsafe_allow_html=True)
        st.plotly_chart(create_gauge("", s_cand_neg, 0, -1000, "#e74c3c"), use_container_width=True)
    with g2:
        st.markdown("<p style='text-align:center'><i class='fa-solid fa-star positive'></i> Cand. Pos</p>", unsafe_allow_html=True)
        st.plotly_chart(create_gauge("", s_cand_pos, 0, 1000, "#2ecc71"), use_container_width=True)
    with g3:
        st.markdown("<p style='text-align:center'><i class='fa-solid fa-triangle-exclamation warning'></i> Res. Flaws</p>", unsafe_allow_html=True)
        st.plotly_chart(create_gauge("", s_res_neg, 0, -1000, "#f39c12"), use_container_width=True)
    with g4:
        st.markdown("<p style='text-align:center'><i class='fa-solid fa-file-contract info'></i> Res. Quality</p>", unsafe_allow_html=True)
        st.plotly_chart(create_gauge("", s_res_pos, 0, 1000, "#3498db"), use_container_width=True)

    # --- Step 4: Overall Match ---
    st.divider()
    total_match = (abs(s_cand_neg) + s_cand_pos + abs(s_res_neg) + s_res_pos) / 4
    
    _, center_col, _ = st.columns([1, 2, 1])
    with center_col:
        st.markdown("<h2 style='text-align:center'><i class='fa-solid fa-award info'></i> OVERALL MATCH</h2>", unsafe_allow_html=True)
        st.plotly_chart(create_gauge("", total_match, 0, 1000, "#1f2c56"), use_container_width=True)

else:
    if 'analysis_results' in st.session_state: del st.session_state.analysis_results
    st.info("Upload a file to begin analysis.")