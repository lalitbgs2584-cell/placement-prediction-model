import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="College Placement Predictor",
    page_icon="🎓",
    layout="centered",
)

# ── Load model & scaler (cached so they load only once) ──────────────────────
@st.cache_resource
def load_artifacts():
    model  = tf.keras.models.load_model("placement_ann_model.keras")
    scaler = joblib.load("placement_scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# ── Helper: build feature vector ─────────────────────────────────────────────
def build_features(cgpa, internships, projects, workshops,
                   aptitude, soft_skills, extracurricular,
                   placement_training, ssc, hsc):
    extra  = 1 if extracurricular == "Yes" else 0
    train  = 1 if placement_training == "Yes" else 0

    base = [cgpa, internships, projects, workshops,
            aptitude, soft_skills, extra, train, ssc, hsc]

    # Engineered features (must match training order)
    cgpa_x_apt   = cgpa * aptitude
    total_marks  = ssc + hsc
    train_x_ext  = train * extra
    cgpa_x_proj  = cgpa * projects
    apt_x_soft   = aptitude * soft_skills

    return np.array(base + [cgpa_x_apt, total_marks,
                             train_x_ext, cgpa_x_proj, apt_x_soft],
                    dtype=np.float32).reshape(1, -1)

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🎓 College Placement Predictor")
st.markdown("Fill in the student details below and click **Predict** to see the placement probability.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Academic")
    cgpa       = st.slider("CGPA",            min_value=0.0, max_value=10.0, value=7.5, step=0.1)
    ssc        = st.slider("SSC Marks (%)",   min_value=0,   max_value=100,  value=70)
    hsc        = st.slider("HSC Marks (%)",   min_value=0,   max_value=100,  value=75)
    aptitude   = st.slider("Aptitude Score",  min_value=0,   max_value=100,  value=70)
    soft_skills = st.slider("Soft Skills Rating", min_value=0.0, max_value=5.0, value=4.0, step=0.1)

with col2:
    st.subheader("Experience & Activities")
    internships  = st.number_input("Internships",             min_value=0, max_value=10, value=1)
    projects     = st.number_input("Projects",                min_value=0, max_value=10, value=1)
    workshops    = st.number_input("Workshops / Certifications", min_value=0, max_value=10, value=1)
    extracurricular    = st.selectbox("Extracurricular Activities", ["Yes", "No"])
    placement_training = st.selectbox("Placement Training",         ["Yes", "No"])

st.divider()

if st.button("🔍 Predict Placement", use_container_width=True):
    features = build_features(
        cgpa, internships, projects, workshops,
        aptitude, soft_skills, extracurricular,
        placement_training, ssc, hsc
    )
    features_scaled = scaler.transform(features)
    prob = float(model.predict(features_scaled, verbose=0)[0][0])
    label = "✅ Placed" if prob >= 0.5 else "❌ Not Placed"

    st.subheader("Prediction Result")
    st.metric(label="Outcome", value=label)

    # Probability bar
    st.progress(prob)
    st.caption(f"Placement probability: **{prob * 100:.1f}%**")

    # Colour-coded message
    if prob >= 0.75:
        st.success("Strong placement prospects! Keep up the great work.")
    elif prob >= 0.5:
        st.info("Moderate placement prospects. A bit more preparation could help.")
    else:
        st.warning("Lower placement prospects. Focus on improving weak areas.")
