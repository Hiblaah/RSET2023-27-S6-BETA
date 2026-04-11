from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt
import streamlit as st
from keybert import KeyBERT
import pandas as pd
import joblib
import time
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
import json
import os


# -------------------------------
# PHONE LOGIN + REGISTRATION
# -------------------------------
USER_FILE = "users.json"

if not os.path.exists(USER_FILE):
    with open(USER_FILE, "w") as f:
        json.dump({}, f)

with open(USER_FILE) as f:
    stored_users = json.load(f)
# -------------------------------
# HISTORY STORAGE FILE
# -------------------------------

HISTORY_FILE = "prediction_history.csv"

if not os.path.exists(HISTORY_FILE):
    pd.DataFrame(columns=[
        "phone",
        "case_number",
        "court",
        "case_type",
        "prediction",
        "confidence",
        "summary"
    ]).to_csv(HISTORY_FILE, index=False)

if "users" not in st.session_state:
    st.session_state.users = {}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "current_user" not in st.session_state:
    st.session_state.current_user = None


def login():

    st.title("📱 Legal AI Login")

    phone = st.text_input("Enter Phone Number")

    if phone:

        # Phone validation
        if not phone.isdigit() or len(phone) != 10:
            st.error("Please enter a valid 10-digit phone number.")
            return

        # ---------------- LOGIN ----------------
        if phone in stored_users:

            if st.button("Login"):

                st.session_state.logged_in = True
                st.session_state.current_user = stored_users[phone]
                st.session_state.phone = phone

                st.success(f"Welcome {stored_users[phone]['name']}!")
                st.rerun()

        # ---------------- REGISTER ----------------
        else:

            st.warning("New user detected. Please register.")

            name = st.text_input("Enter Your Name")

            profession = st.selectbox(
                "Profession",
                ["Lawyer", "Law Student"]
            )

            if st.button("Register"):

                if name.strip() == "":
                    st.error("Please enter your name")

                else:

                    stored_users[phone] = {
                        "name": name,
                        "profession": profession
                    }

                    with open(USER_FILE, "w") as f:
                        json.dump(stored_users, f)

                    st.session_state.logged_in = True
                    st.session_state.current_user = stored_users[phone]
                    st.session_state.phone = phone

                    st.success("Registration successful!")
                    st.rerun()


# -------------------------------
# Page Configuration
# -------------------------------

st.set_page_config(
    page_title="Legal Outcome Prediction System",
    layout="centered"
)

if not st.session_state.logged_in:
    login()
    st.stop()


# -------------------------------
# LOAD MODEL
# -------------------------------

model = joblib.load("outcome_model.pkl")
# Load duration prediction model
try:
    duration_model = joblib.load("duration_model.pkl")
except:
    duration_model = None
kw_model = KeyBERT()


# -------------------------------
# LOAD DATASET
# -------------------------------

df = pd.read_csv("legal_cases.csv")

duration_col = None

for col in df.columns:
    if "duration" in col.lower() or "time" in col.lower():
        duration_col = col
        break

summary_col = None
judgement_col = None

for col in df.columns:

    col_lower = col.lower().strip()

    if "summary" in col_lower:
        summary_col = col

    if "judgement" in col_lower or "judgment" in col_lower:
        judgement_col = col

if summary_col and judgement_col:
    df["text_data"] = df[summary_col].fillna("") + " " + df[judgement_col].fillna("")
else:
    df["text_data"] = ""
# -------------------------------
# BUILD SEARCH INDEX FOR CHATBOT
# -------------------------------

vectorizer = TfidfVectorizer(stop_words="english")

tfidf_matrix = vectorizer.fit_transform(df["text_data"])

# -------------------------------
# PDF GENERATOR
# -------------------------------

def create_case_pdf(case_row):

    styles = getSampleStyleSheet()

    buffer = BytesIO()

    pdf = SimpleDocTemplate(buffer)

    story = []

    story.append(Paragraph("Legal Case Report", styles['Title']))
    story.append(Spacer(1,20))

    for column in case_row.index:

        text = f"<b>{column}:</b> {case_row[column]}"

        story.append(Paragraph(text, styles['BodyText']))
        story.append(Spacer(1,10))

    pdf.build(story)

    buffer.seek(0)

    return buffer


# -------------------------------
# SIDEBAR
# -------------------------------

st.sidebar.title("⚖️ Legal AI System")
if st.session_state.current_user:
 st.sidebar.write(
    f"👤 {st.session_state.current_user['name']} "
    f"({st.session_state.current_user['profession']})"
)

menu = st.sidebar.radio(
    "Navigation",
    ["Home","Prediction","AI Lawyer Chatbot","History","Analytics","Model Training"]
)

if st.sidebar.button("Logout"):

    st.session_state.logged_in = False
    st.session_state.current_user = None
    st.session_state.phone = None
    st.rerun()


# -------------------------------
# HOME
# -------------------------------

if menu == "Home":

    st.title("⚖️ Legal Case Outcome Prediction System")
    if st.session_state.current_user:
       st.write(
          f"👤WELCOME {st.session_state.current_user['name']} "
        )
    st.markdown("""
    This system predicts **legal case outcomes** using AI.

    Features:
    - Case Outcome Prediction
    - Probability Visualization
    - Keyword Extraction
    - Similar Case Retrieval
    - PDF Case Reports
    """)

    st.subheader("AI-based prediction of legal case outcomes")


# -------------------------------
# PREDICTION
# -------------------------------

elif menu == "Prediction":

    st.header("📄 Enter Case Details")

    case_number = st.text_input("Case Number")

    court_level = st.selectbox(
        "Court Level",
        ["District Court","High Court","Supreme Court",
         "Magistrate Court","Sessions Court","Sub Court","Munsiff Court"]
    )

    case_type = st.selectbox("Case Type",["Civil","Criminal"])

    state = st.selectbox("State",["Kerala"])

    district = st.selectbox(
        "District",
        ["Kasaragod","Kannur","Wayanad","Kozhikode","Malappuram","Palakkad",
         "Thrissur","Ernakulam","Idukki","Kottayam","Alappuzha",
         "Pathanamthitta","Kollam","Thiruvanthapuram"]
    )

    year_filed = st.number_input("Year Filed",min_value=1980,max_value=2025)

    case_summary = st.text_area("Case Summary",height=120)

    if st.button("🔍 Predict Case Outcome"):

        if case_summary.strip() == "":
            st.warning("Please enter case summary")
            st.stop()

        probabilities = model.predict_proba([case_summary])[0]
        classes = model.classes_

        st.subheader("📊 Outcome Prediction")

        fig, ax = plt.subplots(figsize=(6,6))

        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")

        colors = ["#4F46E5","#06B6D4","#10B981","#F59E0B","#EF4444"]

        ax.pie(
            probabilities,
            labels=classes,
            autopct="%1.1f%%",
            startangle=140,
            colors=colors[:len(classes)],
            textprops={"color":"white","fontsize":11},
            wedgeprops={"edgecolor":"white","linewidth":2}
        )

        ax.set_title(
            "Predicted Case Outcome Probability",
            fontsize=14,
            color="white",
            weight="bold"
        )

        st.pyplot(fig)

        predicted_class = classes[probabilities.argmax()]
        confidence = probabilities.max()*100

        st.success(f"🏆 Predicted Outcome: {predicted_class}")
        st.info(f"Confidence Score: {confidence:.2f}%")
        # -------------------------------
        # SAVE PREDICTION HISTORY
        # -------------------------------

        history = pd.read_csv(HISTORY_FILE)

        new_entry = pd.DataFrame([{
                                    "phone":st.session_state.phone,
                                    "case_number": case_number,
                                    "court": court_level,
                                    "case_type": case_type,
                                    "prediction": predicted_class,
                                    "confidence": confidence,
                                    "summary": case_summary
                                 }])

        history = pd.concat([history, new_entry])
        history.to_csv(HISTORY_FILE, index=False)
        st.success("Prediction saved to history!")
        # Duration
        # -------------------------------
        # ML BASED CASE DURATION
        # -------------------------------

        st.subheader("⏳ Estimated Case Duration (AI Prediction)")

        predicted_duration = None

        if duration_model:

                predicted_duration = duration_model.predict([case_summary])[0]

                st.success(f"Estimated Duration: {predicted_duration:.1f} months")

        else:

                 st.warning("Duration model not trained yet.")
                 st.subheader("⏳ Estimated Case Duration")

        # -------------------------------
        # DELAY PROBABILITY
        # -------------------------------

        st.subheader("⚠️ Delay Risk Analysis")

        if predicted_duration is not None and duration_col is not None:

            avg_duration = df[duration_col].mean()

            if predicted_duration > avg_duration:

                delay_prob = 75
                risk = "High Delay Risk"
                st.error(f"🚨 {risk} ({delay_prob}%)")

            elif predicted_duration > avg_duration * 0.8:

                delay_prob = 50
                risk = "Medium Delay Risk"
                st.warning(f"⚠️ {risk} ({delay_prob}%)")

            else:

                delay_prob = 25
                risk = "Low Delay Risk"
                st.success(f"✅ {risk} ({delay_prob}%)")

        else:

            st.info("Delay probability cannot be calculated yet.")


        

        # -------------------------------
        # SIMILAR CASES
        # -------------------------------

        st.subheader("📚 Similar Cases")

        vectorizer = TfidfVectorizer(stop_words="english")

        tfidf_matrix = vectorizer.fit_transform(df["text_data"])

        query_vec = vectorizer.transform([case_summary])

        similarity = cosine_similarity(query_vec, tfidf_matrix)

        top_indices = similarity.argsort()[0][-5:][::-1]

        similar_cases = df.iloc[top_indices]

        for i, row in similar_cases.iterrows():

            st.markdown("---")

            st.write(f"### Case {i}")

            st.dataframe(row.to_frame())

            pdf = create_case_pdf(row)

            st.download_button(
                label="Download Case PDF",
                data=pdf,
                file_name=f"case_{i}.pdf",
                mime="application/pdf"
            )


        # -------------------------------
        # REPORT DOWNLOAD
        # -------------------------------

        report = f"""
CASE ANALYSIS REPORT

Case Number: {case_number}
Court Level: {court_level}
Case Type: {case_type}

Predicted Outcome: {predicted_class}
Confidence Score: {confidence:.2f}%

Estimated Duration: {predicted_duration if predicted_duration else "Not Available"} months

Case Summary:
{case_summary}
"""

        st.download_button(
            label="Download Case Report",
            data=report,
            file_name="case_report.txt"
        )
# -------------------------------
# DATASET
# -------------------------------
# -------------------------------
# HISTORY PAGE
# -------------------------------

elif menu == "History":

    st.title("Prediction History")

    history = pd.read_csv(HISTORY_FILE)

    user_history = history[history["phone"].astype(str) == str(st.session_state.phone)]
    if len(user_history) == 0:

        st.info("No predictions recorded yet.")

    else:

        st.dataframe(user_history)

        
# -------------------------------
# LEGAL ANALYTICS DASHBOARD
# -------------------------------
elif menu == "Analytics":

    import plotly.express as px

    st.title("📊 Legal Analytics Dashboard")

    st.markdown("""
<style>
.stApp {
    background-color: #0E1117;
    color: white;
}

h1, h2, h3 {
    color: #E5E7EB;
}

.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

    history = pd.read_csv(HISTORY_FILE)

    user_history = history[history["phone"].astype(str) == str(st.session_state.phone)]

    if len(user_history) == 0:
        st.info("No analytics data available yet.")
        st.stop()

# -------------------------------
# DASHBOARD METRICS
# -------------------------------

    col1,col2,col3 = st.columns(3)

    col1.metric("Total Predictions", len(user_history))
    col2.metric("Average Confidence", f"{user_history['confidence'].mean():.1f}%")
    col3.metric("Unique Case Types", user_history["case_type"].nunique())

    st.markdown("---")

# -------------------------------
# CHART LAYOUT
# -------------------------------

    col1, col2 = st.columns(2)

# -------------------------------
# OUTCOME DISTRIBUTION (ANIMATED)
# -------------------------------

    with col1:

        st.subheader("Case Outcome Distribution")

        outcome_counts = user_history["prediction"].value_counts().reset_index()
        outcome_counts.columns = ["Outcome","Count"]

        fig = px.bar(
            outcome_counts,
            x="Outcome",
            y="Count",
            text="Count",
            template="plotly_dark"
        )

        fig.update_layout(
            height=300,
            xaxis_tickfont=dict(size=9),   # smaller x-axis font
            yaxis_title="Cases",
            xaxis_title="Outcome",
            title=None
        )

        st.plotly_chart(fig,use_container_width=True)

# -------------------------------
# CASE TYPE PIE CHART
# -------------------------------

    with col2:

        st.subheader("Case Type Distribution")

        type_counts = user_history["case_type"].value_counts().reset_index()
        type_counts.columns = ["Case Type","Count"]

        fig2 = px.pie(
            type_counts,
            names="Case Type",
            values="Count",
            template="plotly_dark",
            hole=0.45
        )

        fig2.update_layout(height=300)

        st.plotly_chart(fig2,use_container_width=True)

# -------------------------------
# CONFIDENCE SCORE CHART
# -------------------------------

    st.subheader("Prediction Confidence Distribution")

    fig3 = px.histogram(
        user_history,
        x="confidence",
        nbins=10,
        template="plotly_dark"
    )

    fig3.update_layout(
        height=300,
        xaxis_title="Confidence %",
        yaxis_title="Frequency"
    )

    st.plotly_chart(fig3,use_container_width=True)

# -------------------------------
# AI LEGAL INSIGHTS PANEL
# -------------------------------

    st.markdown("---")
    st.subheader("⚖️ AI Legal Insights")

    most_common_outcome = user_history["prediction"].value_counts().idxmax()
    most_common_type = user_history["case_type"].value_counts().idxmax()
    avg_conf = user_history["confidence"].mean()

    st.info(f"""
**Key Insights from Your Predictions**

• Most predicted outcome: **{most_common_outcome}**

• Most common case type: **{most_common_type}**

• Average prediction confidence: **{avg_conf:.2f}%**

**AI Interpretation**

Your prediction history suggests that similar disputes often resolve based on documentation strength, legal precedents, and court interpretation patterns.
""")
# -------------------------------
# MODEL TRAINING
# -------------------------------

elif menu == "Model Training":

    st.header("🤖 Model Training")

    outcome_col = None

    for col in df.columns:
        col_lower = col.lower().strip()

        if "outcome" in col_lower or "result" in col_lower or "decision" in col_lower:
            outcome_col = col
            break

    if outcome_col:

        df_clean = df.dropna(subset=[outcome_col])

        X = df_clean["text_data"]
        y = df_clean[outcome_col]

        X_train,X_test,y_train,y_test = train_test_split(
            X,y,test_size=0.2,random_state=42
        )

        pipeline = Pipeline([
            ("tfidf",TfidfVectorizer(stop_words="english",max_features=5000)),
            ("classifier",LogisticRegression(max_iter=1000))
        ])

        pipeline.fit(X_train,y_train)

        accuracy = pipeline.score(X_test,y_test)

        st.success(f"Outcome model trained! Accuracy: {accuracy:.2f}")

        joblib.dump(pipeline,"outcome_model.pkl")

        st.info("Outcome model saved as outcome_model.pkl")

    else:

        st.error("Outcome column not found in dataset")


    # -------------------------------
    # DURATION MODEL TRAINING
    # -------------------------------

    
    duration_col = None

    for col in df.columns:
        if col.lower().strip() == "case_duration":
            duration_col = col
            break
    

    if duration_col:

        df2 = df.dropna(subset=[duration_col])

        X = df2["text_data"]
        y = df2[duration_col]

        X_train,X_test,y_train,y_test = train_test_split(
            X,y,test_size=0.2,random_state=42
        )

        duration_pipeline = Pipeline([
            ("tfidf",TfidfVectorizer(stop_words="english",max_features=5000)),
            ("regressor",LinearRegression())
        ])

        duration_pipeline.fit(X_train,y_train)

        joblib.dump(duration_pipeline,"duration_model.pkl")

        st.success("Duration prediction model trained!")

    else:

        st.warning("No duration column found in dataset")
# -------------------------------
# ABOUT
# -------------------------------
# -------------------------------
# CHAT HISTORY MEMORY
# -------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------------
# AI LAWYER CHATBOT
# -------------------------------

elif menu == "AI Lawyer Chatbot":

    st.title("⚖️ AI Lawyer Assistant")

    st.write("Ask legal questions or chat with the AI.")

    user_input = st.chat_input("Type your message...")

    if user_input:

        # Save user message
        st.session_state.chat_history.append(("user", user_input))

        user_text = user_input.lower()

        greetings = ["hello","hi","hey","good morning","good evening"]
        thanks = ["thanks","thank you"]
        bye = ["bye","goodbye"]

        # Typing animation
        with st.spinner("AI Lawyer is analyzing case law..."):
            time.sleep(1.5)

            # Greeting response
            if any(g in user_text for g in greetings):

                response = "Hello! 👋 I am your AI Lawyer assistant. You can ask me about legal disputes, case outcomes, or legal procedures."

            # Thanks response
            elif any(t in user_text for t in thanks):

                response = "You're welcome! 😊 Feel free to ask another legal question."

            # Bye response
            elif any(b in user_text for b in bye):

                response = "Goodbye! ⚖️ I wish you the best with your legal matters."

            else:

                # Convert question to vector
                query_vec = vectorizer.transform([user_input])

                similarity = cosine_similarity(query_vec, tfidf_matrix)

                # Get top 3 similar cases
                top_indices = similarity.argsort()[0][-3:][::-1]

                similar_cases = df.iloc[top_indices]

                response = "📚 **Relevant Legal Cases:**\n\n"

                for i,row in similar_cases.iterrows():

                    if summary_col:
                        response += f"**Case Summary:** {row[summary_col]}\n\n"

                    if judgement_col:
                        response += f"**Judgement:** {row[judgement_col]}\n\n"

                response += "⚖️ **AI Insight:** Outcomes depend on evidence strength, legal documentation, and court interpretation."

        # Save AI response
        st.session_state.chat_history.append(("bot", response))

    # Display conversation
    for role, message in st.session_state.chat_history:

        if role == "user":
            st.chat_message("user").write(message)

        else:
            st.chat_message("assistant").write(message)
# -------------------------------
# FOOTER
# -------------------------------

st.markdown("---")
st.caption("Mini Project | Legal Outcome Prediction using AI & NLP")