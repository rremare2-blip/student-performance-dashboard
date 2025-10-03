# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="Student Performance Dashboard", layout="wide")

# ---------- Raw dataset (embedded) ----------
RAW_DATA = [
    {"Student_ID":101,"Age":18,"Gender":"Female","Attendance (%)":100,"Quiz_Avg":95,"Project_Score":59,"Final_Exam":79,"Final_Grade":"C"},
    {"Student_ID":102,"Age":18,"Gender":"Female","Attendance (%)":83,"Quiz_Avg":70,"Project_Score":90,"Final_Exam":50,"Final_Grade":"C"},
    {"Student_ID":103,"Age":20,"Gender":"Male","Attendance (%)":70,"Quiz_Avg":63,"Project_Score":60,"Final_Exam":96,"Final_Grade":"C"},
    {"Student_ID":104,"Age":19,"Gender":"Male","Attendance (%)":83,"Quiz_Avg":91,"Project_Score":100,"Final_Exam":96,"Final_Grade":"A"},
    {"Student_ID":105,"Age":19,"Gender":"Female","Attendance (%)":82,"Quiz_Avg":81,"Project_Score":93,"Final_Exam":66,"Final_Grade":"B"},
    {"Student_ID":106,"Age":19,"Gender":"Male","Attendance (%)":73,"Quiz_Avg":75,"Project_Score":77,"Final_Exam":82,"Final_Grade":"C"},
    {"Student_ID":107,"Age":18,"Gender":"Male","Attendance (%)":77,"Quiz_Avg":91,"Project_Score":88,"Final_Exam":98,"Final_Grade":"A"},
    {"Student_ID":108,"Age":22,"Gender":"Female","Attendance (%)":64,"Quiz_Avg":79,"Project_Score":54,"Final_Exam":61,"Final_Grade":"D"},
    {"Student_ID":109,"Age":18,"Gender":"Male","Attendance (%)":98,"Quiz_Avg":59,"Project_Score":74,"Final_Exam":82,"Final_Grade":"C"},
    {"Student_ID":110,"Age":22,"Gender":"Female","Attendance (%)":100,"Quiz_Avg":66,"Project_Score":74,"Final_Exam":56,"Final_Grade":"D"},
    {"Student_ID":111,"Age":21,"Gender":"Female","Attendance (%)":70,"Quiz_Avg":58,"Project_Score":88,"Final_Exam":90,"Final_Grade":"C"},
    {"Student_ID":112,"Age":18,"Gender":"Female","Attendance (%)":94,"Quiz_Avg":65,"Project_Score":79,"Final_Exam":69,"Final_Grade":"C"},
    {"Student_ID":113,"Age":18,"Gender":"Male","Attendance (%)":75,"Quiz_Avg":97,"Project_Score":83,"Final_Exam":90,"Final_Grade":"A"},
    {"Student_ID":114,"Age":18,"Gender":"Female","Attendance (%)":70,"Quiz_Avg":85,"Project_Score":66,"Final_Exam":82,"Final_Grade":"C"},
    {"Student_ID":115,"Age":19,"Gender":"Male","Attendance (%)":89,"Quiz_Avg":84,"Project_Score":85,"Final_Exam":88,"Final_Grade":"B"},
    {"Student_ID":116,"Age":19,"Gender":"Female","Attendance (%)":84,"Quiz_Avg":66,"Project_Score":50,"Final_Exam":62,"Final_Grade":"F"},
    {"Student_ID":117,"Age":22,"Gender":"Male","Attendance (%)":77,"Quiz_Avg":97,"Project_Score":93,"Final_Exam":59,"Final_Grade":"B"},
    {"Student_ID":118,"Age":22,"Gender":"Female","Attendance (%)":100,"Quiz_Avg":87,"Project_Score":96,"Final_Exam":73,"Final_Grade":"B"},
    {"Student_ID":119,"Age":18,"Gender":"Female","Attendance (%)":95,"Quiz_Avg":77,"Project_Score":57,"Final_Exam":98,"Final_Grade":"C"},
    {"Student_ID":120,"Age":22,"Gender":"Male","Attendance (%)":74,"Quiz_Avg":87,"Project_Score":93,"Final_Exam":60,"Final_Grade":"B"},
    {"Student_ID":121,"Age":19,"Gender":"Male","Attendance (%)":80,"Quiz_Avg":75,"Project_Score":84,"Final_Exam":84,"Final_Grade":"B"},
    {"Student_ID":122,"Age":22,"Gender":"Male","Attendance (%)":63,"Quiz_Avg":73,"Project_Score":98,"Final_Exam":99,"Final_Grade":"A"},
    {"Student_ID":123,"Age":21,"Gender":"Male","Attendance (%)":74,"Quiz_Avg":64,"Project_Score":67,"Final_Exam":83,"Final_Grade":"C"},
    {"Student_ID":124,"Age":19,"Gender":"Female","Attendance (%)":62,"Quiz_Avg":58,"Project_Score":99,"Final_Exam":50,"Final_Grade":"D"},
    {"Student_ID":125,"Age":21,"Gender":"Male","Attendance (%)":80,"Quiz_Avg":82,"Project_Score":91,"Final_Exam":88,"Final_Grade":"B"},
    {"Student_ID":126,"Age":22,"Gender":"Male","Attendance (%)":85,"Quiz_Avg":81,"Project_Score":71,"Final_Exam":70,"Final_Grade":"C"},
    {"Student_ID":127,"Age":20,"Gender":"Male","Attendance (%)":77,"Quiz_Avg":55,"Project_Score":57,"Final_Exam":81,"Final_Grade":"D"},
    {"Student_ID":128,"Age":18,"Gender":"Female","Attendance (%)":64,"Quiz_Avg":98,"Project_Score":68,"Final_Exam":51,"Final_Grade":"C"},
    {"Student_ID":129,"Age":19,"Gender":"Female","Attendance (%)":73,"Quiz_Avg":53,"Project_Score":77,"Final_Exam":57,"Final_Grade":"D"},
    {"Student_ID":130,"Age":21,"Gender":"Female","Attendance (%)":96,"Quiz_Avg":57,"Project_Score":60,"Final_Exam":73,"Final_Grade":"D"},
]

# ---------- Helper functions ----------
def load_data():
    df = pd.DataFrame(RAW_DATA)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    # Standardize gender
    df['Gender'] = df['Gender'].astype(str).str.strip().str.title().replace({'M':'Male','F':'Female'})

    # Replace invalid numeric entries and ensure numeric dtype
    numeric_cols = ['Age','Attendance (%)','Quiz_Avg','Project_Score','Final_Exam']
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        # Handle negative values: replace with NaN (then impute)
        df.loc[df[c] < 0, c] = np.nan

    # Handle missing values:
    # For simplicity: drop rows if Student_ID missing; impute numeric with column median
    df = df.dropna(subset=['Student_ID'])
    for c in numeric_cols:
        median_val = df[c].median()
        df[c] = df[c].fillna(median_val)

    # Make sure Final_Grade exists; otherwise compute from Final_Exam
    if 'Final_Grade' not in df.columns or df['Final_Grade'].isnull().any():
        df['Final_Grade'] = df['Final_Exam'].apply(score_to_grade)

    # Add binary pass column (Final_Exam >= 60 -> pass)
    df['Pass'] = (df['Final_Exam'] >= 60).astype(int)

    # Round numeric columns where sensible
    df['Attendance (%)'] = df['Attendance (%)'].round(1)
    df['Quiz_Avg'] = df['Quiz_Avg'].round(1)
    df['Project_Score'] = df['Project_Score'].round(1)
    df['Final_Exam'] = df['Final_Exam'].round(1)
    return df

def score_to_grade(score):
    try:
        s = float(score)
    except:
        return 'C'
    if s >= 90:
        return 'A'
    if s >= 80:
        return 'B'
    if s >= 70:
        return 'C'
    if s >= 60:
        return 'D'
    return 'F'

def descriptive_stats(df: pd.DataFrame, cols):
    stats = {}
    for c in cols:
        arr = df[c].values
        stats[c] = {
            'mean': np.mean(arr),
            'median': np.median(arr),
            'std': np.std(arr, ddof=1),
            'min': np.min(arr),
            'max': np.max(arr)
        }
    return stats

def pearson_correlation(x, y):
    # returns r and p-value approx using numpy (no p-value). We'll compute r only.
    r = np.corrcoef(x, y)[0,1]
    return r

# Static rule-based recommendation engine
def rule_recommendations(row):
    recs = []
    # Rule 1: attendance threshold
    if row['Attendance (%)'] < 75:
        recs.append("High risk (Attendance < 75%). Recommend immediate intervention & weekly check-ins.")
    # Rule 2: quiz high, final low => exam support
    if row['Quiz_Avg'] > 85 and row['Final_Exam'] < 70:
        recs.append("Quiz high but final low: suggest exam skills workshops (time management, test-taking).")
    # Rule 3: project high, final low
    if row['Project_Score'] > 90 and row['Final_Exam'] < 70:
        recs.append("Strong project skills but low exam: consider alternative assessments or oral exams.")
    # Rule 4: excellent overall
    if row['Final_Grade'] == 'A':
        recs.append("Consider for peer-mentor or recognition program.")
    # If none, generic
    if not recs:
        recs.append("No immediate red flags. Continue monitoring.")
    return recs

# ---------- Load & clean ----------
df_raw = load_data()
df = clean_data(df_raw)

# ---------- Sidebar / Navigation ----------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Statistics", "Top/Bottom Performers", "Predictor", "Recommendations", "Download Data"])

# ---------- Overview Page ----------
if page == "Overview":
    st.title("📊 Student Performance Dashboard — Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Students", int(df.shape[0]))
    col2.metric("Average Final Exam", f"{df['Final_Exam'].mean():.1f}")
    col3.metric("Pass Rate", f"{df['Pass'].mean()*100:.0f}%")
    col4.metric("Average Attendance", f"{df['Attendance (%)'].mean():.1f}%")

    st.markdown("### Grade Distribution")
    grade_counts = df['Final_Grade'].value_counts().sort_index()
    fig_grade = px.bar(x=grade_counts.index, y=grade_counts.values, labels={'x':'Grade','y':'Count'},
                       title="Final Grade Distribution")
    st.plotly_chart(fig_grade, use_container_width=True)

    st.markdown("### Attendance vs Final Exam Score (scatter)")
    # scatter with regression line (simple linear fit)
    x = df['Attendance (%)'].values
    y = df['Final_Exam'].values
    coeffs = np.polyfit(x, y, 1)
    fit_y = np.polyval(coeffs, x)
    scatter = go.Figure()
    scatter.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Students',
                                 marker=dict(size=8)))
    scatter.add_trace(go.Scatter(x=x, y=fit_y, mode='lines', name=f'Fit (y={coeffs[0]:.2f}x+{coeffs[1]:.2f})'))
    scatter.update_layout(xaxis_title='Attendance (%)', yaxis_title='Final Exam Score', title='Attendance vs Final Exam')
    st.plotly_chart(scatter, use_container_width=True)

    r = pearson_correlation(x, y)
    st.markdown(f"**Pearson correlation (attendance vs final exam):** {r:.2f}")

    st.markdown("### Gender-wise Comparison (Average Final Exam)")
    gender_avg = df.groupby('Gender')['Final_Exam'].mean().reset_index()
    fig_gender = px.bar(gender_avg, x='Gender', y='Final_Exam', labels={'Final_Exam':'Average Final Exam'}, title='Average Final Exam by Gender')
    st.plotly_chart(fig_gender, use_container_width=True)

# ---------- Statistics Page ----------
elif page == "Statistics":
    st.title("🔬 Descriptive Statistics")
    cols = ['Quiz_Avg','Project_Score','Final_Exam']
    stats = descriptive_stats(df, cols)
    st.dataframe(pd.DataFrame(stats).T.style.format("{:.2f}"))
    st.markdown("#### Distribution plots")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Quiz Average Distribution**")
        fig_q = px.histogram(df, x='Quiz_Avg', nbins=15, title='Quiz Average Distribution')
        st.plotly_chart(fig_q, use_container_width=True)
    with c2:
        st.markdown("**Project Score Distribution**")
        fig_p = px.histogram(df, x='Project_Score', nbins=15, title='Project Score Distribution')
        st.plotly_chart(fig_p, use_container_width=True)

    st.markdown("#### Attendance vs Final Exam (Binned averages)")
    df['att_bin'] = pd.cut(df['Attendance (%)'], bins=[0,60,70,80,90,100], labels=['<=60','61-70','71-80','81-90','91-100'])
    binned = df.groupby('att_bin')['Final_Exam'].mean().reset_index()
    fig_bin = px.line(binned, x='att_bin', y='Final_Exam', title='Average Final Exam by Attendance Bin', markers=True)
    st.plotly_chart(fig_bin, use_container_width=True)

# ---------- Top/Bottom Performers ----------
elif page == "Top/Bottom Performers":
    st.title("🏆 Top & Bottom Performers")
    sorted_by_final = df.sort_values('Final_Exam', ascending=False)
    st.subheader("Top 5 Performers")
    st.table(sorted_by_final.head(5)[['Student_ID','Age','Gender','Attendance (%)','Quiz_Avg','Project_Score','Final_Exam','Final_Grade']])

    st.subheader("Bottom 5 Performers")
    st.table(sorted_by_final.tail(5)[['Student_ID','Age','Gender','Attendance (%)','Quiz_Avg','Project_Score','Final_Exam','Final_Grade']])

    st.markdown("### All Students")
    st.dataframe(df[['Student_ID','Age','Gender','Attendance (%)','Quiz_Avg','Project_Score','Final_Exam','Final_Grade']].sort_values('Student_ID'))

# ---------- Predictor Page ----------
elif page == "Predictor":
    st.title("🧠 Pass/Fail Predictor (Logistic Regression)")
    st.markdown("We'll train a simple logistic regression model to predict whether a student will pass (Final_Exam >= 60) using Attendance and Quiz_Avg.")

    features = ['Attendance (%)','Quiz_Avg']
    X = df[features].values
    y = df['Pass'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.markdown(f"**Model accuracy on test set:** {acc:.3f}")

    st.markdown("**Confusion matrix**")
    cm = confusion_matrix(y_test, y_pred)
    st.write(cm)
    st.text(classification_report(y_test, y_pred, target_names=['Fail','Pass']))

    st.markdown("### Quick prediction")
    with st.form("predict_form"):
        st.write("Enter student features:")
        att_in = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=75.0)
        quiz_in = st.number_input("Quiz Average", min_value=0.0, max_value=100.0, value=70.0)
        submitted = st.form_submit_button("Predict")
        if submitted:
            pred = model.predict([[att_in, quiz_in]])[0]
            prob = model.predict_proba([[att_in, quiz_in]])[0][pred]
            st.success(f"Predicted: {'Pass' if pred==1 else 'Fail'} (probability {prob:.2f})")

# ---------- Recommendations Page ----------
elif page == "Recommendations":
    st.title("🎯 Static Rule-based Recommendations")
    st.markdown("Recommendations are produced by a set of static rules applied per student. These are simple actionable suggestions for interventions.")

    # Show summary counts of flagged students by attendance rule
    at_risk = df[df['Attendance (%)'] < 75]
    st.markdown(f"**Students with Attendance < 75%:** {len(at_risk)}")
    if len(at_risk):
        st.table(at_risk[['Student_ID','Attendance (%)','Final_Exam','Final_Grade']])

    # Generate recommendation per student (demo few)
    st.markdown("Sample recommendations for each student (first 20):")
    rec_df = df.copy()
    rec_df['Recommendations'] = rec_df.apply(lambda r: " | ".join(rule_recommendations(r)), axis=1)
    st.dataframe(rec_df[['Student_ID','Attendance (%)','Quiz_Avg','Final_Exam','Final_Grade','Recommendations']].head(20))

    st.markdown("**Top recommended actions (aggregated):**")
    # Aggregate reasons
    reasons = []
    for _, r in df.iterrows():
        reasons.extend(rule_recommendations(r))
    reason_counts = pd.Series(reasons).value_counts().reset_index()
    reason_counts.columns = ['Recommendation','Count']
    st.table(reason_counts)

# ---------- Download Data Page ----------
elif page == "Download Data":
    st.title("📥 Download cleaned dataset")
    st.markdown("You can download the cleaned dataset (CSV) for your report and further analysis.")
    csv = df.to_csv(index=False)
    st.download_button("Download CSV", data=csv, file_name="cleaned_student_data.csv", mime="text/csv")
    st.markdown("#### Quick preview")
    st.dataframe(df.head(50))

# ---------- Footer / Notes ----------
st.markdown("---")
st.caption("Generated with static rules & a simple logistic regression model for educational/demo purposes. Interpret results cautiously; more robust models require larger datasets and cross-validation.")
