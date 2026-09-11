import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import google.generativeai as genai

st.set_page_config(page_title="AI Sales Analytics Dashboard", layout="wide")

EMBED_MODEL = "models/embedding-001"
CHAT_MODEL = "gemini-1.5-flash"

st.title("📊 AI-Powered Sales Analytics Dashboard with RAG Chatbot")
st.caption("Upload sales data, explore auto-generated insights, and ask natural-language questions answered via Retrieval-Augmented Generation.")

# Reads GEMINI_API_KEY from the environment (e.g. a Render env var) if set,
# otherwise falls back to whatever the user types in the sidebar.
env_key = os.environ.get("GEMINI_API_KEY", "")
if env_key:
    st.sidebar.success("Gemini API key loaded from environment ✅")
    api_key = env_key
else:
    api_key = st.sidebar.text_input(
        "Gemini API Key", type="password",
        help="Get one free at https://aistudio.google.com/apikey"
    )
if api_key:
    genai.configure(api_key=api_key)

uploaded_file = st.sidebar.file_uploader("Upload sales CSV/Excel", type=["csv", "xlsx"])


@st.cache_data(show_spinner=False)
def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)


def embed_text(text, task_type="retrieval_document"):
    result = genai.embed_content(model=EMBED_MODEL, content=text, task_type=task_type)
    return np.array(result["embedding"])


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def build_knowledge_base(df):
    docs = [f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns. Columns: {list(df.columns)}"]
    num_df = df.select_dtypes(include=np.number)
    for col in num_df.columns:
        docs.append(
            f"Column '{col}' stats: mean={num_df[col].mean():.2f}, "
            f"max={num_df[col].max():.2f}, min={num_df[col].min():.2f}, sum={num_df[col].sum():.2f}"
        )
    cat_df = df.select_dtypes(exclude=np.number)
    for col in cat_df.columns:
        top = df[col].value_counts().head(5)
        docs.append(f"Top values in '{col}': " + ", ".join(f"{k} ({v})" for k, v in top.items()))
    for i in range(0, min(len(df), 50), 10):
        chunk = df.iloc[i:i + 10].to_string()
        docs.append(f"Sample rows {i}-{i+10}:\n{chunk}")
    return docs


if uploaded_file and api_key:
    df = load_data(uploaded_file)

    st.subheader("Data Preview")
    st.dataframe(df.head())

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    c1, c2, c3 = st.columns(3)
    if num_cols:
        c1.metric(f"Total {num_cols[0]}", f"{df[num_cols[0]].sum():,.2f}")
        c2.metric(f"Average {num_cols[0]}", f"{df[num_cols[0]].mean():,.2f}")
    c3.metric("Total Rows", len(df))

    st.subheader("Exploratory Data Analysis")
    if num_cols:
        chosen = st.selectbox("Numeric column to visualize", num_cols)
        st.plotly_chart(px.histogram(df, x=chosen, title=f"Distribution of {chosen}"), use_container_width=True)

    if len(num_cols) >= 2:
        st.plotly_chart(px.imshow(df[num_cols].corr(), text_auto=True, title="Correlation Heatmap"),
                         use_container_width=True)

    date_cols = [c for c in df.columns if "date" in c.lower()]
    if date_cols and num_cols:
        dcol = date_cols[0]
        try:
            df[dcol] = pd.to_datetime(df[dcol])
            trend = df.groupby(df[dcol].dt.to_period("M"))[num_cols[0]].sum().reset_index()
            trend[dcol] = trend[dcol].astype(str)
            st.plotly_chart(px.line(trend, x=dcol, y=num_cols[0], title=f"{num_cols[0]} Trend Over Time"),
                             use_container_width=True)
        except Exception:
            pass

    st.subheader("💬 Ask questions about your data (RAG)")
    if "kb_embeddings" not in st.session_state or st.session_state.get("kb_file") != uploaded_file.name:
        with st.spinner("Indexing your data..."):
            docs = build_knowledge_base(df)
            embeddings = [embed_text(d) for d in docs]
            st.session_state["kb_docs"] = docs
            st.session_state["kb_embeddings"] = embeddings
            st.session_state["kb_file"] = uploaded_file.name

    question = st.text_input("Ask a question about your sales data")
    if question:
        q_emb = embed_text(question, task_type="retrieval_query")
        sims = [cosine_sim(q_emb, e) for e in st.session_state["kb_embeddings"]]
        top_idx = np.argsort(sims)[-4:][::-1]
        context = "\n\n".join(st.session_state["kb_docs"][i] for i in top_idx)
        model = genai.GenerativeModel(CHAT_MODEL)
        prompt = (
            "You are a data analyst assistant. Use the context to answer the question precisely, "
            f"citing numbers where relevant.\n\nContext:\n{context}\n\nQuestion: {question}"
        )
        with st.spinner("Thinking..."):
            response = model.generate_content(prompt)
        st.markdown("**Answer:**")
        st.write(response.text)
else:
    st.info("Enter your Gemini API key and upload a CSV/Excel file to get started.")
