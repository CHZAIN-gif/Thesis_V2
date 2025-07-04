# The Final, Complete, and Polished Master Code for Thesis

# --- 1. ALL IMPORTS ---
import streamlit as st
import sqlite3
import bcrypt
import os
import uuid
import json
import io
import numpy as np
import faiss
import pdfplumber
import google.generativeai as genai
from gtts import gTTS
from PIL import Image
import pytesseract
import graphviz

# --- 2. CONFIGURATION ---
st.set_page_config(page_title="Thesis", layout="wide", initial_sidebar_state="auto")

# Tell pytesseract where to find the Tesseract program on the RDP server
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

# Configure the Gemini API key
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except (KeyError, AttributeError):
    pass # We will check for the key when it's used

DATABASE_NAME = 'thesis_database.db'

# --- 3. DATABASE FUNCTIONS ---
@st.cache_resource
def get_db_connection():
    conn = sqlite3.connect(DATABASE_NAME, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def initialize_database():
    conn = get_db_connection()
    with conn:
        conn.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT UNIQUE, password_hash TEXT)')
        conn.execute('CREATE TABLE IF NOT EXISTS documents (id INTEGER PRIMARY KEY, user_id INTEGER, filename TEXT, path TEXT UNIQUE, faiss_index BLOB, chunks_json TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)')
        conn.execute('CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY, doc_id INTEGER, role TEXT, content TEXT, timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)')

# --- 4. AUTH & FILE HELPER FUNCTIONS ---
def hash_password(p): return bcrypt.hashpw(p.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
def verify_password(p, h): return bcrypt.checkpw(p.encode('utf-8'), h.encode('utf-8'))

def save_uploaded_file(file):
    DIR = "user_uploads"; os.makedirs(DIR, exist_ok=True)
    path = os.path.join(DIR, f"{uuid.uuid4().hex}-{file.name}")
    with open(path, "wb") as f: f.write(file.getbuffer())
    return path

def save_text_to_cache(doc_id, text_content):
    DIR = "text_cache"; os.makedirs(DIR, exist_ok=True)
    cache_path = os.path.join(DIR, f"{doc_id}.txt")
    with open(cache_path, "w", encoding="utf-8") as f: f.write(text_content)

def load_text_from_cache(doc_id):
    path = os.path.join("text_cache", f"{doc_id}.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f: return f.read()
    return None

# --- 5. AI CORE FUNCTIONS ---
def ai_extract_text(path):
    text = ""
    try:
        with pdfplumber.open(path) as pdf:
            for p in pdf.pages: text += p.extract_text(x_tolerance=2, layout=True) or ""
        if len(text.strip()) < 200:
            text = ""
            with pdfplumber.open(path) as pdf:
                for p in pdf.pages: text += pytesseract.image_to_string(p.to_image(resolution=300).original)
    except Exception as e: return f"Error reading PDF: {e}"
    return text if text.strip() else None

def ai_split_chunks(text): return [text[i:i + 1500] for i in range(0, len(text), 1300)]

def ai_create_embeddings(chunks):
    if not chunks: return None
    try:
        result = genai.embed_content(model="models/embedding-001", content=chunks, task_type="retrieval_document")
        index = faiss.IndexFlatL2(len(result['embedding'][0]))
        index.add(np.array(result['embedding']).astype('float32'))
        with io.BytesIO() as bio: faiss.write_index(index, faiss.PyCallbackIOWriter(bio.write)); return bio.getvalue()
    except Exception: return None

def ai_get_chat_response(faiss_index, question, chunks):
    try:
        index = faiss.read_index(faiss.PyCallbackIOReader(io.BytesIO(faiss_index).read))
        q_embedding = genai.embed_content(model='models/embedding-001', content=question, task_type="retrieval_query")['embedding']
        _, indices = index.search(np.array([q_embedding]).astype('float32'), k=5)
        context = "".join([chunks[i] for i in indices[0] if i < len(chunks)])
        prompt = f'Answer based ONLY on the context. If not in context, say "I could not find the answer in the document." CONTEXT: {context} QUESTION: {question}'
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        return model.generate_content(prompt).text
    except Exception as e: return f"An error occurred: {e}"

def ai_generate_mind_map(text):
    prompt = f"Create a hierarchical mind map of this text in Graphviz DOT language. Use concise labels. Root node should be the main topic. TEXT: {text[:20000]}"
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(prompt)
        return response.text.strip().replace("```dot", "").replace("```", "")
    except Exception as e: return f'digraph G {{ error [label="Could not generate mind map"]; }}'

# --- 6. PAGE RENDERING FUNCTIONS ---
def render_login_page():
    st.title("Welcome to Thesis 🧠")
    c1,c2 = st.columns(2)
    conn = get_db_connection()
    with c1:
        with st.form("signup"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Sign Up"):
                if username and password:
                    try:
                        conn.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, hash_password(password)))
                        conn.commit()
                        st.success("Account created!")
                    except sqlite3.IntegrityError: st.error("Username taken.")
                else: st.warning("Fill all fields.")
    with c2:
        with st.form("login"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Log In"):
                user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
                if user and verify_password(password, user['password_hash']):
                    st.session_state.logged_in = True
                    st.session_state.user_id = user['id']
                    st.session_state.username = user['username']
                    st.session_state.page = 'dashboard'
                    st.rerun()
                else: st.error("Invalid credentials.")

def render_dashboard():
    st.title(f"My Documents 📚")
    if st.button("＋ Upload New Document", type="primary"): st.session_state.page = 'upload'; st.rerun()
    st.write("---")
    docs = get_db_connection().execute("SELECT * FROM documents WHERE user_id = ?", (st.session_state.user_id,)).fetchall()
    if not docs: st.info("You have no documents. Upload one to get started!")
    else:
        for doc in docs:
            with st.expander(f"**{doc['filename']}**"):
                c1, c2 = st.columns(2)
                if c1.button("Chat 💬", key=f"c_{doc['id']}", use_container_width=True):
                    st.session_state.page = 'chat'; st.session_state.doc_id = doc['id']; st.rerun()
                if c2.button("Mind Map 🗺️", key=f"m_{doc['id']}", use_container_width=True):
                    st.session_state.page = 'mind_map'; st.session_state.doc_id = doc['id']; st.rerun()

def render_upload_page():
    st.title("Upload a New Document 📄")
    uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
    if uploaded_file:
        with st.spinner('Processing... This may take time for scanned PDFs.'):
            path = save_uploaded_file(uploaded_file)
            text = ai_extract_text(path)
            if text:
                chunks = ai_split_chunks(text)
                faiss_index = ai_create_embeddings(chunks)
                if faiss_index:
                    conn = get_db_connection()
                    doc_id = conn.execute("INSERT INTO documents (user_id, filename, path, faiss_index, chunks_json) VALUES (?, ?, ?, ?, ?)", (st.session_state.user_id, uploaded_file.name, path, faiss_index, json.dumps(chunks))).lastrowid
                    conn.commit()
                    save_text_to_cache(doc_id, text)
                    st.success("Document processed successfully!")
                    st.session_state.page = 'dashboard'; st.rerun()
                else: st.error("Failed to create AI memory.")
            else: st.error("Could not extract any text.")

def render_chat_page():
    conn = get_db_connection()
    doc = conn.execute("SELECT * FROM documents WHERE id = ?", (st.session_state.doc_id,)).fetchone()
    st.title(f"Chat with: *{doc['filename']}* 💬")

    if f"messages_{st.session_state.doc_id}" not in st.session_state: st.session_state[f"messages_{st.session_state.doc_id}"] = []

    for msg in st.session_state[f"messages_{st.session_state.doc_id}"]:
        with st.chat_message(msg['role']): st.markdown(msg['content'])

    if prompt := st.chat_input("Ask a question..."):
        st.session_state[f"messages_{st.session_state.doc_id}"].append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = ai_get_chat_response(doc['faiss_index'], prompt, json.loads(doc['chunks_json']))
                st.markdown(response)
        st.session_state[f"messages_{st.session_state.doc_id}"].append({"role": "assistant", "content": response})

def render_mind_map_page():
    conn = get_db_connection()
    doc = conn.execute("SELECT * FROM documents WHERE id = ?", (st.session_state.doc_id,)).fetchone()
    st.title(f"Visual Mind Map for: *{doc['filename']}* 🗺️")
    text = load_text_from_cache(st.session_state.doc_id)
    if text:
        with st.spinner("AI is creating the mind map..."):
            dot_code = ai_generate_mind_map(text)
            st.graphviz_chart(dot_code)
    else: st.error("Could not load text for mind map.")

# --- 7. MAIN APP ROUTER ---
initialize_database()

if 'logged_in' not in st.session_state: st.session_state.logged_in = False
if 'page' not in st.session_state: st.session_state.page = 'login'

if not st.session_state.logged_in:
    render_login_page()
else:
    PAGES = {"dashboard": render_dashboard, "upload": render_upload_page, "chat": render_chat_page, "mind_map": render_mind_map_page}

    st.sidebar.title("Thesis")
    st.sidebar.success(f"Logged in as **{st.session_state.username}**")
    if st.sidebar.button("My Documents 📚", use_container_width=True): st.session_state.page = 'dashboard'; st.rerun()
    if st.sidebar.button("Upload Document 📄", use_container_width=True): st.session_state.page = 'upload'; st.rerun()
    st.sidebar.write("---")
    if st.sidebar.button("Log Out", use_container_width=True):
        for key in list(st.session_state.keys()): del st.session_state[key]
        st.rerun()

    page_function = PAGES.get(st.session_state.page, render_dashboard)
    page_function()