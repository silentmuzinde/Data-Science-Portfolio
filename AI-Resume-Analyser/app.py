import os
from flask import Flask, render_template, request, send_file
import pdfplumber
import docx
from werkzeug.utils import secure_filename
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Flask app setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['RESULTS_FOLDER'] = 'results/'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'txt', 'docx'}

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# LangChain setup
llm = ChatGroq(
    api_key="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",  # Replace with your actual API key
    model="llama-3.3-70b-versatile",
    temperature=0.0
)

# LangChain prompt template
cv_prompt = PromptTemplate(
    input_variables=["cv_text", "job_description"],
    template="""
You are an AI assistant helping to analyze a CV against a job description. Review the CV text below and the job description, and provide insights, including:
- Comment on the candidate suitability.
- Score the candidate against required skills and give overal score. Format: Skill: Score

CV Text:
{cv_text}

Job Description:
{job_description}
"""
)

cv_chain = LLMChain(llm=llm, prompt=cv_prompt)

# File validation
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Text extraction
def extract_text_from_file(file_path):
    ext = file_path.rsplit('.', 1)[1].lower()
    if ext == 'pdf':
        with pdfplumber.open(file_path) as pdf:
            return ''.join([page.extract_text() for page in pdf.pages if page.extract_text()])
    elif ext == 'docx':
        doc = docx.Document(file_path)
        return ' '.join([para.text for para in doc.paragraphs])
    elif ext == 'txt':
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    return None

# Analyze CV
def analyze_cv(cv_text, job_description):
    response = cv_chain.run({"cv_text": cv_text, "job_description": job_description})
    return response.strip()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyse', methods=['POST'])
def generate_analysis():
    if 'file' not in request.files:
        return "No file uploaded."

    file = request.files['file']
    job_description = request.form['job_description']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        cv_text = extract_text_from_file(file_path)
        if cv_text:
            analysis_result = analyze_cv(cv_text, job_description)

            # Save analysis
            analysis_file_name = f"cv_analysis_{filename}.txt"
            analysis_file_path = os.path.join(app.config['RESULTS_FOLDER'], analysis_file_name)
            with open(analysis_file_path, 'w', encoding='utf-8') as f:
                f.write(analysis_result)

            # Display result
            return render_template('results.html', analysis=analysis_result, txt_filename=analysis_file_name)

    return "Invalid file format or upload error."

@app.route('/download/<filename>')
def download_file(filename):
    path = os.path.join(app.config['RESULTS_FOLDER'], filename)
    return send_file(path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
