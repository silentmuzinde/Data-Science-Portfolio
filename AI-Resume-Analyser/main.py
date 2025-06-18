import os
import pdfplumber
import docx
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Paths
CV_FILE = "Silent Muzinde.docx" 
OUTPUT_FOLDER = "results"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# LangChain setup
llm = ChatGroq(
    api_key="YOUR_API_KEY_HERE",
    model="llama-3.3-70b-versatile",
    temperature=0.0
)

# Prompt template for CV analysis
cv_prompt = PromptTemplate(
    input_variables=["cv_text"],
    template="""
You are an AI assistant helping to analyze a CV. Review the text below and provide insights, including:
- Key skills and qualifications
- Overall strengths and weaknesses
- Suggestions for improvement

CV Text:
{cv_text}
"""
)

cv_chain = LLMChain(llm=llm, prompt=cv_prompt)

# Text extraction function
def extract_text(file_path):
    ext = file_path.rsplit('.', 1)[-1].lower()
    if ext == "pdf":
        with pdfplumber.open(file_path) as pdf:
            return ''.join([p.extract_text() for p in pdf.pages if p.extract_text()])
    elif ext == "docx":
        doc = docx.Document(file_path)
        return ' '.join([para.text for para in doc.paragraphs])
    elif ext == "txt":
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        raise ValueError("Unsupported file type")
    
# Run process
def main():
    try:
        cv_text = extract_text(CV_FILE)
    except Exception as e:
        print(f"Error extracting text: {e}")
        return

    if not cv_text:
        print("No text extracted from CV.")
        return

    print("Analyzing CV...")
    analysis_result = cv_chain.run({"cv_text": cv_text}).strip()

    # Save the analysis result
    analysis_file_path = os.path.join(OUTPUT_FOLDER, "cv_analysis.txt")
    with open(analysis_file_path, 'w', encoding='utf-8') as f:
        f.write(analysis_result)
    
    print(f"Analysis completed. Results saved to {analysis_file_path}")

if __name__ == "__main__":
    main()