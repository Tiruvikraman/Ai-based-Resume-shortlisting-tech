from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from docx import Document
import re
import traceback
import pandas as pd

app = Flask(__name__)

# Setup resume folder
resume_folder = os.path.join('static', 'Resumes')
if not os.path.exists(resume_folder):
    os.makedirs(resume_folder)
    print(f"Created directory: {resume_folder}")

# Load a smaller model for testing
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Function to read text from a docx file
def read_docx(file_path):
    doc = Document(file_path)
    return " ".join([paragraph.text for paragraph in doc.paragraphs])

# Function to preprocess text
def preprocess_text(text):
    # Remove unnecessary special characters that might confuse the regex
    text = re.sub(r'[^a-zA-Z0-9@.\s+-]', '', text)
    # Normalize multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text

# Function to extract years of experience from text
def extract_experience(text):
    experience_patterns = [
        r'(\d+)\s*(?:years?|yrs?)\s*(?:of)?\s*experience',
        r'experience\s*(?:of)?\s*(\d+)\s*(?:years?|yrs?)',
    ]
    for pattern in experience_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return 0

# Function to extract email
def extract_email(text):
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}'
    match = re.search(email_pattern, text)
    return match.group(0) if match else "No email found"

# Function to extract contact number
def extract_contact(text):
    contact_patterns = [
        r'(\+?\d{1,4}[\s-]?)?(?:\(\d{1,4}\)[\s-]?)?\d{1,4}[\s-]?\d{1,4}[\s-]?\d{4,10}',  # general format
        r'\b\d{10}\b',  # 10 digit format
        r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'  # US format with dashes, dots, or spaces
    ]
    for pattern in contact_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    return "No contact found"

# Function to get resume score based on job description
def get_resume_score(resume_text, job_description, required_experience):
    prompt = f"""
    Job Description: {job_description}
    Required Experience: {required_experience} years
    
    Resume:
    {resume_text[:1000]}  # Truncate resume text to fit within model's context window
    
    Based on the job description and required experience, rate how well this resume matches on a scale of 0 to 10.
    Provide only the numeric score without any additional text.
    """
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=10, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"Raw model response: {response}")  # Debug print
    
    # Try to extract a number from the response
    match = re.search(r'\d+(?:\.\d+)?', response)
    if match:
        score = float(match.group())
        return min(max(score, 0), 10)  # Ensure score is between 0 and 10
    else:
        print(f"Failed to extract score from model response: {response}")
        return 0  # Default to 0 if we can't parse a numeric score

# Flask server starts here!
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/get_resumes', methods=['POST'])
def get_resumes():
    try:
        data = request.json
        job_description = data['job_description']
        required_experience = int(data['work_experience'])
        
        print(f"Job Description: {job_description}")
        print(f"Required Experience: {required_experience}")
        print(f"Files in resume folder: {os.listdir(resume_folder)}")
        
        matching_resumes = []
        
        for filename in os.listdir(resume_folder):
            if filename.endswith('.docx'):
                file_path = os.path.join(resume_folder, filename)  # Define file_path here
                resume_text = preprocess_text(read_docx(file_path))  # Use file_path after defining it
                experience = extract_experience(resume_text)
                email = extract_email(resume_text)
                contact = extract_contact(resume_text)
                
                print(f"Extracted experience for {filename}: {experience}")
                
                score = get_resume_score(resume_text, job_description, required_experience)
                print(f"Score for {filename}: {score}")
                
                if experience >= required_experience:
                    name = os.path.splitext(filename)[0]
                    matching_resumes.append({
                        'name': name,
                        'email': email,
                        'contact': contact,
                        'experience': experience,
                        'resume_link': f'/static/Resumes/{filename}',
                        'score': score
                    })
        
        # Sort resumes by score in descending order
        matching_resumes.sort(key=lambda x: x['score'], reverse=True)

        # Add rank to resumes
        for i, resume in enumerate(matching_resumes, 1):
            resume['rank'] = i
        
        # Convert to DataFrame and save to CSV
        df = pd.DataFrame(matching_resumes)
        df.to_csv('matching_resumes.csv', index=False)
        
        # Return all matching resumes for debugging
        print(f"Number of matching resumes: {len(matching_resumes)}")
        print(f"Matching resumes: {matching_resumes}")
        
        return jsonify({'resumes': matching_resumes})
    except Exception as e:
        error_message = f"An error occurred: {str(e)}\n{traceback.format_exc()}"
        print(error_message)
        return jsonify({'error': error_message}), 500
    
if __name__ == '__main__':
    app.run(debug=True)
