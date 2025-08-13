# fundamentalATSchecker

An AI-driven **resume parsing & job description matching** web application built with **Flask**, capable of analyzing resumes and job descriptions to generate detailed match scores, skill gap insights, and improvement recommendations.

This tool uses **NLP, semantic similarity, keyword density analysis, ATS scoring**, and **skill categorization** to determine how well a candidate’s resume aligns with a given job description.

---

## 🚀 Features

- **Resume Parsing**  
  - Supports **PDF, DOCX/DOC, TXT** formats.
  - Extracts contact info, experience, education, skills, certifications.
  - Calculates total years of experience, readability score, and keyword density.

- **Job Description Analysis**  
  - Extracts required and preferred skills.
  - Detects experience level, role type, domain, and company size.
  - Identifies critical keywords.

- **AI Matching Engine**  
  - **Semantic similarity** using spaCy.
  - **Keyword matching** with stopword filtering & skill gap detection.
  - **ATS friendliness scoring**.
  - **Cultural fit analysis** based on soft skills.

- **Detailed Reports**  
  - Overall score with weighted scoring breakdown.
  - Matched/missing skills and keywords.
  - Recommendations, strengths, and weaknesses.

- **Performance Optimizations**  
  - Caching system for repeated parsing.
  - Parallel processing for large batches.

---

## 🛠️ Tech Stack

- **Backend**: Flask, Flask-CORS
- **NLP**: spaCy (`en_core_web_lg` / `en_core_web_sm` fallback), NLTK, YAKE, TextBlob, textstat
- **ML Tools**: scikit-learn, NumPy
- **Document Parsing**: pdfplumber, python-docx, pytesseract (OCR support)
- **Data Processing**: Regex, Counter, JSON, Pathlib

---

## 📦 Installation

