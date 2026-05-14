# Real-Time Gen AI Customer Service Bot — Project Report

## 1. Abstract
This project presents a real-time Gen AI customer service bot that combines retrieval-based NLP, optional Gemini-based generation, sentiment analysis, multilingual support, multimodal image understanding, and domain-specific assistants for medical and research queries.

## 2. Introduction
Customer service automation has evolved from simple rule-based bots to intelligent systems that understand natural language, retrieve knowledge, detect sentiment, and respond across multiple languages and modalities.

## 3. Objectives
- Build a real-time customer support bot
- Implement dynamic knowledge updates
- Support image + text input
- Create specialized medical and research assistants
- Detect sentiment and adapt tone
- Support multiple languages

## 4. Technologies Used
- Python
- Streamlit
- scikit-learn
- TF-IDF
- Word2Vec concept module
- LangDetect
- Deep Translator
- PIL
- Transformers
- Gemini API (optional)
- Local vector storage

## 5. System Architecture
User Input -> Language Detection -> Sentiment Analysis -> Retrieval from Vector Store -> Optional LLM Generation -> Translation/Response Formatting -> UI Output

## 6. Module Description
### 6.1 Customer Service Bot
Retrieves answers from a support knowledge base and optionally generates responses through Gemini.

### 6.2 Dynamic Knowledge Expansion
New documents can be ingested into the vector store and the index can be rebuilt to keep answers current.

### 6.3 Multimodal Chatbot
Supports image upload and image understanding. Gemini is used when available; otherwise the system returns local image metadata and instructions.

### 6.4 Medical Q&A Chatbot
Uses MedQuAD-style question-answer retrieval and recognizes symptoms, diseases, and treatments.

### 6.5 Research Expert Chatbot
Retrieves arXiv-style paper abstracts and produces extractive summaries and concept hints.

### 6.6 Sentiment-Aware Chatbot
Classifies messages as positive, negative, or neutral and changes tone accordingly.

### 6.7 Multilingual Chatbot
Detects language automatically and translates responses into supported languages.

## 7. Task Implementation Summary
1. Dynamic knowledge base expansion implemented through `update_knowledge_base()`.
2. Multimodal chatbot implemented through `analyze_image()` and optional Gemini integration.
3. Medical Q&A bot implemented in `MedicalQABot`.
4. Research expert bot implemented in `ResearchExpertBot`.
5. Sentiment analysis implemented in `detect_sentiment()`.
6. Multilingual support implemented in `detect_language()` and `respond_in_language()`.

## 8. Results
The completed bot can:
- Answer support questions in real time
- Update its knowledge base
- Handle images
- Answer medical FAQs
- Summarize technical papers
- Respond with sentiment-aware tone
- Work across multiple languages

## 9. Conclusion
The project successfully demonstrates a practical Gen AI customer service assistant that can be extended for enterprise support, healthcare, research assistance, and multilingual service environments.

## 10. Future Scope
- Voice support
- Better long-term memory
- Full document ingestion pipeline
- Deployment to cloud
- Stronger medical safety guardrails
- More language models and domain adapters
