# Real-Time Gen AI Customer Service Bot

A complete end-to-end GenAI customer service chatbot project with:
- Retrieval-based customer support
- Dynamic knowledge base expansion
- Multimodal image + text support
- Medical Q&A assistant
- Research paper expert chatbot
- Sentiment-aware responses
- Multilingual support
- Streamlit UI

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Optional environment variables
- `GEMINI_API_KEY` for Gemini responses and multimodal analysis
- `PRIMARY_LANGUAGE` default output language (`en`, `hi`, `es`, `fr`)

## Project structure
- `app/` main logic
- `data/` sample knowledge datasets
- `reports/` daily and final reports
- `scripts/` helper scripts

## Notes
This project works in offline/demo mode too. If external APIs are not configured, it falls back to deterministic retrieval and local NLP methods.
