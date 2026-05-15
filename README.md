# AI Roast ↔ Motivation Switcher

A beginner-friendly, portfolio-ready GenAI app built with Python and Streamlit. Enter any thought, confession, or problem, choose a personality mode, and the app calls a Groq OpenAI-compatible chat endpoint to generate a response that matches the selected tone.

## Features

- Roast, Motivation, Mentor, Savage, and Mixed modes
- Groq API integration using OpenAI-compatible chat completions
- Streamlit UI with a dark modern aesthetic
- Temperature slider for creativity control
- Conversation memory in `st.session_state`
- Streaming token-by-token responses
- Prompt logging for learning and debugging
- Export chat history to JSON or Markdown
- Beginner-friendly comments and educational notes

## Folder Structure

```text
project/
├── app.py
├── .env
├── requirements.txt
├── README.md
├── prompts/
│   ├── roast.txt
│   ├── motivation.txt
│   ├── mentor.txt
│   ├── savage.txt
│   └── mixed.txt
├── utils/
│   ├── api_client.py
│   ├── prompt_builder.py
│   └── helpers.py
└── assets/
```

## Setup

### 1) Create a virtual environment

```bash
python -m venv .venv
```

### 2) Activate it

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Create your `.env`

Add your Groq API key and optional model name:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

## Get a Groq API Key

1. Create or sign in to your Groq account.
2. Open the Groq API dashboard.
3. Generate an API key.
4. Paste the key into `.env` as `GROQ_API_KEY`.

## Run the app

```bash
streamlit run app.py
```

If Streamlit opens a browser automatically, great. If not, visit the local URL printed in the terminal.

## How it works

- The selected personality mode loads a dedicated system prompt from `prompts/`.
- The user text becomes the prompt content sent to the model.
- `temperature` controls creativity and randomness.
- `requests` sends an OpenAI-compatible chat completion request to Groq.
- The UI streams tokens as they arrive and stores the exchange in session memory.

## API Endpoint

This project uses the OpenAI-compatible Groq endpoint:

```text
https://api.groq.com/openai/v1/chat/completions
```

## Extending the project

You can easily add:

- more personality modes
- per-mode prompt tuning
- persistent database chat history
- user authentication
- image or voice input
- analytics around token usage

## Notes

- No local models are used.
- API keys are never hardcoded.
- The app is intentionally modular so each part is easy to explain in a portfolio or interview.
