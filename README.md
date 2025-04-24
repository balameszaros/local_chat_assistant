# local_chat_assistant
Creating a local chat assistant, that can run on your own machine.

```bash
# Create a virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Folder structure:
```text
local_chat_assistant/
├── scripts/
│   └── chatbot_local.py
├── state_db/
│   └── chat_history.db
├── requirements.txt
└── README.md
```

Start the application wiht streamlit:
```bash
python3 -m streamlit run scripts/new_chatbot_local.py   
```