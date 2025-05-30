import os

# OpenAI-compatible endpoint (Ollama, LM Studio, vLLM, etc.)
OPENAI_API_URL = os.environ.get("OPENAI_API_URL", "http://localhost:11434/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "ollama")

# Kurtis E1 system prompt.
SYSTEM_PROMPT = "You are Kurtis, an empathetic mental health assistant. Keep responses short and conversational, as if you're on a calm phone call. Don't use glyphs or emoticons."

# Supported Lannguages
SUPPORTED_LANGUAGES = {
    "english": {
        "code": "en",
        "name": "English",
        "default_speaker": "Daisy Studious",
    },
    "portuguese": {
        "code": "pt",
        "name": "Portuguese",
        "default_speaker": "Gilberto Mathias",
    },
    "spanish": {
        "code": "es",
        "name": "Spanish",
        "default_speaker": "Alma María",
    },
    "french": {
        "code": "fr",
        "name": "French",
        "default_speaker": "Zacharie Aimilios",
    },
    "german": {
        "code": "de",
        "name": "German",
        "default_speaker": "Brenda Stern",
    },
    "dutch": {
        "code": "nl",
        "name": "Dutch",
        "default_speaker": "Annmarie Nele",
    },
    "italian": {
        "code": "it",
        "name": "Italian",
        "default_speaker": "Eugenio Mataracı",
    },
    "korean": {
        "code": "ko",
        "name": "Korean",
        "default_speaker": "Asya Anara",
    },
    "chinese": {
        "code": "zh",
        "name": "Chinese",
        "default_speaker": "Xavier Hayasaka",
    },
    "russian": {
        "code": "ru",
        "name": "Russian",
        "default_speaker": "Lidiya Szekeres",
    },
}


# Available Speakers
SPEAKERS = [
    "Claribel Dervla",
    "Daisy Studious",
    "Gracie Wise",
    "Tammie Ema",
    "Alison Dietlinde",
    "Ana Florence",
    "Annmarie Nele",
    "Asya Anara",
    "Brenda Stern",
    "Gitta Nikolina",
    "Henriette Usha",
    "Sofia Hellen",
    "Tammy Grit",
    "Tanja Adelina",
    "Vjollca Johnnie",
    "Andrew Chipper",
    "Badr Odhiambo",
    "Dionisio Schuyler",
    "Royston Min",
    "Viktor Eka",
    "Abrahan Mack",
    "Adde Michal",
    "Baldur Sanjin",
    "Craig Gutsy",
    "Damien Black",
    "Gilberto Mathias",
    "Ilkin Urbano",
    "Kazuhiko Atallah",
    "Ludvig Milivoj",
    "Suad Qasim",
    "Torcull Diarmuid",
    "Viktor Menelaos",
    "Zacharie Aimilios",
    "Nova Hogarth",
    "Maja Ruoho",
    "Uta Obando",
    "Lidiya Szekeres",
    "Chandra MacFarland",
    "Szofi Granger",
    "Camilla Holmström",
    "Lilya Stainthorpe",
    "Zofija Kendrick",
    "Narelle Moon",
    "Barbora MacLean",
    "Alexandra Hisakawa",
    "Alma María",
    "Rosemary Okafor",
    "Ige Behringer",
    "Filip Traverse",
    "Damjan Chapman",
    "Wulf Carlevaro",
    "Aaron Dreschner",
    "Kumar Dahl",
    "Eugenio Mataracı",
    "Ferran Simen",
    "Xavier Hayasaka",
    "Luis Moray",
    "Marcos Rudaski",
]
