# 🤖 AI MCQ Aggregator

A web application that queries multiple AI models simultaneously to solve multiple-choice questions and provides consensus-based answers with confidence percentages.

## ✨ Features

- **Multi-Model Analysis**: Query multiple AI models in parallel for comprehensive answers
- **Consensus Voting**: Aggregates responses and shows confidence percentages for each option
- **Flexible Input Methods**:
  - Direct text input
  - Image upload (with OCR)
  - Paste images directly (Ctrl+V / Cmd+V)
- **Real-time Streaming**: See responses as they arrive from each model
- **Clean UI**: Modern, dark-themed interface with responsive design

## 🎯 Supported AI Models

Currently using **free-tier models** because, you know, I'm a broke developer who values my wallet more than my dignity:

- **Google Gemini 2.0 Flash** - Google's generosity at its finest
- **Meta Llama 3.3 70B** - Courtesy of OpenRouter's free tier (bless them)
- **Qwen 2.5 72B** - Another OpenRouter freebie (I promise I'll upgrade... someday)

*Note: Yes, I know there are fancier paid models out there like GPT-4, Claude Opus, etc. But until my GitHub sponsors start rolling in, we're riding the free tier train! 🚂💨*

## 🚀 Installation

### Prerequisites

- Python 3.8+
- Tesseract OCR (for image processing)

### Steps

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-mcq-aggregator.git
cd ai-mcq-aggregator
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install Tesseract OCR:
   - **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - **Mac**: `brew install tesseract`
   - **Linux**: `sudo apt-get install tesseract-ocr`

5. Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_google_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

## 🔑 Getting API Keys

### Google Gemini API (Free)
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add to `.env` file

### OpenRouter API (Free Tier Available)
1. Visit [OpenRouter](https://openrouter.ai/)
2. Sign up and get your API key
3. Add to `.env` file

## 🎮 Usage

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Input your MCQ question by:
   - Typing/pasting text directly
   - Uploading an image file
   - Pasting an image with Ctrl+V (Cmd+V on Mac)

4. Click "Analyze" and watch the magic happen!

## 📁 Project Structure

```
ai-mcq-aggregator/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── .env               # API keys (not in repo)
├── templates/
│   └── index.html     # Main UI
├── static/
│   ├── style.css      # Styling
│   └── script.js      # Frontend logic
└── README.md          # You are here!
```

## 🛠️ Technology Stack

- **Backend**: Flask (Python)
- **AI APIs**: Google Gemini, OpenRouter
- **OCR**: Tesseract, Pillow
- **Frontend**: Vanilla JavaScript, HTML5, CSS3
- **Streaming**: Server-Sent Events (SSE)

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests
- Donate API credits (just kidding... unless? 👀)

## 📝 License

MIT License - feel free to use this project however you'd like!

## ⚠️ Limitations

- Free tier models have rate limits
- OCR accuracy depends on image quality
- Response time varies by model availability
- Consensus only as good as the models used (garbage in, garbage out!)

## 🙏 Acknowledgments

- Google for the free Gemini API
- OpenRouter for their generous free tier
- The open-source community for all the amazing libraries
- Coffee, for making this project possible

## 📧 Contact

Have questions? Open an issue or reach out!

---

**Made with 💚 (and a lot of free API calls)**
