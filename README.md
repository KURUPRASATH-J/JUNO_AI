JUNO AI 🤖
JUNO AI is an advanced, conversational AI assistant featuring a sleek web interface. It's designed to be a powerful tool for information processing, capable of understanding and discussing uploaded PDF documents, scraping and analyzing web content, and retaining memory across conversations. Built with a robust Python Flask backend and a dynamic vanilla JavaScript front end, JUNO AI leverages the power of Google's Gemini models and the LangChain framework for Retrieval-Augmented Generation (RAG).

(You can replace this placeholder image with a screenshot or GIF of your application)

✨ Core Features
📄 Document Analysis (RAG): Upload PDF files directly into the chat. JUNO AI will process, index, and use the document's content to answer your questions with context-aware responses.

🌐 Web Scraping & Analysis: Provide a URL, and JUNO AI will scrape its content, analyze it, and make it available for conversation, allowing you to discuss articles, blog posts, and more.

🧠 Smart Memory:

Session Memory: Remembers the context of the current conversation.

Conversation Management: Save, load, and delete entire chat sessions from a convenient sidebar menu.

Personalization: Remembers your name if you mention it, providing a more personalized experience.

🎤 Voice Input: Use your microphone to dictate messages directly into the chat interface (browser-dependent).

⚡ Real-time Streaming: Responses are streamed from the server in real-time, creating a smooth and dynamic conversational experience.

🎨 Dual Theme UI: Toggle between a sleek light mode and a cool dark mode to suit your preference.

🐳 Dockerized for Easy Deployment: The entire application is containerized with a Dockerfile, ensuring a consistent and straightforward deployment process.

🚀 Advanced Prompt System: Utilizes a centralized prompts.py module to manage and enforce the AI's personality, capabilities, and response structure for various tasks.
🚀 Getting Started
You can run JUNO AI either locally using Python or as a Docker container.

Prerequisites
Python 3.9+

Git

Google Gemini API Key

Docker (for containerized deployment)

Tesseract OCR (for local deployment, if your PDFs are image-based)

1. Local Installation
Step 1: Clone the repository

Bash

git clone https://github.com/your-username/juno-ai.git
cd juno-ai
Step 2: Set up the backend
Create a virtual environment and install the required Python packages.

Bash

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
Step 3: Configure environment variables
Create a file named .env in the root directory and add your Gemini API key.

Code snippet

# .env
GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"

Step 4: Run the application
Start the Flask server.

Bash

python app.py

Step 5: Access JUNO AI
Open your web browser and navigate to http://localhost:7860. You should now see the JUNO AI interface!

2. Docker Deployment
The simplest way to get started is with Docker, as it handles all system dependencies like Tesseract automatically.

Step 1: Clone the repository and create the .env file
Follow steps 1 and 3 from the local installation guide above.

Step 2: Build the Docker image
From the root directory of the project, run:

Bash

docker build -t juno-ai .
Step 3: Run the Docker container
This command runs the container and passes the .env file for API key configuration.

Bash

docker run -p 7860:7860 --env-file .env juno-ai
Step 4: Access JUNO AI
Open your web browser and navigate to http://localhost:7860.

🏗️ How It Works
The application follows a standard client-server architecture with a sophisticated RAG pipeline on the backend.

Frontend (script.js): The user interacts with the UI. All actions (sending messages, uploading files) are sent as API requests to the Flask backend.

Backend (app.py): The Flask server handles these requests.

RAG Pipeline for Documents/Web Pages:

Extraction: Text is extracted from PDFs (using PyPDF2 with a Tesseract OCR fallback) or scraped from URLs (BeautifulSoup).

Chunking: The extracted text is split into smaller, overlapping chunks using LangChain's RecursiveCharacterTextSplitter.

Embedding: Each chunk is converted into a numerical vector representation using HuggingFaceEmbeddings (all-MiniLM-L6-v2).

Storage: These embeddings are stored in a ChromaDB in-memory vector store for efficient similarity searching.

Query Processing:

When a user asks a question, the backend searches the vector store for the most relevant chunks of text.

This retrieved context is combined with the user's query and the conversation history.

A detailed, structured prompt is generated using the prompts.py module.

The complete prompt is sent to the Gemini API to generate a final, context-aware response.

Streaming Response: The response from the Gemini API is streamed back to the client, providing a real-time typing effect in the UI.
🤝 Contributing
Contributions are welcome! If you have suggestions for improvements or new features, feel free to open an issue or submit a pull request.

Fork the Project

Create your Feature Branch (git checkout -b feature/AmazingFeature)

Commit your Changes (git commit -m 'Add some AmazingFeature')

Push to the Branch (git push origin feature/AmazingFeature)

Open a Pull Request

