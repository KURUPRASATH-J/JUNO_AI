Of course. Here is a revised version of your `README.md` file, formatted to be more attractive and engaging for readers on platforms like GitHub.

It includes a table of contents, badges, emojis, collapsible sections for cleaner navigation, and clearer visual separation between topics.

-----

\<div align="center"\>

# JUNO AI 🤖

### An advanced, conversational AI assistant with RAG, web scraping, and smart memory.

\<p align="center"\>
\<img alt="Python Version" src="[https://img.shields.io/badge/python-3.9+-blue.svg?style=for-the-badge\&logo=python\&logoColor=white](https://www.google.com/search?q=https://img.shields.io/badge/python-3.9%2B-blue.svg%3Fstyle%3Dfor-the-badge%26logo%3Dpython%26logoColor%3Dwhite)"\>
\<img alt="Framework" src="[https://img.shields.io/badge/Flask-000000?style=for-the-badge\&logo=flask\&logoColor=white](https://www.google.com/search?q=https://img.shields.io/badge/Flask-000000%3Fstyle%3Dfor-the-badge%26logo%3Dflask%26logoColor%3Dwhite)"\>
\<img alt="License" src="[https://img.shields.io/badge/license-MIT-green.svg?style=for-the-badge](https://www.google.com/search?q=https://img.shields.io/badge/license-MIT-green.svg%3Fstyle%3Dfor-the-badge)"\>
\</p\>

\</div\>

-----

**JUNO AI** is a powerful tool for information processing, capable of understanding and discussing uploaded PDF documents, scraping and analyzing web content, and retaining memory across conversations. Built with a robust Python Flask backend and a dynamic vanilla JavaScript front end, JUNO AI leverages Google's Gemini models and the LangChain framework for Retrieval-Augmented Generation (RAG).

-----

### **Table of Contents**

  * [✨ Core Features](https://www.google.com/search?q=%23-core-features)
  * [🛠️ Tech Stack](https://www.google.com/search?q=%23%EF%B8%8F-tech-stack)
  * [🚀 Getting Started](https://www.google.com/search?q=%23-getting-started)
  * [🏗️ How It Works](https://www.google.com/search?q=%23%EF%B8%8F-how-it-works)
  * [🤝 Contributing](https://www.google.com/search?q=%23-contributing)

-----

## ✨ Core Features

| Feature                | Description                                                                                                                                              |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 📄 **Document Analysis** | Upload PDFs directly into the chat. JUNO AI processes, indexes, and uses the content to provide context-aware answers.                                     |
| 🌐 **Web Scraping** | Provide a URL to scrape and analyze its content, allowing you to discuss articles, blog posts, and more.                                                     |
| 🧠 **Smart Memory** | Includes session memory, full conversation management (save/load/delete), and personalization by remembering your name.                               |
| 🎤 **Voice Input** | Use your microphone to dictate messages directly into the chat interface.                                                                                |
| ⚡ **Real-time Streaming** | Responses are streamed from the server in real-time for a smooth and dynamic conversational experience.                                                  |
| 🎨 **Dual Theme UI** | Toggle between a sleek light mode and a cool dark mode to suit your preference.                                                                          |
| 🐳 **Dockerized** | [cite\_start]The entire application is containerized with a `Dockerfile`, ensuring a consistent and straightforward deployment process[cite: 2].                       |
| 🚀 **Advanced Prompts** | Utilizes a centralized `prompts.py` module to manage and enforce the AI's personality, capabilities, and response structure.                        |

-----

## 🛠️ Tech Stack

| Category               | Technologies                                                                                                                          |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **Backend** | [cite\_start]Python [cite: 2][cite\_start], Flask [cite: 1][cite\_start], Google Generative AI (Gemini) [cite: 1][cite\_start], LangChain [cite: 1][cite\_start], ChromaDB, HuggingFace Embeddings [cite: 1][cite\_start], BeautifulSoup [cite: 1] |
| **Frontend** | [cite\_start]HTML5, CSS3, Vanilla JavaScript (ES6) [cite: 3]                                                                                        |
| **DevOps & Tools** | [cite\_start]Docker [cite: 2][cite\_start], Tesseract OCR[cite: 2], Git                                                                                                |

-----

## 🚀 Getting Started

### **Prerequisites**

  * Python 3.9+
  * Git
  * Docker (Recommended)
  * Google Gemini API Key

\<br/\>

\<details\>
\<summary\>\<strong\>Option 1: Docker Deployment (Recommended)\</strong\>\</summary\>

The simplest way to get started is with Docker, as it handles all system dependencies like Tesseract automatically.

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/juno-ai.git
    cd juno-ai
    ```
2.  **Create `.env` File**
    Create a `.env` file in the root directory and add your API key:
    ```
    GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"
    ```
3.  **Build the Docker Image**
    ```bash
    docker build -t juno-ai .
    ```
4.  **Run the Container**
    ```bash
    docker run -p 7860:7860 --env-file .env juno-ai
    ```
5.  **Access JUNO AI**
    Open your browser and navigate to `http://localhost:7860`.

\</details\>

\<details\>
\<summary\>\<strong\>Option 2: Local Python Installation\</strong\>\</summary\>

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/juno-ai.git
    cd juno-ai
    ```
2.  **Set up Virtual Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Create `.env` File**
    Create a `.env` file in the root directory and add your API key:
    ```
    GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"
    ```
5.  **Run the Application**
    ```bash
    python app.py
    ```
6.  **Access JUNO AI**
    Open your browser and navigate to `http://localhost:7860`.

\</details\>

-----

## 🏗️ How It Works

The application uses a sophisticated RAG (Retrieval-Augmented Generation) pipeline.

1.  [cite\_start]**Frontend Interaction (`script.js`)**: The user sends messages, uploads files, or submits URLs through the UI[cite: 3]. These actions are sent as API requests to the backend.

2.  **Backend Processing (`app.py`)**: The Flask server handles all incoming requests.

3.  **The RAG Pipeline**:

      * **📜 Extraction**: Text is extracted from PDFs (using `PyPDF2` with a `Tesseract` OCR fallback) or scraped from URLs (`BeautifulSoup`).
      * **✂️ Chunking**: The text is split into smaller, overlapping chunks using `LangChain's RecursiveCharacterTextSplitter`.
      * [cite\_start]**🧠 Embedding**: Each chunk is converted into a numerical vector using `HuggingFaceEmbeddings`[cite: 1].
      * [cite\_start]**💾 Storage**: Embeddings are stored in a `ChromaDB` in-memory vector store for efficient searching[cite: 1].

4.  **Query & Response Generation**:

      * **🔍 Retrieval**: When a user asks a question, the backend searches the vector store for the most relevant text chunks.
      * **📝 Prompting**: The retrieved context, conversation history, and user query are used to generate a detailed prompt via the `prompts.py` module.
      * **💬 Generation**: This complete prompt is sent to the Gemini API to generate a final, context-aware response, which is then streamed back to the user.

-----

## 🤝 Contributing

Contributions are welcome\! If you have suggestions or new features, please feel free to open an issue or submit a pull request.

1.  **Fork** the Project
2.  **Create** your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  **Commit** your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  **Push** to the Branch (`git push origin feature/AmazingFeature`)
5.  **Open** a Pull Request
