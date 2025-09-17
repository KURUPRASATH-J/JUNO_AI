# ADDED: Standard library imports for logging
import logging
import os
import sys

# ADDED: Fix for sqlite3 compatibility on platforms like Hugging Face Spaces
# This needs to be at the top before other imports that might use sqlite3
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    print("Successfully patched sqlite3 with pysqlite3.")
except ImportError:
    print("pysqlite3 not found, using standard sqlite3 library.")

# NEWLY ADDED: Set up proper cache directories for deployment environments
os.environ['TRANSFORMERS_CACHE'] = '/code/.cache/huggingface'
os.environ['HF_HOME'] = '/code/.cache/huggingface'
os.environ['TORCH_HOME'] = '/code/.cache/torch'
os.environ['HF_HUB_CACHE'] = '/code/.cache/huggingface'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = '/code/.cache/sentence_transformers'

import json
import uuid
import time
import random
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, GoogleAPIError
from langchain.text_splitter import RecursiveCharacterTextSplitter
# MODIFIED: LangChain imports updated for compatibility
from langchain_community.vectorstores import Chroma
# NEWLY MODIFIED: Use the dedicated langchain-huggingface package for embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import PyPDF2
import io
from typing import List, Dict, Any
import requests
from bs4 import BeautifulSoup
import re
import pytesseract
from PIL import Image

# Import Juno AI Prompts System
from prompts import juno_prompts, get_main_conversation_prompt, get_document_summary_prompt, get_rag_prompt, get_streaming_prompt, get_fallback_responses

# Load environment variables
load_dotenv()

# ADDED: Set up proper logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)

# --- Configuration ---
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GENERATIVE_MODEL = os.getenv('GENERATIVE_MODEL', 'gemini-2.5-flash')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')

# Configure Gemini
if not GEMINI_API_KEY:
    logging.error("GEMINI_API_KEY environment variable not set.")
else:
    genai.configure(api_key=GEMINI_API_KEY)

class ChatbotWithMemoryAndRAG:
    """
    A comprehensive chatbot class that handles conversations, memory,
    document processing (RAG), and web scraping for the Juno AI assistant.
    """

    def __init__(self):
        """Initializes the chatbot instance."""
        logging.info("Initializing Juno AI...")
        # NEWLY MODIFIED: More robust embedding model initialization
        try:
            cache_dir = os.environ.get('SENTENCE_TRANSFORMERS_HOME', '/code/.cache/sentence_transformers')
            os.makedirs(cache_dir, exist_ok=True)
            logging.info(f"Initializing embeddings with model: {EMBEDDING_MODEL}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                cache_folder=cache_dir,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logging.info("HuggingFace Embeddings initialized successfully.")
        except Exception as e:
            logging.error(f"CRITICAL: Could not initialize embeddings: {e}", exc_info=True)
            logging.warning("Continuing without embeddings - RAG features will be disabled.")
            self.embeddings = None

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.vectorstore = None
        self.chat_history = []
        self.memory = {}
        # NEWLY ADDED: Simple user name storage
        self.user_name = None
        self.session_id = str(uuid.uuid4())
        self.last_rate_limit = None
        self.consecutive_rate_limits = 0
        self.prompts = juno_prompts
        logging.info(f"🤖 Juno AI initialized with session ID: {self.session_id}")

    def extract_name_from_message(self, user_message):
        """Simple name extraction from user messages"""
        message_lower = user_message.lower()

        # Pattern matching for name extraction
        patterns = [
            r"i am ([a-zA-Z]+)",
            r"i'm ([a-zA-Z]+)", 
            r"my name is ([a-zA-Z]+)",
            r"call me ([a-zA-Z]+)",
            r"name's ([a-zA-Z]+)"
        ]

        for pattern in patterns:
            match = re.search(pattern, message_lower)
            if match:
                name = match.group(1).capitalize()
                self.user_name = name
                logging.info(f"Extracted and stored user name: {name}")
                return name
        return None

    def check_for_name_query(self, user_message):
        """Check if user is asking about their name"""
        message_lower = user_message.lower()
        name_queries = [
            "what is my name",
            "what's my name", 
            "do you know my name",
            "remember my name",
            "my name"
        ]
        return any(query in message_lower for query in name_queries)

    def _retry_with_backoff(self, func, max_retries=5, base_delay=2):
        """Improved retry function with progressive backoff for rate limit handling"""
        if self.last_rate_limit and datetime.now() - self.last_rate_limit < timedelta(seconds=30):
            additional_wait = min(self.consecutive_rate_limits * 5, 30)
            logging.warning(f"Recent rate limits detected, waiting additional {additional_wait}s")
            time.sleep(additional_wait)

        for attempt in range(max_retries):
            try:
                result = func()
                self.consecutive_rate_limits = 0
                self.last_rate_limit = None
                return result
            except ResourceExhausted as e:
                self.last_rate_limit = datetime.now()
                self.consecutive_rate_limits += 1
                if attempt == max_retries - 1:
                    logging.error(f"Max retries ({max_retries}) exceeded for rate limit.")
                    raise e
                delay = base_delay * (2 ** attempt) + random.uniform(1, 3)
                delay = min(delay, 60)
                logging.warning(f"Rate limit hit (attempt {attempt + 1}/{max_retries}), waiting {delay:.1f}s...")
                time.sleep(delay)
            except GoogleAPIError as e:
                logging.error(f"Google API Error: {e}")
                if "quota" in str(e).lower() or "rate" in str(e).lower():
                    self.last_rate_limit = datetime.now()
                    self.consecutive_rate_limits += 1
                    if attempt == max_retries - 1:
                        raise ResourceExhausted("API quota exceeded")
                    delay = base_delay * (2 ** attempt) + random.uniform(1, 3)
                    delay = min(delay, 60)
                    logging.warning(f"API quota issue, waiting {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    raise e
            except Exception as e:
                logging.error(f"Non-retryable error: {e}", exc_info=True)
                raise e

    def _fallback_response(self, user_message):
        """Generate a fallback response when API is unavailable"""
        logging.warning(f"Generating fallback response for message: '{user_message[:50]}...'")
        fallback_templates = get_fallback_responses()
        template = random.choice(fallback_templates)
        response = template.format(user_message_preview=user_message[:50])
        self.chat_history.append({"user": user_message, "bot": response, "timestamp": datetime.now().isoformat(), "fallback": True})
        return response

    def extract_text_from_pdf(self, pdf_content):
        """Extract text content from PDF bytes with OCR fallback"""
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
            text = ""
            for i, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text and len(page_text.strip()) > 10:
                    text += page_text + "\n"
                else:
                    logging.info(f"Poor text extraction on page {i+1}. Attempting OCR fallback.")
                    try:
                        for image_file_object in page.images:
                            img = Image.open(io.BytesIO(image_file_object.data))
                            ocr_text = pytesseract.image_to_string(img)
                            if ocr_text:
                                text += ocr_text + "\n"
                    except Exception as ocr_error:
                        logging.warning(f"OCR fallback failed for a page: {ocr_error}")
            return text
        except Exception as e:
            logging.error(f"Error extracting PDF: {e}", exc_info=True)
            return f"Error extracting PDF: {str(e)}"

    def process_document(self, text_content, filename="document"):
        """Process document text and create vector store"""
        # NEWLY ADDED: Graceful handling if embeddings failed to initialize
        if self.embeddings is None:
            logging.error("Embeddings are not available. Cannot process document.")
            return "Error: Document processing is disabled because the embedding model could not be loaded."

        try:
            logging.info(f"Processing document: {filename}")
            chunks = self.text_splitter.split_text(text_content)
            documents = [Document(page_content=chunk, metadata={"source": filename, "chunk_id": i}) for i, chunk in enumerate(chunks)]
            if self.vectorstore is None:
                self.vectorstore = Chroma.from_documents(documents=documents, embedding=self.embeddings, collection_name=f"collection_{self.session_id}")
            else:
                self.vectorstore.add_documents(documents)
            logging.info(f"Successfully processed {len(chunks)} chunks from {filename}")
            return f"Successfully processed {len(chunks)} chunks from {filename}"
        except Exception as e:
            logging.error(f"Error processing document: {e}", exc_info=True)
            return f"Error processing document: {str(e)}"

    def retrieve_relevant_context(self, query, k=3):
        """Retrieve relevant context from vector store"""
        if self.vectorstore is None:
            return ""
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            return "\n".join([doc.page_content for doc in docs])
        except Exception as e:
            logging.error(f"Error retrieving context: {e}", exc_info=True)
            return ""

    def summarize_text(self, text, max_length=500):
        """Summarize long text using Juno AI prompts"""
        def _summarize():
            model = genai.GenerativeModel(GENERATIVE_MODEL)
            prompt = self.prompts.get_document_summarization_prompt(text, max_length)
            return model.generate_content(prompt).text

        try:
            return self._retry_with_backoff(_summarize)
        except (ResourceExhausted, GoogleAPIError):
            logging.warning("Summarization failed due to high API usage.")
            return f"📄 Document uploaded successfully ({len(text)} characters). \n\n✨ **Juno AI Note:** Summary temporarily unavailable due to high API usage, but the document content is fully searchable and ready for your questions!"
        except Exception as e:
            logging.error(f"Error summarizing text: {e}", exc_info=True)
            return f"Error summarizing text: {str(e)}"

    def generate_response(self, user_message, context=""):
        """Generate response using Juno AI prompts"""
        # NEWLY ADDED: Extract name from message first
        self.extract_name_from_message(user_message)

        # NEWLY ADDED: Handle name queries directly
        if self.check_for_name_query(user_message):
            if self.user_name:
                response = f"Your name is {self.user_name}! I remember when you told me."
                self.chat_history.append({"user": user_message, "bot": response, "timestamp": datetime.now().isoformat()})
                return response
            else:
                response = "I don't know your name yet. Would you like to tell me what it is?"
                self.chat_history.append({"user": user_message, "bot": response, "timestamp": datetime.now().isoformat()})
                return response

        def _generate():
            model = genai.GenerativeModel(GENERATIVE_MODEL)
            conversation_history = []
            if self.chat_history:
                for exchange in self.chat_history[-3:]:
                    if not exchange.get('fallback', False):
                        conversation_history.append({'user': exchange['user'], 'bot': exchange['bot'], 'timestamp': exchange.get('timestamp', '')})

            # NEWLY ADDED: Include user name in prompt
            base_prompt = self.prompts.get_conversation_prompt(user_message=user_message, context=context, conversation_history=conversation_history, memory_context=self.memory)
            if self.user_name:
                base_prompt += f"\n\nIMPORTANT: The user's name is {self.user_name}. Address them personally when appropriate."

            return model.generate_content(base_prompt).text

        try:
            bot_response = self._retry_with_backoff(_generate)
            self.chat_history.append({"user": user_message, "bot": bot_response, "timestamp": datetime.now().isoformat()})
            self.update_memory(user_message, bot_response)
            return bot_response
        except (ResourceExhausted, GoogleAPIError):
            return self._fallback_response(user_message)
        except Exception as e:
            logging.error(f"Error generating response: {e}", exc_info=True)
            return f"Error generating response: {str(e)}"

    def update_memory(self, user_message, bot_response):
        """Update session memory with important information"""
        if "memory" not in self.memory: 
            self.memory["memory"] = []
        self.memory["memory"].append({"user": user_message, "bot": bot_response, "timestamp": datetime.now().isoformat()})
        if len(self.memory["memory"]) > 10: 
            self.memory["memory"] = self.memory["memory"][-10:]

    def scrape_web_content(self, url):
        """Scrape content from a web URL"""
        try:
            logging.info(f"Scraping web content from: {url}")
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            for script in soup(["script", "style"]): 
                script.decompose()
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            return ' '.join(chunk for chunk in chunks if chunk)[:10000]
        except Exception as e:
            logging.error(f"Error scraping URL '{url}': {e}", exc_info=True)
            return f"Error scraping URL: {str(e)}"

    def analyze_web_content(self, url, content):
        """Analyze scraped web content using Juno AI prompts"""
        def _analyze():
            model = genai.GenerativeModel(GENERATIVE_MODEL)
            prompt = self.prompts.get_web_content_analysis_prompt(url, content)
            return model.generate_content(prompt).text

        try:
            return self._retry_with_backoff(_analyze)
        except (ResourceExhausted, GoogleAPIError):
            logging.warning(f"Web content analysis failed for '{url}' due to high API usage.")
            return f"🌐 **Web Content Scraped Successfully**\n\n**URL:** {url}\n**Content Length:** {len(content)} characters\n\n**Juno AI Note:** Analysis temporarily unavailable due to high API usage, but the content has been processed and is ready for your questions!"
        except Exception as e:
            logging.error(f"Error analyzing web content for '{url}': {e}", exc_info=True)
            return f"Error analyzing web content: {str(e)}"

    def generate_rag_response(self, user_query, context, sources=None):
        """Generate RAG response using Juno AI prompts"""
        # NEWLY ADDED: Extract name from query first
        self.extract_name_from_message(user_query)

        # NEWLY ADDED: Handle name queries directly
        if self.check_for_name_query(user_query):
            if self.user_name:
                return f"Your name is {self.user_name}! I remember when you told me."
            else:
                return "I don't know your name yet. Would you like to tell me what it is?"

        def _generate_rag():
            model = genai.GenerativeModel(GENERATIVE_MODEL)
            context_chunks = [context[i:i+2000] for i in range(0, len(context), 2000)]

            base_prompt = self.prompts.get_rag_response_prompt(user_query=user_query, retrieved_chunks=context_chunks[:3], source_info=sources)
            if self.user_name:
                base_prompt += f"\n\nIMPORTANT: The user's name is {self.user_name}. Address them personally when appropriate."

            return model.generate_content(base_prompt).text

        try:
            return self._retry_with_backoff(_generate_rag)
        except (ResourceExhausted, GoogleAPIError):
            return self._fallback_response(user_query)
        except Exception as e:
            logging.error(f"Error generating RAG response: {e}", exc_info=True)
            return f"Error generating RAG response: {str(e)}"

    def save_conversation(self, conversation_id, title=""):
        """Save current conversation to memory"""
        if not title: 
            title = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        conversation_data = {"id": conversation_id, "title": title, "messages": self.chat_history, "created_at": datetime.now().isoformat(), "last_updated": datetime.now().isoformat()}
        if "conversations" not in self.memory: 
            self.memory["conversations"] = {}
        self.memory["conversations"][conversation_id] = conversation_data
        logging.info(f"Conversation '{conversation_id}' saved with title '{title}'.")
        return conversation_data

    def load_conversation(self, conversation_id):
        """Load a specific conversation"""
        if "conversations" in self.memory and conversation_id in self.memory["conversations"]:
            conversation = self.memory["conversations"][conversation_id]
            self.chat_history = conversation["messages"]
            logging.info(f"Conversation '{conversation_id}' loaded.")
            return conversation
        logging.warning(f"Attempted to load non-existent conversation '{conversation_id}'.")
        return None

    def delete_conversation(self, conversation_id):
        """Delete a specific conversation"""
        if "conversations" in self.memory and conversation_id in self.memory["conversations"]:
            del self.memory["conversations"][conversation_id]
            logging.info(f"Conversation '{conversation_id}' deleted.")
            return True
        logging.warning(f"Attempted to delete non-existent conversation '{conversation_id}'.")
        return False

    def rename_conversation(self, conversation_id, new_title):
        """Rename a conversation"""
        if "conversations" in self.memory and conversation_id in self.memory["conversations"]:
            self.memory["conversations"][conversation_id]["title"] = new_title
            self.memory["conversations"][conversation_id]["last_updated"] = datetime.now().isoformat()
            logging.info(f"Conversation '{conversation_id}' renamed to '{new_title}'.")
            return True
        logging.warning(f"Attempted to rename non-existent conversation '{conversation_id}'.")
        return False

    def generate_streaming_response(self, user_message, context=""):
        """Generate streaming response using Juno AI prompts"""
        # NEWLY ADDED: Extract name from message first
        self.extract_name_from_message(user_message)

        # NEWLY ADDED: Handle name queries directly - NO STREAMING for simple responses
        if self.check_for_name_query(user_message):
            if self.user_name:
                return f"Your name is {self.user_name}! I remember when you told me."
            else:
                return "I don't know your name yet. Would you like to tell me what it is?"

        def _generate_stream():
            model = genai.GenerativeModel(GENERATIVE_MODEL)

            base_prompt = self.prompts.get_streaming_response_prompt(user_message, context)
            if self.user_name:
                base_prompt += f"\n\nIMPORTANT: The user's name is {self.user_name}. Address them personally when appropriate."

            return model.generate_content(base_prompt, stream=True)

        try:
            return self._retry_with_backoff(_generate_stream, max_retries=3, base_delay=1)
        except (ResourceExhausted, GoogleAPIError):
            logging.warning("Streaming response failed due to API rate limit.")
            return None
        except Exception as e:
            logging.error(f"Error generating streaming response: {e}", exc_info=True)
            return None

# Initialize Juno AI chatbot
chatbot = ChatbotWithMemoryAndRAG()

@app.route('/')
def serve_frontend():
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('.', filename)

# --- API Endpoints ---

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400

        context = chatbot.retrieve_relevant_context(user_message)
        if context:
            bot_response = chatbot.generate_rag_response(user_message, context)
        else:
            bot_response = chatbot.generate_response(user_message, context)

        return jsonify({
            'response': bot_response,
            'has_context': bool(context),
            'session_id': chatbot.session_id
        })
    except Exception as e:
        logging.error(f"Error in /api/chat: {e}", exc_info=True)
        return jsonify({'error': 'An internal server error occurred.'}), 500

@app.route('/api/upload', methods=['POST'])
def upload_document():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if file and file.filename.lower().endswith('.pdf'):
            text_content = chatbot.extract_text_from_pdf(file.read())
            if text_content.startswith("Error"):
                return jsonify({'error': text_content}), 400

            result = chatbot.process_document(text_content, file.filename)
            summary = chatbot.summarize_text(text_content)

            return jsonify({
                'message': result,
                'summary': summary,
                'filename': file.filename,
                'text_length': len(text_content)
            })
        else:
            return jsonify({'error': 'Only PDF files are supported'}), 400

    except Exception as e:
        logging.error(f"Error in /api/upload: {e}", exc_info=True)
        return jsonify({'error': 'An internal server error occurred during file upload.'}), 500

@app.route('/api/summarize', methods=['POST'])
def summarize_document():
    try:
        data = request.json
        text = data.get('text', '')
        max_length = data.get('max_length', 500)

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        summary = chatbot.summarize_text(text, max_length)
        return jsonify({
            'summary': summary,
            'original_length': len(text),
            'summary_length': len(summary)
        })

    except Exception as e:
        logging.error(f"Error in /api/summarize: {e}", exc_info=True)
        return jsonify({'error': 'An internal server error occurred during summarization.'}), 500

@app.route('/api/memory', methods=['GET'])
def get_memory():
    return jsonify({
        'memory': chatbot.memory,
        'user_name': chatbot.user_name,  # NEWLY ADDED: Show stored user name
        'chat_history_length': len(chatbot.chat_history),
        'has_vectorstore': chatbot.vectorstore is not None,
        'session_id': chatbot.session_id
    })

@app.route('/api/clear', methods=['POST'])
def clear_session():
    global chatbot
    logging.info("Clearing session and re-initializing Juno AI.")
    chatbot = ChatbotWithMemoryAndRAG()
    return jsonify({'message': 'Juno AI session cleared successfully'})

@app.route('/api/scrape', methods=['POST'])
def scrape_url():
    try:
        data = request.json
        url = data.get('url', '')

        if not url:
            return jsonify({'error': 'No URL provided'}), 400

        if not re.match(r'^https?://', url):
            url = 'https://' + url

        content = chatbot.scrape_web_content(url)
        if content.startswith("Error"):
            return jsonify({'error': content}), 400

        result = chatbot.process_document(content, f"Web: {url}")
        analysis = chatbot.analyze_web_content(url, content)

        return jsonify({
            'message': result,
            'summary': analysis,
            'url': url,
            'content_length': len(content)
        })

    except Exception as e:
        logging.error(f"Error in /api/scrape: {e}", exc_info=True)
        return jsonify({'error': 'An internal server error occurred during web scraping.'}), 500

@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    try:
        data = request.json
        user_message = data.get('message', '')

        if not user_message:
            return jsonify({'error': 'No message provided'}), 400

        # NEWLY ADDED: Handle name queries without streaming to avoid errors
        if chatbot.check_for_name_query(user_message):
            chatbot.extract_name_from_message(user_message)
            if chatbot.user_name:
                bot_response = f"Your name is {chatbot.user_name}! I remember when you told me."
            else:
                bot_response = "I don't know your name yet. Would you like to tell me what it is?"

            chatbot.chat_history.append({"user": user_message, "bot": bot_response, "timestamp": datetime.now().isoformat()})

            return jsonify({
                'response': bot_response,
                'has_context': False,
                'session_id': chatbot.session_id,
                'streaming': False
            })

        context = chatbot.retrieve_relevant_context(user_message)
        streaming_response = chatbot.generate_streaming_response(user_message, context)

        if streaming_response is None or isinstance(streaming_response, str):
            # Handle non-streaming response
            if isinstance(streaming_response, str):
                bot_response = streaming_response
            elif context:
                bot_response = chatbot.generate_rag_response(user_message, context)
            else:
                bot_response = chatbot.generate_response(user_message, context)
            return jsonify({
                'response': bot_response,
                'has_context': bool(context),
                'session_id': chatbot.session_id,
                'streaming': False
            })

        full_response, response_chunks = "", []
        try:
            for chunk in streaming_response:
                if chunk.text:
                    full_response += chunk.text
                    response_chunks.append(chunk.text)
        except (ResourceExhausted, GoogleAPIError):
            if context:
                bot_response = chatbot.generate_rag_response(user_message, context)
            else:
                bot_response = chatbot.generate_response(user_message, context)
            return jsonify({
                'response': bot_response,
                'has_context': bool(context),
                'session_id': chatbot.session_id,
                'streaming': False
            })

        chatbot.chat_history.append({"user": user_message, "bot": full_response, "timestamp": datetime.now().isoformat()})
        chatbot.update_memory(user_message, full_response)

        return jsonify({
            'response': full_response,
            'chunks': response_chunks,
            'has_context': bool(context),
            'session_id': chatbot.session_id,
            'streaming': True
        })

    except Exception as e:
        logging.error(f"Error in /api/chat/stream: {e}", exc_info=True)
        return jsonify({'error': 'An internal server error occurred during streaming.'}), 500

@app.route('/api/conversations', methods=['GET'])
def get_conversations():
    try:
        conversations = []
        if "conversations" in chatbot.memory:
            for conv_id, conv_data in chatbot.memory["conversations"].items():
                conversations.append({'id': conv_id, 'title': conv_data['title'], 'created_at': conv_data['created_at'], 'last_updated': conv_data['last_updated'], 'message_count': len(conv_data['messages'])})
        conversations.sort(key=lambda x: x['last_updated'], reverse=True)
        return jsonify({'conversations': conversations})
    except Exception as e:
        logging.error(f"Error in /api/conversations GET: {e}", exc_info=True)
        return jsonify({'error': 'An internal server error occurred.'}), 500

@app.route('/api/conversations', methods=['POST'])
def save_conversation():
    try:
        data = request.json
        conversation_id = data.get('id', str(uuid.uuid4()))
        title = data.get('title', '')
        conversation = chatbot.save_conversation(conversation_id, title)
        return jsonify({'message': 'Conversation saved successfully', 'conversation': conversation})
    except Exception as e:
        logging.error(f"Error in /api/conversations POST: {e}", exc_info=True)
        return jsonify({'error': 'An internal server error occurred.'}), 500

@app.route('/api/conversations/<conversation_id>', methods=['GET'])
def load_conversation(conversation_id):
    try:
        conversation = chatbot.load_conversation(conversation_id)
        if conversation:
            return jsonify({'message': 'Conversation loaded successfully', 'conversation': conversation})
        else:
            return jsonify({'error': 'Conversation not found'}), 404
    except Exception as e:
        logging.error(f"Error in /api/conversations/{conversation_id} GET: {e}", exc_info=True)
        return jsonify({'error': 'An internal server error occurred.'}), 500

@app.route('/api/conversations/<conversation_id>', methods=['DELETE'])
def delete_conversation(conversation_id):
    try:
        success = chatbot.delete_conversation(conversation_id)
        if success:
            return jsonify({'message': 'Conversation deleted successfully'})
        else:
            return jsonify({'error': 'Conversation not found'}), 404
    except Exception as e:
        logging.error(f"Error in /api/conversations/{conversation_id} DELETE: {e}", exc_info=True)
        return jsonify({'error': 'An internal server error occurred.'}), 500

@app.route('/api/conversations/<conversation_id>/rename', methods=['PUT'])
def rename_conversation(conversation_id):
    try:
        data = request.json
        new_title = data.get('title', '')
        if not new_title:
            return jsonify({'error': 'No title provided'}), 400

        success = chatbot.rename_conversation(conversation_id, new_title)
        if success:
            return jsonify({'message': 'Conversation renamed successfully'})
        else:
            return jsonify({'error': 'Conversation not found'}), 404
    except Exception as e:
        logging.error(f"Error in /api/conversations/{conversation_id}/rename: {e}", exc_info=True)
        return jsonify({'error': 'An internal server error occurred.'}), 500

@app.route('/api/messages/<int:message_index>/edit', methods=['PUT'])
def edit_message(message_index):
    try:
        data = request.json
        new_message = data.get('message', '')

        if not new_message:
            return jsonify({'error': 'No message provided'}), 400

        if 0 <= message_index < len(chatbot.chat_history):
            chatbot.chat_history[message_index]['user'] = new_message
            chatbot.chat_history[message_index]['edited'] = True
            chatbot.chat_history[message_index]['edited_at'] = datetime.now().isoformat()
            chatbot.chat_history = chatbot.chat_history[:message_index + 1]
            return jsonify({'message': 'Message edited successfully', 'updated_history': chatbot.chat_history})
        else:
            return jsonify({'error': 'Invalid message index'}), 400

    except Exception as e:
        logging.error(f"Error in /api/messages/{message_index}/edit: {e}", exc_info=True)
        return jsonify({'error': 'An internal server error occurred.'}), 500

if __name__ == '__main__':
    logging.info("🚀 Starting Juno AI Server...")
    logging.info("🤖 Advanced AI Assistant with Document Processing, Web Scraping, and Memory")
    logging.info("🌟 Powered by Juno AI Prompts System")
    logging.info("🧠 Enhanced with Name Memory (Streaming-Safe)")
    port = int(os.environ.get("PORT", 7860))
    app.run(debug=False, host='0.0.0.0', port=port)
