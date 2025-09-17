class ChatbotUI {
    constructor() {
        // Auto-detect API base URL for different deployment environments
        if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
            this.apiBase = 'http://localhost:7860/api';
        } else {
            // For Hugging Face Spaces or other deployments, use relative path
            this.apiBase = '/api';
        }
        this.isTyping = false;
        this.hasDocuments = false;
        this.conversations = [];
        this.currentConversationId = null;
        this.isStreaming = false;

        // Speech recognition properties
        this.recognition = null;
        this.isRecording = false;
        this.isListening = false;

        this.initializeElements();
        this.attachEventListeners();
        this.applyInitialTheme();
        this.loadConversations();
        this.initializeSpeechRecognition();
    }

    initializeElements() {
        // Main elements
        this.messagesContainer = document.getElementById('messagesContainer');
        this.messageInput = document.getElementById('messageInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.fileInput = document.getElementById('fileInput');
        this.documentStatus = document.getElementById('documentStatus');
        this.welcomeMessage = document.getElementById('welcomeMessage');

        // Buttons
        this.attachBtn = document.getElementById('attachBtn');
        this.memoryBtn = document.getElementById('memoryBtn');
        this.clearBtn = document.getElementById('clearBtn');
        this.themeToggleBtn = document.getElementById('themeToggleBtn');
        this.voiceBtn = document.getElementById('voiceBtn');
        this.stopVoiceBtn = document.getElementById('stopVoiceBtn');
        this.conversationsBtn = document.getElementById('conversationsBtn');

        // Modal and sidebar
        this.memoryModal = document.getElementById('memoryModal');
        this.closeModalBtn = document.getElementById('closeModalBtn');
        this.memoryContent = document.getElementById('memoryContent');
        this.sidebar = document.getElementById('sidebar');
        this.closeSidebarBtn = document.getElementById('closeSidebarBtn');

        // Voice modal and loading
        this.voiceModal = document.getElementById('voiceModal');
        this.loadingOverlay = document.getElementById('loadingOverlay');
    }

    initializeSpeechRecognition() {
        // Check if speech recognition is supported
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            this.recognition = new SpeechRecognition();

            // Configure recognition settings
            this.recognition.continuous = true;
            this.recognition.interimResults = true;
            this.recognition.lang = 'en-US';

            // Event handlers
            this.recognition.onstart = () => {
                console.log('🎤 Speech recognition started');
                this.isListening = true;
                this.voiceBtn.classList.add('voice-recording');
                this.showVoiceModal();
            };

            this.recognition.onresult = (event) => {
                let finalTranscript = '';
                let interimTranscript = '';

                for (let i = event.resultIndex; i < event.results.length; i++) {
                    const transcript = event.results[i][0].transcript;
                    if (event.results[i].isFinal) {
                        finalTranscript += transcript;
                    } else {
                        interimTranscript += transcript;
                    }
                }

                // Update input field with transcription
                this.messageInput.value = finalTranscript + interimTranscript;
                this.autoResizeInput();
                this.updateSendButton();
            };

            this.recognition.onerror = (event) => {
                console.error('🎤 Speech recognition error:', event.error);
                this.showNotification(`Voice recognition error: ${event.error}`, 'error');
                this.stopVoiceRecording();
            };

            this.recognition.onend = () => {
                console.log('🎤 Speech recognition ended');
                this.isListening = false;
                this.stopVoiceRecording();
            };
        } else {
            console.warn('🎤 Speech recognition not supported in this browser');
        }
    }

    attachEventListeners() {
        // Send message events
        this.sendBtn.addEventListener('click', () => this.sendMessage());

        // Shift+Enter for new line
        this.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                if (e.shiftKey) {
                    return;
                } else {
                    e.preventDefault();
                    this.sendMessage();
                }
            }
        });

        // File upload events
        this.attachBtn.addEventListener('click', () => this.fileInput.click());
        this.fileInput.addEventListener('change', (e) => this.handleFileUpload(e));

        // Voice events
        this.voiceBtn.addEventListener('click', () => this.toggleVoiceRecording());
        this.stopVoiceBtn.addEventListener('click', () => this.stopVoiceRecording());

        // Close voice modal on click outside
        this.voiceModal.addEventListener('click', (e) => {
            if (e.target === this.voiceModal) {
                this.stopVoiceRecording();
            }
        });

        // Conversations button (Menu button)
        this.conversationsBtn.addEventListener('click', () => this.toggleSidebar());

        // Modal events
        this.memoryBtn.addEventListener('click', () => this.showMemory());
        this.closeModalBtn.addEventListener('click', () => this.hideMemory());
        this.memoryModal.addEventListener('click', (e) => {
            if (e.target === this.memoryModal) this.hideMemory();
        });

        // Clear session
        this.clearBtn.addEventListener('click', () => this.clearSession());

        // Sidebar events
        this.closeSidebarBtn.addEventListener('click', () => this.closeSidebar());

        // Auto-resize textarea
        this.messageInput.addEventListener('input', () => {
            this.autoResizeInput();
            this.updateSendButton();
        });

        // Theme toggle
        this.themeToggleBtn.addEventListener('click', () => this.toggleTheme());

        // Web scraping button
        const scrapeBtn = document.getElementById('scrapeBtn');
        if (scrapeBtn) {
            scrapeBtn.addEventListener('click', () => this.scrapeWebsite());
        }

        // Save conversation button
        const saveConvBtn = document.getElementById('saveConvBtn');
        if (saveConvBtn) {
            saveConvBtn.addEventListener('click', () => this.saveConversation());
        }

        // New conversation button
        const newConvBtn = document.getElementById('newConvBtn');
        if (newConvBtn) {
            newConvBtn.addEventListener('click', () => this.newConversation());
        }
    }

    // Quick Action Button Helper
    fillInput(text) {
        this.messageInput.value = text;
        this.updateSendButton();
        this.autoResizeInput();
        this.messageInput.focus();
    }

    // Toggle sidebar method
    toggleSidebar() {
        this.sidebar.classList.toggle('active');
        if (this.sidebar.classList.contains('active')) {
            this.loadConversations(); // Refresh conversations when opening
        }
    }

    // Loading overlay methods
    showLoading() {
        if (this.loadingOverlay) {
            this.loadingOverlay.classList.add('active');
        }
    }

    hideLoading() {
        if (this.loadingOverlay) {
            this.loadingOverlay.classList.remove('active');
        }
    }

    // Voice Recording Methods
    toggleVoiceRecording() {
        if (!this.recognition) {
            this.showNotification('Speech recognition not supported in this browser', 'error');
            return;
        }

        if (this.isRecording) {
            this.stopVoiceRecording();
        } else {
            this.startVoiceRecording();
        }
    }

    startVoiceRecording() {
        if (this.isRecording || !this.recognition) return;

        try {
            this.isRecording = true;
            this.recognition.start();
            console.log('🎤 Starting voice recording...');
        } catch (error) {
            console.error('🎤 Error starting voice recording:', error);
            this.showNotification('Failed to start voice recording', 'error');
            this.isRecording = false;
        }
    }

    stopVoiceRecording() {
        if (!this.isRecording && !this.isListening) return;

        try {
            if (this.recognition) {
                this.recognition.stop();
            }

            this.isRecording = false;
            this.isListening = false;
            this.voiceBtn.classList.remove('voice-recording');
            this.hideVoiceModal();

            console.log('🎤 Voice recording stopped');

            // Auto-send if there's content and user preference
            if (this.messageInput.value.trim()) {
                this.updateSendButton();
                this.messageInput.focus();
            }
        } catch (error) {
            console.error('🎤 Error stopping voice recording:', error);
        }
    }

    showVoiceModal() {
        this.voiceModal.classList.add('active');
        document.body.style.overflow = 'hidden';
    }

    hideVoiceModal() {
        this.voiceModal.classList.remove('active');
        document.body.style.overflow = 'auto';
    }

    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message || this.isTyping) return;

        // Use streaming for better UX
        await this.sendMessageWithStreaming(message);
    }

    // Conversation Management
    async loadConversations() {
        try {
            const response = await fetch(`${this.apiBase}/conversations`);
            const data = await response.json();
            if (response.ok) {
                this.conversations = data.conversations;
                this.updateConversationsList();
            }
        } catch (error) {
            console.error('Error loading conversations:', error);
        }
    }

    updateConversationsList() {
        const conversationsList = document.getElementById('conversationsList');
        if (!conversationsList) return;

        conversationsList.innerHTML = '';
        this.conversations.forEach(conversation => {
            const conversationItem = document.createElement('div');
            conversationItem.className = 'conversation-item';
            if (conversation.id === this.currentConversationId) {
                conversationItem.classList.add('active');
            }

            conversationItem.innerHTML = `
                <div class="conversation-info">
                    <div class="conversation-title">${conversation.title}</div>
                    <div class="conversation-meta">${conversation.message_count} messages • ${new Date(conversation.last_updated).toLocaleDateString()}</div>
                </div>
                <div class="conversation-actions">
                    <button class="btn-icon" onclick="chatbot.loadConversation('${conversation.id}')" title="Load">
                        <i class="fas fa-folder-open"></i>
                    </button>
                    <button class="btn-icon" onclick="chatbot.deleteConversation('${conversation.id}')" title="Delete">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            `;

            conversationsList.appendChild(conversationItem);
        });
    }

    async sendMessageWithStreaming(message) {
        this.addMessage(message, 'user');
        this.messageInput.value = '';
        this.autoResizeInput();
        this.updateSendButton();
        this.showTypingIndicator();

        try {
            const response = await fetch(`${this.apiBase}/chat/stream`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message })
            });

            const data = await response.json();
            this.hideTypingIndicator();

            if (response.ok) {
                const botMessage = data.response;
                const hasContext = data.has_context;

                // Add bot message without any prefix
                this.addMessage(botMessage, 'bot');

                // Only show context notification when document context is actually used
                if (hasContext && botMessage.includes('document')) {
                    this.showNotification('Response based on uploaded documents', 'info');
                }
            } else {
                throw new Error(data.error || 'Failed to get response');
            }
        } catch (error) {
            this.hideTypingIndicator();
            this.addMessage(`❌ Error: ${error.message}`, 'bot');
            this.showNotification(`Error: ${error.message}`, 'error');
        }
    }

    addMessage(content, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender} fade-in`;

        const timestamp = new Date().toLocaleTimeString([], { 
            hour: '2-digit', 
            minute: '2-digit' 
        });

        // Create avatar HTML for both user and bot
        let avatarHtml = '';
        if (sender === 'user') {
            // User avatar with 'U'
            avatarHtml = `
                <div class="message-avatar user-avatar">
                    <span class="avatar-text">U</span>
                </div>
            `;
        } else {
            // Bot avatar with image
            avatarHtml = `
                <div class="message-avatar bot-avatar">
                    <img src="juno avatar.jpg" alt="Bot Avatar">
                </div>
            `;
        }

        messageDiv.innerHTML = `
            ${avatarHtml}
            <div class="message-content">
                <div class="message-bubble">
                    <div class="message-text">${this.formatMessage(content)}</div>
                </div>
                <div class="message-time">${timestamp}</div>
            </div>
        `;

        this.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();

        // Hide welcome message after first user message
        if (sender === 'user' && this.welcomeMessage) {
            this.welcomeMessage.style.display = 'none';
        }
    }

    formatMessage(text) {
        return text
            .replace(/\n/g, '<br>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>');
    }

    showTypingIndicator() {
        this.isTyping = true;
        this.sendBtn.disabled = true;

        const typingDiv = document.createElement('div');
        typingDiv.className = 'message bot typing fade-in';
        typingDiv.innerHTML = `
            <div class="message-avatar bot-avatar">
                <img src="juno avatar.jpg" alt="Bot Avatar">
            </div>
            <div class="message-content">
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        `;

        this.messagesContainer.appendChild(typingDiv);
        this.scrollToBottom();
    }

    hideTypingIndicator() {
        this.isTyping = false;
        this.sendBtn.disabled = false;
        const typingIndicator = this.messagesContainer.querySelector('.typing');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }

    handleFileUpload(event) {
        const file = event.target.files[0];
        if (file) {
            this.processFile(file);
        }
    }

    async processFile(file) {
        if (file.type !== 'application/pdf') {
            this.showNotification('Please select a PDF file', 'error');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);
        this.showLoading();
        this.showNotification(`Uploading ${file.name}...`, 'info');

        try {
            const response = await fetch(`${this.apiBase}/upload`, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            this.hideLoading();

            if (response.ok) {
                this.hasDocuments = true;
                this.updateDocumentStatus(`Processed: ${data.filename}`, true);

                // Hide welcome message when document is uploaded
                if (this.welcomeMessage) {
                    this.welcomeMessage.style.display = 'none';
                }

                this.addMessage(
                    `📄 **Document Processed Successfully**\n\n**File:** ${data.filename}\n**Length:** ${data.text_length.toLocaleString()} characters\n\n**Summary:**\n${data.summary}`,
                    'bot'
                );
                this.showNotification('Document processed successfully!', 'success');
            } else {
                throw new Error(data.error || 'Failed to upload file');
            }

        } catch (error) {
            this.hideLoading();
            this.showNotification(`Upload failed: ${error.message}`, 'error');
        }

        this.fileInput.value = '';
    }

    updateDocumentStatus(message, hasDocuments) {
        const iconClass = hasDocuments ? 'fas fa-file-pdf' : 'fas fa-file-pdf';
        const color = hasDocuments ? 'var(--success-color)' : 'var(--text-muted)';
        this.documentStatus.innerHTML = `<i class="${iconClass}" style="color: ${color};"></i><span>${message}</span>`;
    }

    async showMemory() {
        try {
            const response = await fetch(`${this.apiBase}/memory`);
            const data = await response.json();

            let memoryHtml = `**Session ID:** ${data.session_id}
**Chat History:** ${data.chat_history_length} messages
**Documents Loaded:** ${data.has_vectorstore ? 'Yes' : 'No'}

**Memory Data:**
${JSON.stringify(data.memory, null, 2)}`;

            this.memoryContent.innerHTML = `<pre>${memoryHtml}</pre>`;
            this.memoryModal.classList.add('active');
        } catch (error) {
            this.showNotification(`Failed to load memory: ${error.message}`, 'error');
        }
    }

    hideMemory() {
        this.memoryModal.classList.remove('active');
    }

    async clearSession() {
        if (!confirm('Are you sure you want to clear the current session? This will remove all chat history and uploaded documents.')) {
            return;
        }

        try {
            const response = await fetch(`${this.apiBase}/clear`, {
                method: 'POST'
            });

            const data = await response.json();

            if (response.ok) {
                this.messagesContainer.innerHTML = `
                    <div class="welcome-message" id="welcomeMessage">
                        <div class="welcome-icon">
                            <img src="juno logo.jpg" alt="Juno Logo" class="welcome-logo">
                        </div>
                        <h2>Welcome to AI Assistant</h2>
                        <p>Upload documents, ask questions, or start a conversation!</p>

                        <div class="feature-grid">
                            <div class="feature-item">
                                <i class="fas fa-file-upload"></i>
                                <span>Upload PDFs</span>
                            </div>
                            <div class="feature-item">
                                <i class="fas fa-brain"></i>
                                <span>Smart Memory</span>
                            </div>
                            <div class="feature-item">
                                <i class="fas fa-globe"></i>
                                <span>Web Scraping</span>
                            </div>
                            <div class="feature-item">
                                <i class="fas fa-microphone"></i>
                                <span>Voice Input</span>
                            </div>
                        </div>

                        <div class="quick-actions">
                            <button class="quick-btn" onclick="chatbot.fillInput('Summarize this document')">
                                <i class="fas fa-file-text"></i> Summarize Doc
                            </button>
                            <button class="quick-btn" onclick="chatbot.fillInput('Extract key insights from this content')">
                                <i class="fas fa-lightbulb"></i> Key Insights
                            </button>
                            <button class="quick-btn" onclick="chatbot.fillInput('What is the main topic of this document?')">
                                <i class="fas fa-bullseye"></i> Main Topic
                            </button>
                            <button class="quick-btn" onclick="chatbot.fillInput('List the important points mentioned')">
                                <i class="fas fa-list"></i> Key Points
                            </button>
                        </div>
                    </div>
                `;

                // Re-initialize welcome message reference
                this.welcomeMessage = document.getElementById('welcomeMessage');

                this.hasDocuments = false;
                this.updateDocumentStatus('No documents loaded', false);
                this.showNotification('Session cleared successfully!', 'success');
                this.currentConversationId = null;
                this.loadConversations();
            } else {
                throw new Error(data.error || 'Failed to clear session');
            }

        } catch (error) {
            this.showNotification(`Clear failed: ${error.message}`, 'error');
        }
    }

    async scrapeWebsite() {
        const url = prompt('Enter the website URL to scrape:');
        if (!url) return;

        this.showLoading();
        this.showNotification('Scraping website...', 'info');

        try {
            const response = await fetch(`${this.apiBase}/scrape`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ url: url })
            });

            const data = await response.json();
            this.hideLoading();

            if (response.ok) {
                this.hasDocuments = true;
                this.updateDocumentStatus(`Scraped: ${data.url}`, true);

                // Hide welcome message when content is scraped
                if (this.welcomeMessage) {
                    this.welcomeMessage.style.display = 'none';
                }

                this.addMessage(
                    `🌐 **Website Scraped Successfully**\n\n**URL:** ${data.url}\n**Content Length:** ${data.content_length.toLocaleString()} characters\n\n**Summary:**\n${data.summary}`,
                    'bot'
                );
                this.showNotification('Website scraped successfully!', 'success');
            } else {
                throw new Error(data.error || 'Failed to scrape website');
            }

        } catch (error) {
            this.hideLoading();
            this.showNotification(`Scraping failed: ${error.message}`, 'error');
        }
    }

    closeSidebar() {
        this.sidebar.classList.remove('active');
    }

    autoResizeInput() {
        this.messageInput.style.height = 'auto';
        this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 120) + 'px';
    }

    updateSendButton() {
        const hasText = this.messageInput.value.trim().length > 0;
        this.sendBtn.disabled = !hasText || this.isTyping;
    }

    scrollToBottom() {
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }

    showNotification(message, type = 'info', duration = 4000) {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;

        document.body.appendChild(notification);

        // Auto-hide notifications except errors
        if (type !== 'error') {
            setTimeout(() => {
                if (document.body.contains(notification)) {
                    notification.remove();
                }
            }, duration);
        } else {
            setTimeout(() => {
                if (document.body.contains(notification)) {
                    notification.remove();
                }
            }, 7000); // Errors stay longer
        }
    }

    applyInitialTheme() {
        const savedTheme = localStorage.getItem('theme') || 'light';
        document.body.setAttribute('data-theme', savedTheme);
        this.updateThemeIcon(savedTheme);
    }

    toggleTheme() {
        const currentTheme = document.body.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

        document.body.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        this.updateThemeIcon(newTheme);
    }

    updateThemeIcon(theme) {
        const icon = this.themeToggleBtn.querySelector('i');
        icon.className = theme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
    }

    // Conversation management methods
    async saveConversation() {
        const title = prompt('Enter conversation title:') || `Chat ${new Date().toLocaleString()}`;

        try {
            const response = await fetch(`${this.apiBase}/conversations`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ title: title })
            });

            const data = await response.json();

            if (response.ok) {
                this.showNotification('Conversation saved successfully!', 'success');
                this.loadConversations();
            } else {
                throw new Error(data.error || 'Failed to save conversation');
            }

        } catch (error) {
            this.showNotification(`Save failed: ${error.message}`, 'error');
        }
    }

    async loadConversation(conversationId) {
        try {
            const response = await fetch(`${this.apiBase}/conversations/${conversationId}`);
            const data = await response.json();

            if (response.ok) {
                this.currentConversationId = conversationId;
                this.messagesContainer.innerHTML = '';

                // Hide welcome message when loading conversation
                if (this.welcomeMessage) {
                    this.welcomeMessage.style.display = 'none';
                }

                // Load messages
                data.conversation.messages.forEach(msg => {
                    this.addMessage(msg.user, 'user');
                    this.addMessage(msg.bot, 'bot');
                });

                this.showNotification('Conversation loaded successfully!', 'success');
                this.updateConversationsList();
                this.closeSidebar();
            } else {
                throw new Error(data.error || 'Failed to load conversation');
            }

        } catch (error) {
            this.showNotification(`Load failed: ${error.message}`, 'error');
        }
    }

    async deleteConversation(conversationId) {
        if (!confirm('Are you sure you want to delete this conversation?')) {
            return;
        }

        try {
            const response = await fetch(`${this.apiBase}/conversations/${conversationId}`, {
                method: 'DELETE'
            });

            const data = await response.json();

            if (response.ok) {
                this.showNotification('Conversation deleted successfully!', 'success');
                this.loadConversations();

                if (this.currentConversationId === conversationId) {
                    this.currentConversationId = null;
                }
            } else {
                throw new Error(data.error || 'Failed to delete conversation');
            }

        } catch (error) {
            this.showNotification(`Delete failed: ${error.message}`, 'error');
        }
    }

    newConversation() {
        this.currentConversationId = null;
        this.messagesContainer.innerHTML = `
            <div class="welcome-message" id="welcomeMessage">
                <div class="welcome-icon">
                    <img src="juno logo.jpg" alt="Juno Logo" class="welcome-logo">
                </div>
                <h2>Welcome to AI Assistant</h2>
                <p>Upload documents, ask questions, or start a conversation!</p>

                <div class="feature-grid">
                    <div class="feature-item">
                        <i class="fas fa-file-upload"></i>
                        <span>Upload PDFs</span>
                    </div>
                    <div class="feature-item">
                        <i class="fas fa-brain"></i>
                        <span>Smart Memory</span>
                    </div>
                    <div class="feature-item">
                        <i class="fas fa-globe"></i>
                        <span>Web Scraping</span>
                    </div>
                    <div class="feature-item">
                        <i class="fas fa-microphone"></i>
                        <span>Voice Input</span>
                    </div>
                </div>

                <div class="quick-actions">
                    <button class="quick-btn" onclick="chatbot.fillInput('Summarize this document')">
                        <i class="fas fa-file-text"></i> Summarize Doc
                    </button>
                    <button class="quick-btn" onclick="chatbot.fillInput('Extract key insights from this content')">
                        <i class="fas fa-lightbulb"></i> Key Insights
                    </button>
                    <button class="quick-btn" onclick="chatbot.fillInput('What is the main topic of this document?')">
                        <i class="fas fa-bullseye"></i> Main Topic
                    </button>
                    <button class="quick-btn" onclick="chatbot.fillInput('List the important points mentioned')">
                        <i class="fas fa-list"></i> Key Points
                    </button>
                </div>
            </div>
        `;

        // Re-initialize welcome message reference
        this.welcomeMessage = document.getElementById('welcomeMessage');

        this.updateConversationsList();
        this.closeSidebar();
        this.showNotification('New conversation started!', 'success');
    }
}

// Initialize the chatbot when the page loads
let chatbot;
document.addEventListener('DOMContentLoaded', () => {
    chatbot = new ChatbotUI();
});