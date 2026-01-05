import React, { useState, useEffect, useRef } from 'react';
import styles from './Chatbot.module.css';

export default function Chatbot() {
  const [messages, setMessages] = useState([
    {
      id: 1,
      sender: 'Assistant',
      text: 'Hello! I\'m your RAG Chatbot. How can I assist you today?',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [selectedText, setSelectedText] = useState('');
  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Handle text selection
  useEffect(() => {
    const handleSelection = () => {
      const selection = window.getSelection().toString();
      if (selection) {
        setSelectedText(selection);
      }
    };
    window.addEventListener('mouseup', handleSelection);
    return () => {
      window.removeEventListener('mouseup', handleSelection);
    };
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    // Add user message
    const userMessage = {
      id: Date.now(),
      sender: 'You',
      text: input,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await fetch('https://muhammad-ayaan-ragchatbot-backend.hf.space/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: input, selected_text: selectedText }),
      });

      const data = await response.json();

      // Add assistant message
      const assistantMessage = {
        id: Date.now() + 1,
        sender: 'Assistant',
        text: data.answer,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error fetching chat response:', error);
      const errorMessage = {
        id: Date.now() + 1,
        sender: 'Assistant',
        text: 'Sorry, something went wrong. Please try again.',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
      setSelectedText('');
    }
  };

  return (
    <div className={styles.chatContainer}>
      <div className={styles.chatHeader}>
        <h3>RAG Chatbot</h3>
      </div>

      <div className={styles.messagesContainer}>
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`${styles.message} ${msg.sender === 'You' ? styles.userMessage : styles.assistantMessage}`}
          >
            <div className={styles.messageHeader}>
              <span className={styles.sender}>{msg.sender}</span>
              <span className={styles.timestamp}>{msg.timestamp}</span>
            </div>
            <div className={styles.messageText}>{msg.text}</div>
          </div>
        ))}
        {loading && (
          <div className={`${styles.message} ${styles.assistantMessage}`}>
            <div className={styles.messageHeader}>
              <span className={styles.sender}>Assistant</span>
            </div>
            <div className={styles.typingIndicator}>
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {selectedText && (
        <div className={styles.selectedTextContainer}>
          <div className={styles.selectedTextHeader}>
            <strong>Selected Text:</strong>
          </div>
          <p className={styles.selectedText}>{selectedText}</p>
        </div>
      )}

      <form onSubmit={handleSubmit} className={styles.inputForm}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your message here..."
          className={styles.inputField}
          disabled={loading}
        />
        <button
          type="submit"
          className={styles.sendButton}
          disabled={loading || !input.trim()}
          aria-label="Send message"
        >
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
            <path d="M3.478 2.405a.75.75 0 00-.926.94l2.432 7.905H13.5a.75.75 0 010 1.5H4.984l-2.432 7.905a.75.75 0 00.926.94 60.519 60.519 0 0018.445-8.986.75.75 0 000-1.218A60.517 60.517 0 003.478 2.405z" />
          </svg>
        </button>
      </form>
    </div>
  );
}
