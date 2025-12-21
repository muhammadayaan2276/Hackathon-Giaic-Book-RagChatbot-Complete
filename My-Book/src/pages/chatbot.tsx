import React, { useState, useRef, useEffect, FormEvent } from 'react';
import Head from '@docusaurus/Head';
import Layout from '@theme/Layout';
import clsx from 'clsx';
import styles from './chatbot.module.css';
import { bookData } from '../data/bookData';

/* ================== STOP WORDS ================== */
const STOP_WORDS = [
  'what',
  'is',
  'the',
  'of',
  'in',
  'and',
  'to',
  'name',
  'tell',
  'me',
  'about',
  'please',
  'module',
  'chapter',
];

/* ================== HELPER FUNCTIONS ================== */
const extractKeywords = (text: string): string[] => {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, '')
    .split(/\s+/)
    .filter(word => word && !STOP_WORDS.includes(word));
};

const findBestMatch = (userInput: string): string => {
  const inputWords = extractKeywords(userInput);

  if (inputWords.length === 0) {
    return 'Answer not found in the book.';
  }

  let bestScore = 0;
  let bestAnswer: string | null = null;

  for (const item of bookData) {
    const questionWords = extractKeywords(item.question);

    let matchCount = 0;
    for (const word of inputWords) {
      if (questionWords.includes(word)) {
        matchCount++;
      }
    }

    if (matchCount > bestScore) {
      bestScore = matchCount;
      bestAnswer = item.answer;
    }
  }

  return bestAnswer ?? 'Answer not found in the book.';
};

/* ================== MESSAGE TYPE ================== */
type Message = {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  timestamp: string;
};

const ChatbotPage: React.FC = () => {
  const [inputValue, setInputValue] = useState('');
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 'init',
      text: 'Hello! I am your Book Assistant. Ask me anything about the book content.',
      sender: 'bot',
      timestamp: new Date().toISOString(),
    },
  ]);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const userText = inputValue.trim();
    if (!userText) return;

    const lowerText = userText.toLowerCase();

    // Add user message
    setMessages(prev => [
      ...prev,
      {
        id: crypto.randomUUID(),
        text: userText,
        sender: 'user',
        timestamp: new Date().toISOString(),
      },
    ]);

    setInputValue('');

    let reply: string;

    /* ================== FIXED RESPONSES ================== */

    // Greetings
    if (['hi', 'hello', 'hey'].includes(lowerText)) {
      reply = 'Hello! Ask me something from the book.';
    }

    // Identity questions (IMPORTANT FIX)
    else if (
      lowerText === 'who are you' ||
      lowerText === 'who are you?' ||
      lowerText === 'what are you' ||
      lowerText === 'what can you do'
    ) {
      reply =
        'I am your Book Assistant. I answer questions based only on the book content.';
    }

    // Book-based questions
    else {
      reply = findBestMatch(userText);
    }

    // Add bot message
    setMessages(prev => [
      ...prev,
      {
        id: crypto.randomUUID(),
        text: reply,
        sender: 'bot',
        timestamp: new Date().toISOString(),
      },
    ]);
  };

  return (
    <Layout>
      <Head>
        <title>Book Assistant</title>
        <meta name="description" content="Chat with the Book Assistant" />
      </Head>

      <div className={styles.chatContainer}>
        <div className={styles.chatHeader}>
          <h1>Book Assistant</h1>
          <p>Ask me anything about the book content</p>
        </div>

        <div className={styles.chatArea}>
          <div className={styles.messagesContainer}>
            {messages.map(msg => (
              <div
                key={msg.id}
                className={clsx(
                  styles.message,
                  msg.sender === 'user'
                    ? styles.userMessage
                    : styles.botMessage
                )}
              >
                <div className={styles.messageContent}>{msg.text}</div>
                <div className={styles.timestamp}>
                  {new Date(msg.timestamp).toLocaleTimeString([], {
                    hour: '2-digit',
                    minute: '2-digit',
                  })}
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          <form onSubmit={handleSubmit} className={styles.inputForm}>
            <input
              type="text"
              value={inputValue}
              onChange={e => setInputValue(e.target.value)}
              placeholder="Type your question here..."
              className={styles.inputField}
            />
            <button
              type="submit"
              className={styles.sendButton}
              disabled={!inputValue.trim()}
            >
              Send
            </button>
          </form>
        </div>
      </div>
    </Layout>
  );
};

export default ChatbotPage;
