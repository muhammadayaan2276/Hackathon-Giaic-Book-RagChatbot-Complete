import React, { useState } from 'react';
import styles from './FloatingChatWidget.module.css';
import Chatbot from './Chatbot';

export default function FloatingChatWidget() {
  const [isOpen, setIsOpen] = useState(false);

  const toggleChat = () => {
    setIsOpen(!isOpen);
  };

  return (
    <div className={styles.chatWidgetContainer}>
      {isOpen && (
        <div className={styles.chatPanel}>
          <button onClick={toggleChat} className={styles.closeButton} aria-label="Close chat">
            &times;
          </button>
          <Chatbot />
        </div>
      )}
      <button className={styles.chatButton} onClick={toggleChat} aria-label="Toggle chat">
        <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          {isOpen ? (
            <>
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </>
          ) : (
            <>
              <path d="M12 8V4H8" />
              <rect width="16" height="12" x="4" y="8" rx="2" />
              <path d="M2 14h2" />
              <path d="M20 14h2" />
              <path d="M15 13v2" />
              <path d="M9 13v2" />
            </>
          )}
        </svg>
      </button>
    </div>
  );
}
