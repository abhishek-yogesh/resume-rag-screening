import React, { useState, useEffect, useRef } from "react";
import { sendChat } from "../api/api";
import "./Chat.css";

interface ChatProps {
  resumeId: string | null;
}

const Chat: React.FC<ChatProps> = ({ resumeId }) => {
  const [messages, setMessages] = useState<any[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto scroll to latest message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = async () => {
    if (!resumeId || !input.trim()) return;

    const userMsg = { sender: "user", text: input };
    setMessages((prev) => [...prev, userMsg]);

    setLoading(true);

    try {
      const res = await sendChat(resumeId, input);
      const aiMsg = { sender: "ai", text: res.answer || "No response" };
      setMessages((prev) => [...prev, aiMsg]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { sender: "ai", text: "⚠ Error fetching response." },
      ]);
    }

    setInput("");
    setLoading(false);
  };

  return (
    <div className="chat-container">
      <div className="chat-header">💬 Ask Questions About This Candidate</div>

      <div className="chat-messages">
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.sender}`}>
            <div className="sender-label">
              {msg.sender === "user" ? "You" : "AI Recruiter Assistant"}
            </div>
            {msg.text}
          </div>
        ))}

        <div ref={messagesEndRef}></div>
      </div>

      <div className="chat-input-bar">
        <input
          className="chat-input"
          placeholder="Type your question..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !loading && sendMessage()}
        />

        <button
          className="send-btn"
          disabled={loading}
          onClick={sendMessage}
        >
          {loading ? "..." : "Send"}
        </button>
      </div>
    </div>
  );
};

export default Chat;
