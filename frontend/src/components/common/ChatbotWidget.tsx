import React, { useState, useRef, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import apiClient from '../../services/api-client';

/* ═══════════════════════════════════════════════════════════
   MedAgentix RAG Chatbot Widget
   ─────────────────────────────────────────────────────────
   A floating chat-bubble icon fixed to the bottom-right of
   every page. Clicking it reveals a sleek slide-up panel
   connected to the Flask /api/v1/chatbot RAG endpoint.
   ═══════════════════════════════════════════════════════════ */

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  sources?: { disease: string; category: string; severity: string; relevance: number }[];
}

/* ─── SVG Icon Components ─── */
const BotIcon = ({ className = '' }: { className?: string }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 8V4H8" />
    <rect width="16" height="12" x="4" y="8" rx="2" />
    <path d="M2 14h2" />
    <path d="M20 14h2" />
    <path d="M15 13v2" />
    <path d="M9 13v2" />
  </svg>
);

const SendIcon = ({ className = '' }: { className?: string }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="m22 2-7 20-4-9-9-4Z" />
    <path d="M22 2 11 13" />
  </svg>
);

const CloseIcon = ({ className = '' }: { className?: string }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M18 6 6 18" />
    <path d="m6 6 12 12" />
  </svg>
);

const MinimizeIcon = ({ className = '' }: { className?: string }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M8 3v3a2 2 0 0 1-2 2H3" />
    <path d="M21 8h-3a2 2 0 0 1-2-2V3" />
    <path d="M3 16h3a2 2 0 0 1 2 2v3" />
    <path d="M16 21v-3a2 2 0 0 1 2-2h3" />
  </svg>
);

/* ─── Typing Indicator ─── */
const TypingIndicator = () => (
  <div className="flex items-center gap-1 px-4 py-3">
    <div className="flex items-center gap-1">
      <span className="w-2 h-2 rounded-full bg-slate-400 animate-[bounce_1.4s_ease-in-out_infinite]" style={{ animationDelay: '0ms' }} />
      <span className="w-2 h-2 rounded-full bg-slate-400 animate-[bounce_1.4s_ease-in-out_infinite]" style={{ animationDelay: '200ms' }} />
      <span className="w-2 h-2 rounded-full bg-slate-400 animate-[bounce_1.4s_ease-in-out_infinite]" style={{ animationDelay: '400ms' }} />
    </div>
    <span className="text-[10px] text-slate-400 ml-2 font-semibold">MedAgentix is thinking…</span>
  </div>
);

/* ─── Markdown-lite renderer (bold, italic, bullet, headings) ─── */
function renderMarkdown(text: string): React.ReactNode[] {
  const lines = text.split('\n');
  const elements: React.ReactNode[] = [];

  lines.forEach((line, lineIdx) => {
    let processed: React.ReactNode;

    // Heading (###, ##, #)
    if (line.startsWith('### ')) {
      processed = <h4 key={lineIdx} className="text-xs font-extrabold text-slate-800 mt-2 mb-0.5">{renderInline(line.slice(4))}</h4>;
    } else if (line.startsWith('## ')) {
      processed = <h3 key={lineIdx} className="text-sm font-extrabold text-slate-800 mt-2 mb-0.5">{renderInline(line.slice(3))}</h3>;
    } else if (line.startsWith('# ')) {
      processed = <h2 key={lineIdx} className="text-sm font-extrabold text-slate-800 mt-2 mb-0.5">{renderInline(line.slice(2))}</h2>;
    }
    // Bullet list
    else if (line.startsWith('- ') || line.startsWith('* ')) {
      processed = (
        <div key={lineIdx} className="flex items-start gap-1.5 ml-1">
          <span className="w-1 h-1 rounded-full bg-slate-400 mt-1.5 shrink-0" />
          <span>{renderInline(line.slice(2))}</span>
        </div>
      );
    }
    // Numbered list
    else if (/^\d+\.\s/.test(line)) {
      const match = line.match(/^(\d+)\.\s(.*)$/);
      if (match) {
        processed = (
          <div key={lineIdx} className="flex items-start gap-1.5 ml-1">
            <span className="text-[10px] font-bold text-primary mt-0.5 shrink-0">{match[1]}.</span>
            <span>{renderInline(match[2])}</span>
          </div>
        );
      } else {
        processed = <p key={lineIdx}>{renderInline(line)}</p>;
      }
    }
    // Horizontal rule
    else if (line.trim() === '---' || line.trim() === '***') {
      processed = <hr key={lineIdx} className="border-slate-200 my-1.5" />;
    }
    // Empty line → spacer
    else if (line.trim() === '') {
      processed = <div key={lineIdx} className="h-1" />;
    }
    // Normal paragraph
    else {
      processed = <p key={lineIdx}>{renderInline(line)}</p>;
    }

    elements.push(processed);
  });

  return elements;
}

/* Inline bold / italic / code */
function renderInline(text: string): React.ReactNode {
  // Process **bold**, *italic*, `code`
  const parts: React.ReactNode[] = [];
  const regex = /(\*\*(.+?)\*\*|\*(.+?)\*|`(.+?)`)/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }
    if (match[2]) {
      parts.push(<strong key={match.index} className="font-bold text-slate-800">{match[2]}</strong>);
    } else if (match[3]) {
      parts.push(<em key={match.index} className="italic">{match[3]}</em>);
    } else if (match[4]) {
      parts.push(<code key={match.index} className="bg-slate-100 text-[10px] px-1 py-0.5 rounded font-mono">{match[4]}</code>);
    }
    lastIndex = regex.lastIndex;
  }

  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }

  return parts.length > 0 ? <>{parts}</> : text;
}

/* ════════════════════════════════════════════════════════════
   Main ChatbotWidget Component
   ════════════════════════════════════════════════════════════ */
export const ChatbotWidget: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'welcome',
      role: 'assistant',
      content: '👋 **Hello! I am the MedAgentix AI Assistant.**\n\nI can help you with:\n- Medical questions from our knowledge base\n- Information about the MedAgentix platform\n- Explain how our diagnostic pipeline works\n\nAsk me anything!',
      timestamp: new Date(),
    },
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showSources, setShowSources] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  // Focus input when panel opens
  useEffect(() => {
    if (isOpen) {
      setTimeout(() => inputRef.current?.focus(), 300);
    }
  }, [isOpen]);

  const sendMessage = useCallback(async () => {
    const text = inputValue.trim();
    if (!text || isLoading) return;

    const userMsg: ChatMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: text,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMsg]);
    setInputValue('');
    setIsLoading(true);

    try {
      const response = await apiClient.post('/chatbot', { message: text });

      if (response.data?.success) {
        const botMsg: ChatMessage = {
          id: `bot-${Date.now()}`,
          role: 'assistant',
          content: response.data.answer,
          timestamp: new Date(),
          sources: response.data.sources,
        };
        setMessages((prev) => [...prev, botMsg]);
      } else {
        const errMsg: ChatMessage = {
          id: `err-${Date.now()}`,
          role: 'system',
          content: `⚠️ ${response.data?.error || 'Something went wrong. Please try again.'}`,
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, errMsg]);
      }
    } catch (err: any) {
      const errorText =
        err.response?.data?.error || 'Could not reach the chatbot service. Make sure the Flask backend is running.';
      const errMsg: ChatMessage = {
        id: `err-${Date.now()}`,
        role: 'system',
        content: `⚠️ ${errorText}`,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errMsg]);
    } finally {
      setIsLoading(false);
    }
  }, [inputValue, isLoading]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const sendQuickMessage = useCallback(async (text: string) => {
    if (isLoading) return;

    const userMsg: ChatMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: text,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMsg]);
    setInputValue('');
    setIsLoading(true);

    try {
      const response = await apiClient.post('/chatbot', { message: text });

      if (response.data?.success) {
        const botMsg: ChatMessage = {
          id: `bot-${Date.now()}`,
          role: 'assistant',
          content: response.data.answer,
          timestamp: new Date(),
          sources: response.data.sources,
        };
        setMessages((prev) => [...prev, botMsg]);
      } else {
        const errMsg: ChatMessage = {
          id: `err-${Date.now()}`,
          role: 'system',
          content: `⚠️ ${response.data?.error || 'Something went wrong. Please try again.'}`,
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, errMsg]);
      }
    } catch (err: any) {
      const errorText =
        err.response?.data?.error || 'Could not reach the chatbot service. Make sure the Flask backend is running.';
      const errMsg: ChatMessage = {
        id: `err-${Date.now()}`,
        role: 'system',
        content: `⚠️ ${errorText}`,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errMsg]);
    } finally {
      setIsLoading(false);
    }
  }, [isLoading]);

  const quickQuestions = [
    'What is MedAgentix AI?',
    'How does the diagnostic pipeline work?',
    'Tell me about cardiac diseases',
  ];

  return (
    <>
      {/* ═══ Floating Chat Bubble ═══ */}
      <AnimatePresence>
        {!isOpen && (
          <motion.button
            id="chatbot-fab"
            initial={{ scale: 0, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0, opacity: 0 }}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            onClick={() => setIsOpen(true)}
            className="fixed bottom-6 right-6 z-[9999] w-14 h-14 rounded-full bg-gradient-to-br from-[#0F4C81] to-[#14B8A6] text-white shadow-lg shadow-primary/30 flex items-center justify-center hover:shadow-xl hover:shadow-primary/40 transition-shadow duration-300"
            aria-label="Open MedAgentix Chatbot"
          >
            <BotIcon className="w-7 h-7" />

            {/* Notification Pulse */}
            <span className="absolute -top-0.5 -right-0.5 w-3.5 h-3.5 bg-emerald-400 rounded-full border-2 border-white animate-pulse" />
          </motion.button>
        )}
      </AnimatePresence>

      {/* ═══ Chat Panel ═══ */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: 40, scale: 0.92 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 40, scale: 0.92 }}
            transition={{ type: 'spring', stiffness: 300, damping: 28 }}
            className="fixed bottom-6 right-6 z-[9999] w-[400px] h-[600px] max-h-[calc(100vh-48px)] bg-white rounded-2xl shadow-2xl shadow-slate-900/15 border border-slate-200/80 flex flex-col overflow-hidden"
            style={{ maxWidth: 'calc(100vw - 48px)' }}
          >
            {/* ── Header ── */}
            <div className="bg-gradient-to-r from-[#0F4C81] to-[#0d6b6e] px-5 py-3.5 flex items-center justify-between shrink-0">
              <div className="flex items-center gap-3">
                <div className="w-9 h-9 rounded-xl bg-white/15 backdrop-blur-sm flex items-center justify-center">
                  <BotIcon className="w-5 h-5 text-white" />
                </div>
                <div>
                  <h3 className="text-sm font-bold text-white tracking-tight">MedAgentix AI</h3>
                  <div className="flex items-center gap-1.5">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                    <span className="text-[10px] text-white/70 font-medium">RAG Knowledge Engine</span>
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-1">
                <button
                  onClick={() => setIsOpen(false)}
                  className="p-1.5 rounded-lg hover:bg-white/10 transition text-white/80 hover:text-white"
                  aria-label="Minimize chatbot"
                >
                  <MinimizeIcon className="w-4 h-4" />
                </button>
                <button
                  onClick={() => setIsOpen(false)}
                  className="p-1.5 rounded-lg hover:bg-white/10 transition text-white/80 hover:text-white"
                  aria-label="Close chatbot"
                >
                  <CloseIcon className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* ── Messages Area ── */}
            <div className="flex-1 overflow-y-auto px-4 py-3 space-y-3 scroll-smooth" style={{ scrollbarWidth: 'thin' }}>
              {messages.map((msg) => (
                <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div
                    className={`max-w-[85%] rounded-2xl px-3.5 py-2.5 text-[12px] leading-relaxed ${
                      msg.role === 'user'
                        ? 'bg-[#0F4C81] text-white rounded-br-md'
                        : msg.role === 'system'
                        ? 'bg-amber-50 text-amber-800 border border-amber-200 rounded-bl-md'
                        : 'bg-slate-50 text-slate-700 border border-slate-100 rounded-bl-md'
                    }`}
                  >
                    {msg.role === 'user' ? (
                      <p className="whitespace-pre-wrap">{msg.content}</p>
                    ) : (
                      <div className="space-y-0.5">{renderMarkdown(msg.content)}</div>
                    )}

                    {/* Source Citations Toggle */}
                    {msg.sources && msg.sources.length > 0 && (
                      <div className="mt-2 pt-1.5 border-t border-slate-200/60">
                        <button
                          onClick={() => setShowSources(showSources === msg.id ? null : msg.id)}
                          className="text-[10px] font-bold text-primary hover:underline flex items-center gap-1"
                        >
                          <span>{showSources === msg.id ? '▾' : '▸'}</span>
                          {msg.sources.length} source{msg.sources.length > 1 ? 's' : ''} referenced
                        </button>

                        <AnimatePresence>
                          {showSources === msg.id && (
                            <motion.div
                              initial={{ height: 0, opacity: 0 }}
                              animate={{ height: 'auto', opacity: 1 }}
                              exit={{ height: 0, opacity: 0 }}
                              className="overflow-hidden"
                            >
                              <div className="mt-1.5 space-y-1">
                                {msg.sources.map((src, sIdx) => (
                                  <div
                                    key={sIdx}
                                    className="bg-white border border-slate-200 rounded-lg px-2.5 py-1.5 text-[10px]"
                                  >
                                    <span className="font-bold text-slate-700">{src.disease}</span>
                                    <span className="text-slate-400 ml-1.5">
                                      {src.category} · {src.severity} · {(src.relevance * 100).toFixed(0)}%
                                    </span>
                                  </div>
                                ))}
                              </div>
                            </motion.div>
                          )}
                        </AnimatePresence>
                      </div>
                    )}

                    {/* Timestamp */}
                    <p
                      className={`text-[9px] mt-1.5 ${
                        msg.role === 'user' ? 'text-white/50 text-right' : 'text-slate-400'
                      }`}
                    >
                      {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </p>
                  </div>
                </div>
              ))}

              {/* Typing indicator */}
              {isLoading && (
                <div className="flex justify-start">
                  <div className="bg-slate-50 border border-slate-100 rounded-2xl rounded-bl-md">
                    <TypingIndicator />
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>

            {/* ── Quick Questions (shown when few messages) ── */}
            {messages.length <= 2 && !isLoading && (
              <div className="px-4 pb-2 shrink-0">
                <p className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1.5">Quick questions</p>
                <div className="flex flex-wrap gap-1.5">
                  {quickQuestions.map((q) => (
                    <button
                      key={q}
                      onClick={() => sendQuickMessage(q)}
                      className="text-[10px] font-semibold text-primary bg-primary/5 hover:bg-primary/10 border border-primary/15 px-2.5 py-1.5 rounded-lg transition"
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* ── Input Area ── */}
            <div className="border-t border-slate-100 px-3 py-2.5 shrink-0 bg-white">
              <div className="flex items-center gap-2 bg-slate-50 border border-slate-200 rounded-xl px-3 py-1.5 focus-within:border-primary/50 focus-within:ring-2 focus-within:ring-primary/10 transition">
                <input
                  ref={inputRef}
                  type="text"
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Ask about medical conditions, symptoms…"
                  className="flex-1 bg-transparent outline-none text-xs text-slate-700 placeholder:text-slate-400"
                  disabled={isLoading}
                />
                <button
                  onClick={sendMessage}
                  disabled={!inputValue.trim() || isLoading}
                  className="p-1.5 rounded-lg bg-[#0F4C81] text-white disabled:opacity-30 disabled:cursor-not-allowed hover:bg-[#0d3d69] active:scale-95 transition"
                  aria-label="Send message"
                >
                  <SendIcon className="w-3.5 h-3.5" />
                </button>
              </div>
              <p className="text-[9px] text-slate-400 text-center mt-1.5 font-medium">
                Powered by MedAgentix RAG · 6,700+ medical knowledge chunks
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
};

export default ChatbotWidget;
