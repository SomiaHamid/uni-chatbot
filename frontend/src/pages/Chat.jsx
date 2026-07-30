import { useState, useRef, useEffect } from 'react';
import ChatBubble from '../components/ChatBubble';
import ChatInput from '../components/ChatInput';
import TypingIndicator from '../components/TypingIndicator';
import { sendMessage, fetchInitialSuggestions } from '../services/api';

function Chat() {
  const [messages, setMessages] = useState([]);
  const [suggestions, setSuggestions] = useState([]);
  const [initialSuggestions, setInitialSuggestions] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [botTyping, setBotTyping] = useState(false);

  const messagesEndRef = useRef(null);
  const hasGreeted = useRef(false);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading, botTyping]);

  const addBotMessage = (text) => {
    setMessages((prev) => [
      ...prev,
      {
        id: Date.now() + Math.random(),
        text,
        timestamp: new Date().toLocaleTimeString('ar-EG', { hour: '2-digit', minute: '2-digit' }),
        isUser: false,
      },
    ]);
  };

  // ─── Initial greeting on mount ────────────────────────────────────────────
  useEffect(() => {
    if (hasGreeted.current) return;
    hasGreeted.current = true;

    setBotTyping(true);
    setTimeout(() => {
      setBotTyping(false);
      addBotMessage(
        '  أهلاً وسهلاً! 👋\n' +
        'انا المساعد الذكي الخاص بجامعة السودان للعلوم والتكنولوجيا،متخصص في الاجابة على استفساراتك في:\n' +
        '•   اللوائح الاكاديمية\n' +
        '•   الإجراءات الجامعية \n' +
        '•   نظم الامتحانات وضوابط التخرج  \n\n' +
        ' تفضل بسؤالك وأنا هنا للمساعدتك! 😊'
      );
    }, 1200);

    fetchInitialSuggestions().then(({ suggestions: s }) => {
      setInitialSuggestions(s);
      setSuggestions(s);
    });
  }, []);

  // ─── Send message ─────────────────────────────────────────────────────────
  const handleSendMessage = async (question) => {
    if (!question.trim() || isLoading) return;

    const userMsg = {
      id: Date.now(),
      text: question,
      timestamp: new Date().toLocaleTimeString('ar-EG', { hour: '2-digit', minute: '2-digit' }),
      isUser: true,
    };
    setMessages((prev) => [...prev, userMsg]);
    setSuggestions(initialSuggestions);
    setError(null);

    await forwardToBackend(question);
  };

  const forwardToBackend = async (question) => {
    setIsLoading(true);
    try {
      const response = await sendMessage({ question });

      if (import.meta.env.DEV) {
        console.log('[Chat] response:', response);
      }

      addBotMessage(response.answer);
      setSuggestions(response.suggestions || []);
    } catch (err) {
      console.error('[Chat] error:', err);
      setError(err.message || 'حدث خطأ، حاول مرة أخرى');
      setSuggestions(initialSuggestions);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div
      className="container mx-auto p-4 max-w-2xl flex flex-col"
      style={{ height: 'calc(100vh - 64px)' }}
    >
      <div className="bg-white shadow-lg rounded-lg p-4 flex-1 overflow-y-auto mb-2">

        {messages.map((msg) => (
          <ChatBubble key={msg.id} message={msg} />
        ))}

        {(botTyping || isLoading) && <TypingIndicator />}

        {error && (
          <div className="mt-2 p-3 bg-red-50 border border-red-200 text-red-600 rounded-lg text-sm text-right">
            ⚠️ {error}
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {suggestions.length > 0 && !isLoading && (
        <div className="flex flex-wrap gap-2 mb-2 px-1">
          {suggestions.map((s, idx) => (
            <button
              key={idx}
              onClick={() => handleSendMessage(s)}
              className="
               bg-[#F5E6D3]
               hover:bg-[#EAD7C2]
               px-3 py-1.5
               rounded-full
               text-sm
               border border-[#E5D3C0]
               text-[#8B2E2E]
               transition
             "
            >
              💡 {s}
            </button>
          ))}
        </div>
      )}

      <ChatInput
        onSend={handleSendMessage}
        disabled={isLoading || botTyping}
        placeholder="اكتب سؤالك هنا..."
      />
    </div>
  );
}

export default Chat;