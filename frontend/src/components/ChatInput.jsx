import { useState } from 'react';

function ChatInput({ onSend, disabled, onFocus, placeholder }) {
  const [input, setInput] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    const trimmed = input.trim();
    if (!trimmed) return;
    onSend(trimmed);
    setInput('');
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="flex gap-2 mt-1" dir="rtl">
      <input
        type="text"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={handleKeyDown}
        onFocus={onFocus}
        placeholder={placeholder || 'اكتب سؤالك هنا...'}
        className="
          flex-1 p-3 border border-[#E5D3C0] rounded-xl
          focus:outline-none focus:ring-2 focus:ring-[#A94438]
          text-right text-sm bg-white
        "
        disabled={disabled}
        autoComplete="off"
      />
      <button
        type="submit"
        className="
           bg-[#8B2E2E] text-white px-3 py-2 rounded-xl
           hover:bg-[#A94438] active:scale-95
           transition text-sm font-medium
           flex-shrink-0
        "
        disabled={disabled || !input.trim()}
      >
        {disabled ? '...' : 'إرسال'}
      </button>
    </form>
  );
}

export default ChatInput;
