function ChatBubble({ message }) {
  const isUser = message.isUser;

  return (
    <div
      className={`mb-3 flex ${isUser ? 'justify-end' : 'justify-start'}`}
      dir="rtl"
    >
      {/* Bot avatar */}
      {!isUser && (
        <div className="w-8 h-8 rounded-full bg-[#A94438] flex items-center justify-center text-white text-xs font-bold ml-2 mt-1">
          AI
        </div>
      )}

      <div
        className={`
          max-w-xs lg:max-w-md px-4 py-2 rounded-2xl shadow-sm
          ${isUser
            ? 'bg-[#8B2E2E] text-white rounded-tr-none'
            : 'bg-[#F5E6D3] text-gray-800 rounded-tl-none'
          }
        `}
        style={{ direction: 'rtl', textAlign: 'right' }}
      >
        {/* Preserve line breaks in multi-line answers */}
        <p className="text-sm leading-relaxed whitespace-pre-wrap">{message.text}</p>
        <div className={`text-xs mt-1 ${isUser ? 'text-blue-200' : 'text-gray-400'}`}>
          {message.timestamp}
        </div>
      </div>

      {/* User avatar */}
      {isUser && (
        <div className="w-8 h-8 rounded-full bg-gray-300 flex items-center justify-center text-gray-600 text-xs font-bold flex-shrink-0 mr-2 mt-1">
          أنت
        </div>
      )}
    </div>
  );
}

export default ChatBubble;
