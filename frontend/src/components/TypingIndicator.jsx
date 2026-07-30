function TypingIndicator() {
  return (
    <div className="mb-3 flex justify-start" dir="rtl">
      <div className="w-8 h-8 rounded-full bg-[#A94438] flex items-center justify-center text-white text-xs font-bold flex-shrink-0 ml-2 mt-1">        AI
      </div>
      <div className="bg-[#F5E6D3] px-4 py-3 rounded-2xl rounded-tl-none shadow-sm">        <div className="flex space-x-1 rtl:space-x-reverse">
        <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"></div>
        <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0.15s' }}></div>
        <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0.3s' }}></div>
      </div>
        <p className="text-xs text-gray-400 mt-1">جاري التفكير...</p>
      </div>
    </div>
  );
}

export default TypingIndicator;
