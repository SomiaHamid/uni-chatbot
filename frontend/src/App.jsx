import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import Chat from './pages/Chat';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-50">
        <nav className="bg-gradient-to-r from-[#8B2E2E] to-[#A94438] text-white p-4 shadow-md">
          <div className="container mx-auto flex justify-between items-center">

            <h1 className="text-xl font-bold flex items-center gap-2">
               المساعد الجامعي
            </h1>

            <div className="flex gap-6">
             <a href="/" className="hover:text-[#F5E6D3] transition">الرئيسية</a>
             <a href="/chat" className="hover:text-[#F5E6D3] transition">الدردشة</a>
           </div>

          </div>
        </nav>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/chat" element={<Chat />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;