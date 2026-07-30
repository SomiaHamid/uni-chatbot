import { useEffect, useState } from 'react';

function Home() {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    // Trigger animation after mount
    setTimeout(() => setVisible(true), 100);
  }, []);

  return (
    <div className="min-h-[90vh] flex flex-col items-center justify-center bg-[#FAF7F2] overflow-hidden" dir="rtl">

      {/* Hero Section */}
      <div className={`w-full max-w-4xl px-4 transition-all duration-1000 ${visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'}`}>

        {/* University Banner */}
        <div className="relative rounded-2xl overflow-hidden shadow-2xl mb-8 group">
          <img
            src="/pic.png"
            alt="جامعة السودان للعلوم والتكنولوجيا"
            className="w-full h-48 md:h-64 object-cover object-[80%_top] transition-transform duration-700 group-hover:scale-105" />

          <div className="absolute inset-0 bg-gradient-to-t from-[#8B2E2E]/80 via-[#8B2E2E]/20 to-transparent" />

          <div className="absolute bottom-4 right-4 bg-white/10 backdrop-blur-md border border-white/30 rounded-xl px-4 py-2 text-white text-right">
            <p className="text-xs font-light opacity-90">
              Sudan University of Science & Technology
            </p>
            <p className="text-base font-bold">
              جامعة السودان للعلوم والتكنولوجيا
            </p>
          </div>
        </div>

        {/* College Tag */}
        <div className={`flex justify-center mb-6 transition-all duration-1000 delay-200 ${visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-6'}`}>
          <div className="inline-flex items-center gap-2 bg-[#8B2E2E]/10 border border-[#8B2E2E]/30 text-[#8B2E2E] rounded-full px-5 py-2 text-sm font-semibold">
            <span>تم تطوير هذا النظام بواسطة طلاب من قسم الحاسوب ونظم المعلومات
              بكلية علوم الحاسوب وتقانة المعلومات</span>
          </div>
        </div>

        {/* Main card */}
        <div className={`bg-white rounded-2xl shadow-xl p-8 text-center transition-all duration-1000 delay-300 ${visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-6'}`}>

          {/* Animated bot icon */}
          <div className="flex justify-center mb-4">
            <div className="w-20 h-20 rounded-full bg-gradient-to-br from-[#8B2E2E] to-[#A94438] flex items-center justify-center shadow-lg animate-bounce">
            </div>
          </div>

          <h2 className="text-3xl font-bold mb-3 text-[#8B2E2E]">
            المساعد الجامعي الذكي
          </h2>
          <p className="text-gray-500 text-sm mb-1 font-medium">
            College of Computer Science & Information Technology
          </p>
          <p className="text-gray-700 text-base mb-6 leading-relaxed">
            دليلك الذكي للإستفسارات الأكاديمية ،   يجيب على الأسئلة المتعلقة باللوائح الأكاديمية، الامتحانات، النتائج، الإجراءات<br />
            اسأل واستلم الإجابة فوراً بكل سهولة
          </p>

          {/* Features row */}
          <div className="grid grid-cols-3 gap-3 mb-7">
            {[
              { icon: '📋', label: 'اللوائح الأكاديمية' },
              { icon: '🎓', label: 'الإجراءات الجامعية' },
              { icon: '💡', label: 'إجابات فورية' },
            ].map((f, i) => (
              <div key={i} className="bg-[#FAF7F2] rounded-xl p-3 text-center border border-[#E5D3C0]">
                <div className="text-2xl mb-1">{f.icon}</div>
                <p className="text-xs text-gray-600 font-medium">{f.label}</p>
              </div>
            ))}
          </div>

          <a
            href="/chat"
            className="inline-block bg-gradient-to-r from-[#8B2E2E] to-[#A94438] text-white px-8 py-3 rounded-xl font-bold text-base hover:shadow-lg hover:scale-105 transition-all duration-300"
          >
            ابدأ المحادثة الآن
          </a>
        </div>

      </div>
    </div>
  );
}

export default Home;