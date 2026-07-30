import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_BACKEND_URL;

const TIMEOUT_MS = 3000000; // 5 دقايق للـ chat (عادي مع LLM)

// هيدر ثابت لكل الطلبات — بيتخطى صفحة ngrok التحذيرية
const NGROK_HEADERS = {
  "Content-Type": "application/json",
  "ngrok-skip-browser-warning": "true",
};

/**
 * Send a chat message and receive an answer + suggestions.
 */
export const sendMessage = async ({ question, user_id = "default_user" }) => {
  try {
    const response = await axios.post(
      `${API_BASE_URL}/chat`,
      { question, user_id },
      {
        headers: NGROK_HEADERS, // ✅ مضاف
        timeout: TIMEOUT_MS,
      }
    );

    const data = response.data;

    if (import.meta.env.DEV) {
      console.log("[API] /chat response:", data);
    }

    return {
      answer:
        typeof data.answer === "string"
          ? data.answer
          : "لم يصل رد من الخادم",
      suggestions: Array.isArray(data.suggestions) ? data.suggestions : [],
    };
  } catch (error) {
    if (import.meta.env.DEV) {
      console.error("[API] /chat error:", error);
    }

    if (!error.response && error.message?.includes("timeout")) {
      throw new Error("انتهت مهلة الطلب، حاول مرة أخرى 🔄");
    }
    if (error.response) {
      throw new Error(`خطأ في الخادم: ${error.response.status}`);
    }
    throw new Error("تعذّر الوصول إلى الخادم، تحقق من اتصالك");
  }
};

/**
 * Fetch initial welcome message + suggestions when chat opens.
 */
export const fetchInitialSuggestions = async () => {
  try {
    const response = await axios.get(
      `${API_BASE_URL}/start`,
      {
        headers: NGROK_HEADERS, // ✅ مضاف
        timeout: 30000,         // ✅ 30 ثانية بدل 5 — معقول مع ngrok
      }
    );

    const data = response.data;

    return {
      message: data.message || "",
      suggestions: Array.isArray(data.suggestions) ? data.suggestions : [],
    };
  } catch (error) {
    if (import.meta.env.DEV) {
      console.warn("[API] start fallback used", error);
    }
    return {
      message: "",
      suggestions: [
        "كيف أحسب المعدل الفصلي؟",
        "ما هو نظام الساعات المعتمدة؟",
        "كيف أجمّد دراستي؟",
      ],
    };
  }
};