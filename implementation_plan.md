# 🚀 Implementation Plan: Advanced Diabetes Prediction System

এই প্ল্যানটিতে আপনার প্রোজেক্টকে একটি এন্টারপ্রাইজ-লেভেল, ফুল-স্ট্যাক অ্যাপ্লিকেশনে রূপান্তর করার ধাপগুলো তুলে ধরা হয়েছে।

## User Review Required

> [!IMPORTANT]  
> দয়া করে নিচের প্ল্যানটি পড়ুন এবং সব ঠিক থাকলে কনফার্ম করুন, তাহলে আমি কাজ শুরু করে দেব।

## Open Questions

> [!WARNING]  
> ১. **ফ্রন্টএন্ড ফ্রেমওয়ার্ক:** আমি কি React (Vite) ব্যবহার করব নাকি Next.js? (আমি React + Vite দিয়ে দ্রুত এবং সুন্দর Glassmorphism UI বানানোর প্রস্তাব দিচ্ছি)।
> ২. **TabNet মডেল:** TabNet মডেল ট্রেইন করার জন্য আমি একটি নতুন স্ক্রিপ্ট (`train_tabnet.py`) তৈরি করব। আপনি কি চান নতুন মডেলটি ডিফল্ট হিসেবে কাজ করুক, নাকি ইউজার UI থেকে মডেল (Ensemble/TabNet) বেছে নিতে পারবে?

## Proposed Changes

### Phase 1: Backend API (FastAPI) ⚡
Streamlit থেকে মূল লজিকটি আলাদা করে FastAPI দিয়ে একটি শক্তিশালী রেস্ট এপিআই (REST API) তৈরি করা হবে।
- **[NEW]** `backend/main.py`: FastAPI অ্যাপ্লিকেশন, রাউটিং এবং CORS কনফিগারেশন।
- **[NEW]** `backend/schemas.py`: Pydantic মডেল (রিকোয়েস্ট এবং রেসপন্স ভ্যালিডেশনের জন্য)।
- **[MODIFY]** `predict.py` (বা `backend/predict.py`): FastAPI এর সাথে কাজ করার জন্য রিফ্যাক্টর করা হবে।

### Phase 2: Frontend App (React/Vite) 🎨
প্রিমিয়াম Glassmorphism UI যুক্ত একটি সুন্দর, ফাস্ট এবং রেস্পন্সিভ ফ্রন্টএন্ড তৈরি করা হবে।
- **[NEW]** `frontend/`: React + Vite প্রজেক্ট সেটআপ।
- **[NEW]** `frontend/src/App.jsx`: মূল লেআউট এবং রাউটিং।
- **[NEW]** `frontend/src/components/Form.jsx`: পেশেন্ট ডেটা ইনপুট ফর্ম (Validation সহ)।
- **[NEW]** `frontend/src/components/Result.jsx`: প্রেডিকশন রেজাল্ট এবং SHAP গ্রাফ প্রদর্শনের পেজ।
- **[NEW]** `frontend/src/index.css`: Glassmorphism এবং মডার্ন অ্যানিমেশন (Tailwind/Vanilla CSS)।

### Phase 3: Deep Learning (TabNet) 🧠
ট্যাবুলার ডেটার জন্য গুগলের তৈরি TabNet মডেল ইন্টিগ্রেট করা হবে।
- **[NEW]** `train_tabnet.py`: `diabetes.csv` ব্যবহার করে TabNet মডেল ট্রেইন এবং সেভ করার স্ক্রিপ্ট।
- **[MODIFY]** `requirements.txt`: `pytorch-tabnet` এবং `torch` যুক্ত করা হবে।
- **[MODIFY]** `backend/predict.py`: TabNet মডেল লোড এবং প্রেডিকশনের জন্য লজিক আপডেট।

### Phase 4: MLOps (Data Drift Monitoring) ⚙️
Evidently AI ব্যবহার করে ডেটা ড্রিফট মনিটরিং যুক্ত করা হবে।
- **[NEW]** `backend/drift_monitor.py`: ইনকামিং ডেটা এবং মূল `diabetes.csv` এর মধ্যে তুলনা করে ডেটা ড্রিফট রিপোর্ট (HTML/JSON) জেনারেট করবে।
- **[NEW]** `backend/main.py`: ড্রিফট চেক করার জন্য একটি নতুন API এন্ডপয়েন্ট যোগ করা হবে।

### Phase 5: CI/CD & Cloud Deployment 🌐
গিটহাব অ্যাকশনস ব্যবহার করে অটোমেশন।
- **[NEW]** `.github/workflows/deploy.yml`: স্বয়ংক্রিয় টেস্টিং এবং ডকার বিল্ড করার জন্য CI/CD পাইপলাইন।
- **[MODIFY]** `docker-compose.yml`: ফ্রন্টএন্ড এবং ব্যাকএন্ড একসাথে রান করার জন্য।
- **[MODIFY]** `Dockerfile`: FastAPI এবং React রান করার জন্য আপডেট করা হবে।

### Phase 6: Documentation 📚
- **[MODIFY]** `README.md`: প্রফেশনাল, এন্টারপ্রাইজ লেভেলের গিটহাব রিডমি, যেখানে আর্কিটেকচার, API ডকুমেন্টেশন এবং রান করার নিয়ম লেখা থাকবে।

---

## Verification Plan

### Automated Tests
- ব্যাকএন্ড API এর জন্য Pytest ব্যবহার করে হেলথ চেক এবং প্রেডিকশন এন্ডপয়েন্ট টেস্ট করা হবে।
- Docker compose দিয়ে পুরো সিস্টেম লোকালি বিল্ড করে চেক করা হবে।

### Manual Verification
- আমি লোকালি React UI রান করে দেখব Glassmorphism ডিজাইন ঠিকমতো কাজ করছে কি না।
- API এর মাধ্যমে ডাটা পাঠিয়ে প্রেডিকশন এবং SHAP প্লট ঠিকমতো দেখাচ্ছে কি না তা নিশ্চিত করা হবে।
