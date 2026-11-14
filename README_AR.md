# مِرصد — اتجاهات مناخ السعودية وLSTM

هذا هو الدليل العربي لمشروع Mirsad. النسخة الإنجليزية الرسمية موجودة في `README.md`.

مِرصد هو تطبيق Streamlit تفاعلي لتحليل اتجاهات نظام الأرض في المملكة العربية السعودية. يبدأ بمسح وطني كامل عبر خريطة حرارية لخلايا NASA POWER، ثم ينتقل إلى تفسير الإشارة باستخدام Seasonal Mann–Kendall وSen’s slope، ثم يعرض مؤشرات استباقية تقارن أحدث سنة مكتملة بالمناخ الشهري المعتاد.

## التشغيل

يحتاج المشروع Python 3.11 أو 3.12 واتصال إنترنت لتثبيت المكتبات وتحميل بلاطات الخرائط.

### Windows

```bash
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

### macOS أو Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

افتح `http://localhost:8501`. إذا بدت الخريطة فارغة، انتظر من 5 إلى 10 ثوانٍ لتحميل بلاطات OpenStreetMap أو Esri.

## الخصائص

يحتوي التطبيق على خريطة حرارية وطنية، طبقة Sen’s slope، خرائط الشوارع والأقمار الصناعية، مقارنة الرياض وجدة، اختبار Mann–Kendall الموسمي، p-value، Kendall’s tau، Sen’s slope، مؤشرات استباقية، ونموذج LSTM تجريبي اختياري. التطبيق يدعم العربية والإنجليزية.

## تنبيه علمي

النموذج تجريبي وتعليمي وليس توقعاً رسمياً من NASA. قيم NASA POWER تقديرات شبكية/إعادة تحليل وليست قياسات محطات محلية. الدلالة الإحصائية لا تثبت السببية.

## الملفات

راجع `README.md` للنسخة الإنجليزية الرسمية، و`SUBMISSION_HOW_TO_AR.md` لطريقة النشر والتقديم خطوة بخطوة.
