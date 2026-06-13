# 🛡️ Qdrant Backup & Restore Guide (Har 3 Weeks ke Liye)

Aapko **kabhi bhi embeddings re-generate karne ki zaroorat nahi** — bas yeh 6 steps follow karein:

---

## ✅ Step 1: Backup (Har 3 Weeks Ek Baar)
```bash
python qdrant_backup.py
```
→ `docusaurus_book_backup.json` file create ho jaye gi  
→ **13 points** (aapke documents) save ho jayenge

---

## ✅ Step 2: Naya Qdrant Cluster Banayein (Jab Purana Expire Ho Jaye)
- Jaayein: [https://cloud.qdrant.io](https://cloud.qdrant.io)
- "Create Cluster" → Free tier choose karein
- New `QDRANT_URL` aur `QDRANT_API_KEY` copy karein

---

## ✅ Step 3: `.env` Update Karein
Open:  
`D:\Code\Big Hackathons With AI\Hackathon 1 Phase 2\hackathon-Giaic\.env`

Replace:
```env
QDRANT_URL="https://purana-url..."
QDRANT_API_KEY="purana-key"
```
With new ones:
```env
QDRANT_URL="https://naya-cluster-id.region-0.gcp.cloud.qdrant.io"
QDRANT_API_KEY="naya-api-key"
```

---

## ✅ Step 4: Restore Karo
```bash
python qdrant_backup.py restore
```
→ Purani collection delete ho jaye gi  
→ Naye cluster mein 13 points restore ho jayengi  
→ **Embeddings generate karne ki zaroorat nahi**

---

## ✅ Step 5: Backend Restart Karein
```bash
uvicorn api:app --reload --port 8000
```

---

## ✅ Step 6: Frontend Restart Karein
```bash
cd My-Book
npm start
```

---

## 🎯 Result:
- Chatbot wapis chal gaya  
- Aapka data safe hai  
- Har 3 weeks sirf **5 minute** lagte hain  

> 💡 Tip: Is guide ko bookmark karein — ya print kar ke desk pe rakh lein.

---  
*Prepared for: Muhammad Ayaan | Hackathon GIAIC*  
*Last updated: 17 Feb 2026*