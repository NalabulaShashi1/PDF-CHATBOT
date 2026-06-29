# PDF Mind — Smart PDF Chatbot

## Quick Start (Local)

### 1. Install dependencies
```bash
npm install
```

### 2. Create a `.env` file in the project root
```
ANTHROPIC_API_KEY=sk-ant-api03-xxxxxxxxxxxxxxxx
```

### 3. Start the server
```bash
node server.js
```

Open **http://localhost:3000**

---

## Deploy on Render (free)

1. Push this folder to a GitHub repo  
2. Go to [render.com](https://render.com) → New → Web Service → connect repo  
3. **Build command:** `npm install`  
4. **Start command:** `node server.js`  
5. **Environment variable:** `ANTHROPIC_API_KEY` = your key  
6. Deploy ✅

## Deploy on Railway

1. Push to GitHub  
2. [railway.app](https://railway.app) → New Project → Deploy from GitHub  
3. Add env var `ANTHROPIC_API_KEY`  
4. Done ✅

## Deploy on Heroku

```bash
heroku create your-app-name
heroku config:set ANTHROPIC_API_KEY=sk-ant-...
git push heroku main
```

---

## Why a backend server?

Browsers block direct calls to the Anthropic API (CORS). This Express server proxies all requests — your API key stays secure on the server side.
