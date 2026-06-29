require('dotenv').config();
const express = require('express');
const https = require('https');
const path = require('path');

const app = express();
app.use(express.json({ limit: '50mb' }));
app.use(express.static(path.join(__dirname, 'public')));

app.post('/api/chat', (req, res) => {
  const { question, pdfBase64 } = req.body;

  if (!question || !pdfBase64) {
    return res.status(400).json({ error: 'Missing question or PDF data.' });
  }

  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    return res.status(500).json({ error: 'Server is missing ANTHROPIC_API_KEY. Add it to your .env file or environment variables.' });
  }

  const payload = JSON.stringify({
    model: 'claude-sonnet-4-6',
    max_tokens: 1024,
    system: `You are a helpful PDF assistant. Answer questions based strictly on the uploaded PDF document's content.

If the answer to the user's question is NOT found or cannot be reasonably inferred from the document, respond with this polite message:
"I'm sorry, but I couldn't find information about that in the provided PDF. The document may not cover this topic, or the relevant section might be outside what was provided. Feel free to ask about something else from the document!"

Keep answers clear, accurate, and well-structured. Use bullet points or short paragraphs when helpful.`,
    messages: [{
      role: 'user',
      content: [
        { type: 'document', source: { type: 'base64', media_type: 'application/pdf', data: pdfBase64 } },
        { type: 'text', text: question }
      ]
    }]
  });

  const options = {
    hostname: 'api.anthropic.com',
    path: '/v1/messages',
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Content-Length': Buffer.byteLength(payload),
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01'
    }
  };

  const apiReq = https.request(options, (apiRes) => {
    let data = '';
    apiRes.on('data', chunk => data += chunk);
    apiRes.on('end', () => {
      try {
        const parsed = JSON.parse(data);
        if (apiRes.statusCode !== 200) {
          console.error('Anthropic error:', apiRes.statusCode, data);
          return res.status(apiRes.statusCode).json({
            error: parsed.error?.message || `Anthropic API returned ${apiRes.statusCode}`
          });
        }
        const answer = (parsed.content || []).map(b => b.text || '').join('');
        return res.json({ answer });
      } catch (e) {
        console.error('Parse error:', e, data);
        return res.status(500).json({ error: 'Failed to parse Anthropic response.' });
      }
    });
  });

  apiReq.on('error', (e) => {
    console.error('HTTPS request error:', e);
    res.status(500).json({ error: 'Network error reaching Anthropic API: ' + e.message });
  });

  apiReq.write(payload);
  apiReq.end();
});

app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`PDF Mind running at http://localhost:${PORT}`);
  console.log(`API key loaded: ${process.env.ANTHROPIC_API_KEY ? 'YES ✓' : 'NO ✗ — set ANTHROPIC_API_KEY'}`);
});
