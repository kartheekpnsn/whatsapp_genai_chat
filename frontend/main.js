const API = 'http://localhost:8003';

const chatArea = document.getElementById('chat-area');
const msgInput = document.getElementById('msg-input');
const sendBtn = document.getElementById('send-btn');
const headerName = document.getElementById('header-name');

let sending = false;

async function init() {
  try {
    const res = await fetch(`${API}/health`);
    const data = await res.json();
    headerName.textContent = data.user1;
  } catch {
    headerName.textContent = 'Chat (backend offline)';
  }
}

function addBubble(text, type) {
  const div = document.createElement('div');
  div.className = `bubble ${type}`;
  div.textContent = text;
  chatArea.appendChild(div);
  chatArea.scrollTop = chatArea.scrollHeight;
  return div;
}

async function sendMessage() {
  if (sending) return;
  const text = msgInput.value.trim();
  if (!text) return;

  sending = true;
  msgInput.value = '';
  msgInput.disabled = true;
  sendBtn.disabled = true;

  addBubble(text, 'user');
  const typing = addBubble('...', 'bot typing');

  try {
    const res = await fetch(`${API}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text }),
    });
    const data = await res.json();
    typing.remove();
    addBubble(data.reply, 'bot');
  } catch {
    typing.remove();
    addBubble('Error: could not reach backend.', 'bot');
  } finally {
    sending = false;
    msgInput.disabled = false;
    sendBtn.disabled = false;
    msgInput.focus();
  }
}

sendBtn.addEventListener('click', sendMessage);
msgInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) sendMessage();
});

init();
