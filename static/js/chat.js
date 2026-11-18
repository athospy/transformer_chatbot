// Chat page logic: send a question to /api/chat/, show the answer, and handle the
// waiting, error, retry and "clear chat" states. Plain JavaScript, no libraries.
(() => {
  'use strict';

  // Give up on the browser side after 90 s (gunicorn and nginx allow 120 s).
  const REQUEST_TIMEOUT_MS = 90000;

  const settings = JSON.parse(document.getElementById('generation-settings').textContent);
  const messages = document.getElementById('messages');
  const form = document.getElementById('composer');
  const input = document.getElementById('user-input');
  const sendBtn = document.getElementById('send-btn');
  const counter = document.getElementById('counter');
  const clearBtn = document.getElementById('clear-btn');
  const aboutDialog = document.getElementById('about-dialog');
  const maxChars = input.maxLength;

  // The request in flight, or null. Only one at a time, so answers never get mixed up.
  let pending = null;

  const scrollToEnd = () => { messages.scrollTop = messages.scrollHeight; };

  // Add a message. Text always goes in with textContent, never as HTML, so neither
  // the user's message nor the model's answer can inject markup or scripts.
  function addMessage(role, text) {
    const item = document.createElement('div');
    item.className = `msg msg-${role}`;
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = text;
    item.append(bubble);
    messages.append(item);
    scrollToEnd();
    return item;
  }

  function addNote(item, className, text) {
    const note = document.createElement('p');
    note.className = className;
    note.textContent = text;
    item.append(note);
  }

  // "9.8 s · 3 beams · up to 40 new tokens"
  function answerMeta(elapsedMs) {
    if (typeof elapsedMs !== 'number') return null;
    const seconds = (elapsedMs / 1000).toFixed(1);
    return `${seconds} s · ${settings.num_beams} beams · up to ${settings.max_new_tokens} new tokens`;
  }

  // Waiting bubble: dots, a label screen readers announce once, and a seconds counter
  // they skip (aria-hidden), so they aren't told the time every second.
  function addTyping() {
    const item = addMessage('bot', '');
    const bubble = item.querySelector('.bubble');
    bubble.classList.add('typing');
    bubble.innerHTML = '<span class="dots" aria-hidden="true"><span></span><span></span><span></span></span>'
      + '<span>Generating on CPU<span class="seconds" aria-hidden="true"></span></span>';
    return { item, seconds: bubble.querySelector('.seconds') };
  }

  function addError(text, question) {
    const item = addMessage('bot', text);
    item.classList.add('msg-error');
    const retry = document.createElement('button');
    retry.type = 'button';
    retry.className = 'retry-btn';
    retry.textContent = 'Try again';
    retry.addEventListener('click', () => {
      if (pending) return;
      item.remove();
      ask(question, { retry: true });
    });
    item.append(retry);
  }

  // Turn an HTTP error into words a visitor can act on.
  function errorText(status) {
    if (status === 502 || status === 503 || status === 504) {
      return 'The server is busy or restarting. Try again in a moment.';
    }
    if (status >= 500) return 'Something went wrong on the server. Try again.';
    return `The request failed (${status}). Try again.`;
  }

  function setBusy(busy) {
    sendBtn.setAttribute('aria-disabled', String(busy));
    messages.setAttribute('aria-busy', String(busy));
  }

  // Stop tracking the request: timers off, waiting bubble gone, sending allowed again.
  function endRequest() {
    if (!pending) return;
    clearTimeout(pending.timeout);
    clearInterval(pending.ticker);
    pending.typing.item.remove();
    pending = null;
    setBusy(false);
  }

  async function ask(question, { retry = false } = {}) {
    const text = question.trim().slice(0, maxChars);
    if (!text || pending) return;

    if (!retry) addMessage('user', text);

    const controller = new AbortController();
    const typing = addTyping();
    const started = Date.now();
    pending = {
      controller,
      typing,
      ticker: setInterval(() => {
        typing.seconds.textContent = ` · ${Math.round((Date.now() - started) / 1000)} s`;
      }, 1000),
      timeout: setTimeout(() => controller.abort('timeout'), REQUEST_TIMEOUT_MS),
    };
    setBusy(true);

    try {
      const response = await fetch('/api/chat/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text }),
        signal: controller.signal,
      });
      // userFacing marks errors whose message is written for visitors
      if (!response.ok) throw Object.assign(new Error(errorText(response.status)), { userFacing: true });
      const data = await response.json();

      endRequest();
      const item = addMessage('bot', data.response);
      const meta = answerMeta(data.elapsed_ms);
      if (meta) addNote(item, 'meta', meta);
    } catch (error) {
      // "Clear chat" already removed everything, so there's nothing to report.
      if (controller.signal.reason === 'cleared') return;
      endRequest();
      if (controller.signal.reason === 'timeout') {
        addError('That took more than 90 s, so it was stopped. Try again.', text);
      } else if (error instanceof TypeError) {
        addError("Couldn't reach the server. Check your connection and try again.", text);
      } else {
        addError(error.userFacing ? error.message : 'Something went wrong. Try again.', text);
      }
    }
  }

  // ---------- Composer ----------

  function resizeInput() {
    input.style.height = 'auto';
    input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
  }

  // Show the counter only near the limit, so it doesn't distract the rest of the time.
  function updateCounter() {
    const length = input.value.length;
    counter.textContent = length >= maxChars - 50 ? `${length}/${maxChars}` : '';
    counter.classList.toggle('at-limit', length >= maxChars);
  }

  form.addEventListener('submit', (event) => {
    event.preventDefault();
    if (pending || !input.value.trim()) return;
    const question = input.value;
    input.value = '';
    resizeInput();
    updateCounter();
    ask(question);
  });

  // Enter sends; Shift+Enter adds a new line. isComposing skips Enter while an input
  // method (e.g. Japanese or Chinese) is still composing a word.
  input.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      form.requestSubmit();
    }
  });

  input.addEventListener('input', () => {
    resizeInput();
    updateCounter();
  });

  // ---------- Suggestions, clear, "How it works" ----------

  messages.addEventListener('click', (event) => {
    const chip = event.target.closest('.chip');
    if (chip) ask(chip.dataset.question);
  });

  clearBtn.addEventListener('click', () => {
    if (pending) {
      pending.controller.abort('cleared');
      endRequest();
    }
    messages.querySelectorAll('.msg').forEach((item) => item.remove());
    messages.scrollTop = 0;
    input.focus();
  });

  document.querySelectorAll('#about-btn, [data-open-about]').forEach((button) => {
    button.addEventListener('click', () => aboutDialog.showModal());
  });

  // Clicking the dark area around the dialog closes it too.
  aboutDialog.addEventListener('click', (event) => {
    if (event.target === aboutDialog) aboutDialog.close();
  });
})();
