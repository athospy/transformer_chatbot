// Chat page logic. A question goes to /api/chat/ (retrieval: the closest written answer, instant).
// Under each answer, "What the fine-tuned model says" asks /api/generate/ (DialoGPT-small, ~10 s on
// CPU). Also handles waiting, errors, retry and "Clear chat". Plain JavaScript, no libraries.
(() => {
  'use strict';

  // Give up on the browser side after 90 s (gunicorn and nginx allow 120 s).
  const REQUEST_TIMEOUT_MS = 90000;
  // Only links to the university's own site become clickable (they come from the dataset).
  const TRUSTED_LINK_PREFIX = 'https://www.pittstate.edu/';

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

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;   // always text, never HTML
    return node;
  }

  // Add a message. Text always goes in with textContent, never as HTML, so neither
  // the user's message nor any answer can inject markup or scripts.
  function addMessage(role, text) {
    const item = el('div', `msg msg-${role}`);
    item.append(el('div', 'bubble', text));
    messages.append(item);
    scrollToEnd();
    return item;
  }

  // Waiting indicator: dots, a label screen readers announce once, and a seconds counter they
  // skip (aria-hidden), so they aren't told the time every second.
  function addWaiting(container, label) {
    const line = el('div', 'typing');
    const dots = el('span', 'dots');
    dots.setAttribute('aria-hidden', 'true');
    dots.append(el('span'), el('span'), el('span'));
    const text = el('span', null, label);
    const seconds = el('span', 'seconds');
    seconds.setAttribute('aria-hidden', 'true');
    text.append(seconds);
    line.append(dots, text);
    if (container === messages) {
      // A waiting bubble of its own in the conversation
      const item = el('div', 'msg msg-bot');
      line.className = 'bubble typing';
      item.append(line);
      messages.append(item);
      scrollToEnd();
      return { node: item, seconds };
    }
    container.append(line);
    scrollToEnd();
    return { node: line, seconds };
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

  // Stop tracking the request: timers off, waiting indicator gone, sending allowed again.
  function endRequest() {
    if (!pending) return;
    clearTimeout(pending.timeout);
    clearInterval(pending.ticker);
    pending.waiting.node.remove();
    pending = null;
    setBusy(false);
  }

  // POST a question and return the JSON. Errors come back with a message written for visitors;
  // error.cleared means "Clear chat" cancelled it and there's nothing to show.
  async function post(url, question, waiting) {
    const controller = new AbortController();
    const started = Date.now();
    pending = {
      controller,
      waiting,
      ticker: setInterval(() => {
        waiting.seconds.textContent = ` · ${Math.round((Date.now() - started) / 1000)} s`;
      }, 1000),
      timeout: setTimeout(() => controller.abort('timeout'), REQUEST_TIMEOUT_MS),
    };
    setBusy(true);
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: question }),
        signal: controller.signal,
      });
      // userFacing marks errors whose message is written for visitors
      if (!response.ok) throw Object.assign(new Error(errorText(response.status)), { userFacing: true });
      return await response.json();
    } catch (error) {
      const reason = controller.signal.reason;
      if (reason === 'cleared') throw Object.assign(new Error('cleared'), { cleared: true });
      if (reason === 'timeout') throw new Error('That took more than 90 s, so it was stopped. Try again.');
      if (error.userFacing) throw error;
      if (error instanceof TypeError) throw new Error("Couldn't reach the server. Check your connection and try again.");
      throw new Error('Something went wrong. Try again.');
    } finally {
      if (pending && pending.controller === controller) endRequest();
    }
  }

  function retryButton(onClick) {
    const button = el('button', 'retry-btn', 'Try again');
    button.type = 'button';
    button.addEventListener('click', () => { if (!pending) onClick(); });
    return button;
  }

  // ---------- Retrieval answer ----------

  // The answer text, with its official link (from the dataset) made clickable.
  function answerBubble(text, url) {
    const bubble = el('div', 'bubble');
    const at = url && url.startsWith(TRUSTED_LINK_PREFIX) ? text.indexOf(url) : -1;
    if (at === -1) {
      bubble.textContent = text;
      return bubble;
    }
    const link = el('a', null, url.replace('https://www.', '').replace(/\/index\.html$/, ''));
    link.href = url;
    link.target = '_blank';
    link.rel = 'noopener';
    bubble.append(document.createTextNode(text.slice(0, at)), link, document.createTextNode(text.slice(at + url.length)));
    return bubble;
  }

  // "Matched “Tell me about the Art program…” · similarity 0.82 · 12 ms"
  function matchNote(data) {
    const similarity = `similarity ${data.score.toFixed(2)}`;
    return data.found
      ? `Matched “${data.matched_question}” · ${similarity} · ${data.elapsed_ms} ms`
      : `Closest: “${data.matched_question}” · ${similarity}, below the ${data.threshold} cut-off`;
  }

  function addAnswer(data, question) {
    const item = el('div', 'msg msg-bot');
    item.append(answerBubble(data.response, data.source_url), el('p', 'meta', matchNote(data)));
    const button = el('button', 'model-btn', 'What the fine-tuned model says');
    button.type = 'button';
    button.addEventListener('click', () => {
      if (pending) return;
      button.remove();
      askModel(item, question);
    });
    item.append(button);
    messages.append(item);
    scrollToEnd();
  }

  async function ask(question, { retry = false } = {}) {
    const text = question.trim().slice(0, maxChars);
    if (!text || pending) return;
    if (!retry) addMessage('user', text);
    const waiting = addWaiting(messages, 'Searching the answers');
    try {
      addAnswer(await post('/api/chat/', text, waiting), text);
    } catch (error) {
      if (error.cleared) return;
      const item = addMessage('bot', error.message);
      item.classList.add('msg-error');
      item.append(retryButton(() => { item.remove(); ask(text, { retry: true }); }));
    }
  }

  // ---------- The fine-tuned model's answer (on demand) ----------

  async function askModel(item, question) {
    const block = el('div', 'model-block');
    block.append(el('p', 'model-label', 'Fine-tuned DialoGPT-small'));
    item.append(block);
    const waiting = addWaiting(block, 'Generating on CPU');
    try {
      const data = await post('/api/generate/', question, waiting);
      block.append(el('p', 'model-text', data.response));
      const seconds = (data.elapsed_ms / 1000).toFixed(1);
      block.append(el('p', 'meta', `${seconds} s · ${settings.num_beams} beams · up to ${settings.max_new_tokens} new tokens · `
        + 'generated text can be wrong, and its links may not exist'));
    } catch (error) {
      if (error.cleared) return;
      block.append(el('p', 'model-error', error.message),
        retryButton(() => { block.remove(); askModel(item, question); }));
    }
    scrollToEnd();
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
