/* ============================================================
   Higgs Audio v3 TTS — Portable by Neurogen
   app.js — vanilla ES module, no dependencies.
   Talks to the local HTTP API (same origin). Works in pywebview
   and any modern browser. Degrades gracefully when the API is
   unreachable (offline preview mode).
   ============================================================ */

'use strict';

/* ------------------------------------------------------------
   0. Small DOM + util helpers
   ------------------------------------------------------------ */
const $  = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
const el = (tag, cls, html) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html != null) n.innerHTML = html;
  return n;
};
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const fmtTime = (s) => {
  if (!isFinite(s) || s < 0) s = 0;
  const m = Math.floor(s / 60);
  const r = Math.floor(s % 60);
  return `${m}:${String(r).padStart(2, '0')}`;
};
const escapeHtml = (s) => String(s).replace(/[&<>"']/g, c =>
  ({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' }[c]));

const LS = {
  get(k, def) { try { const v = localStorage.getItem(k); return v == null ? def : JSON.parse(v); } catch { return def; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} },
};

/* ------------------------------------------------------------
   1. Fallback control-token catalog (used until /api/info loads)
   ------------------------------------------------------------ */
const EMOJI = {
  elation:'🤩', amusement:'😄', enthusiasm:'🔥', determination:'💪', pride:'😌',
  contentment:'🙂', affection:'🥰', relief:'😮‍💨', contemplation:'🤔', confusion:'😕',
  surprise:'😲', awe:'😯', longing:'🥺', arousal:'😳', anger:'😠', fear:'😨',
  disgust:'🤢', bitterness:'😒', sadness:'😢', shame:'😳', helplessness:'😖',
  singing:'🎵', shouting:'📢', whispering:'🤫',
  cough:'😷', laughter:'😂', crying:'😭', screaming:'😱', burping:'🫧',
  humming:'🎶', sigh:'😮‍💨', sniff:'👃', sneeze:'🤧',
};
const RU = {
  elation:'Ликование', amusement:'Веселье', enthusiasm:'Энтузиазм', determination:'Решимость',
  pride:'Гордость', contentment:'Довольство', affection:'Нежность', relief:'Облегчение',
  contemplation:'Размышление', confusion:'Замешательство', surprise:'Удивление', awe:'Восхищение',
  longing:'Тоска', arousal:'Возбуждение', anger:'Гнев', fear:'Страх', disgust:'Отвращение',
  bitterness:'Горечь', sadness:'Грусть', shame:'Стыд', helplessness:'Беспомощность',
  singing:'Пение', shouting:'Крик', whispering:'Шёпот',
  cough:'Кашель', laughter:'Смех', crying:'Плач', screaming:'Вопль', burping:'Отрыжка',
  humming:'Мычание', sigh:'Вздох', sniff:'Шмыг', sneeze:'Чих',
  speed_very_slow:'Очень медленно', speed_slow:'Медленно', speed_fast:'Быстро', speed_very_fast:'Очень быстро',
  pause:'Пауза', long_pause:'Длинная пауза', pitch_low:'Низкий тон', pitch_high:'Высокий тон',
  expressive_high:'Экспрессивно', expressive_low:'Сдержанно',
};
const ONO = { cough:'Ahem', laughter:'Haha', crying:'Sob', screaming:'Ahh', burping:'Burp', humming:'Hmm', sigh:'Uh', sniff:'Sff', sneeze:'Achoo' };

const FALLBACK = {
  emotions: ['elation','amusement','enthusiasm','determination','pride','contentment','affection','relief','contemplation','confusion','surprise','awe','longing','arousal','anger','fear','disgust','bitterness','sadness','shame','helplessness']
    .map(n => ({ token:`<|emotion:${n}|>`, label_ru:RU[n], label_en:n, name:n })),
  styles: ['singing','shouting','whispering']
    .map(n => ({ token:`<|style:${n}|>`, label_ru:RU[n], label_en:n, name:n })),
  sfx: ['cough','laughter','crying','screaming','burping','humming','sigh','sniff','sneeze']
    .map(n => ({ token:`<|sfx:${n}|>`, label_ru:RU[n], label_en:n, name:n, onomatopoeia:ONO[n] })),
  prosody_speed:   ['speed_very_slow','speed_slow','speed_fast','speed_very_fast'].map(n => ({ token:`<|prosody:${n}|>`, label_ru:RU[n], label_en:n, name:n })),
  prosody_pause:   ['pause','long_pause'].map(n => ({ token:`<|prosody:${n}|>`, label_ru:RU[n], label_en:n, name:n })),
  prosody_pitch:   ['pitch_low','pitch_high'].map(n => ({ token:`<|prosody:${n}|>`, label_ru:RU[n], label_en:n, name:n })),
  prosody_delivery:['expressive_high','expressive_low'].map(n => ({ token:`<|prosody:${n}|>`, label_ru:RU[n], label_en:n, name:n })),
};

/* Token classification: which insert at START (delivery) vs at CURSOR (positional). */
const POSITIONAL = new Set(['sfx','prosody_pause']); // pause/long_pause + sfx insert at cursor
const isDeliveryGroup = (group) => !POSITIONAL.has(group);

/* ------------------------------------------------------------
   2. Global app state
   ------------------------------------------------------------ */
const App = {
  info: null,            // /api/info response
  state: null,           // last /api/state
  defaults: { temperature: 0.8, top_k: 50, top_p: 1.0, max_new_tokens: 2048 },
  userDefaults: LS.get('hg_defaults', null),
  reference: { ref_id: null, name: null, duration_s: null }, // uploaded clone reference
  es: null,              // EventSource
  pollTimer: null,
  ready: false,
};

/* Sampling param schema reused across Generate / Clone / Settings */
const PARAM_SCHEMA = [
  { key:'temperature',    label:'Temperature', min:0,   max:1.5,  step:0.01, tip:'Креативность / разнообразие. Выше — живее, но менее стабильно.' },
  { key:'top_k',          label:'Top-K',       min:0,   max:100,  step:1,    tip:'Сэмплирование из K самых вероятных токенов. 0 — выключено.' },
  { key:'top_p',          label:'Top-P',       min:0,   max:1,    step:0.01, tip:'Nucleus-сэмплирование. 1.0 — выключено.', offAt:1 },
  { key:'max_new_tokens', label:'Max tokens',  min:256, max:4096, step:64,   tip:'Максимальная длина аудио-генерации.' },
];

/* ------------------------------------------------------------
   3. API client
   ------------------------------------------------------------ */
const api = {
  async get(path) {
    const r = await fetch(path, { headers: { 'Accept':'application/json' } });
    if (!r.ok) throw new Error(`${path} → HTTP ${r.status}`);
    return r.json();
  },
  async post(path, body, isForm = false) {
    const opts = { method:'POST' };
    if (isForm) { opts.body = body; }
    else { opts.headers = { 'Content-Type':'application/json' }; opts.body = JSON.stringify(body || {}); }
    const r = await fetch(path, opts);
    let data = null;
    try { data = await r.json(); } catch { /* ignore */ }
    if (!r.ok) throw new Error((data && data.error) || `${path} → HTTP ${r.status}`);
    return data;
  },
  async delete(path) {
    const r = await fetch(path, { method:'DELETE' });
    let data = null;
    try { data = await r.json(); } catch { /* ignore */ }
    if (!r.ok) throw new Error((data && data.error) || `${path} → HTTP ${r.status}`);
    return data;
  },
};

/* ------------------------------------------------------------
   4. Toast notifications
   ------------------------------------------------------------ */
const ICONS = {
  ok:  '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="m5 13 4 4L19 7"/></svg>',
  err: '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6 6 18M6 6l12 12"/></svg>',
  info:'<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 16v-5M12 8h.01"/></svg>',
};
function toast(message, kind = 'info', ttl = 3600) {
  const host = $('#toasts');
  const t = el('div', `toast toast--${kind}`);
  t.innerHTML = `<span class="toast__ic">${ICONS[kind] || ICONS.info}</span><span class="toast__body">${escapeHtml(message)}</span>`;
  host.appendChild(t);
  const close = () => { t.classList.add('is-out'); setTimeout(() => t.remove(), 320); };
  setTimeout(close, ttl);
  t.addEventListener('click', close);
}

/* ------------------------------------------------------------
   5. Tabs
   ------------------------------------------------------------ */
function setupTabs() {
  $$('.tab').forEach(tab => {
    tab.addEventListener('click', () => switchTab(tab.dataset.tab));
  });
}
function switchTab(name) {
  $$('.tab').forEach(t => {
    const on = t.dataset.tab === name;
    t.classList.toggle('is-active', on);
    t.setAttribute('aria-selected', on ? 'true' : 'false');
  });
  $$('.panel').forEach(p => {
    const on = p.id === `panel-${name}`;
    p.hidden = !on;
    p.classList.toggle('is-active', on);
  });
  if (name === 'history') renderHistory();
  $('#main').scrollTo({ top: 0, behavior: 'smooth' });
}

/* ------------------------------------------------------------
   6. Control-token pills
   ------------------------------------------------------------ */
function buildPills() {
  const cat = (App.info && App.info.control_tokens) || FALLBACK;
  // Emotions
  fillPills('#pills-emotions', cat.emotions || FALLBACK.emotions, 'emotions');
  // Styles
  fillPills('#pills-styles', cat.styles || FALLBACK.styles, 'styles');
  // Prosody — merge speed + pause + pitch + delivery into one bar
  const prosody = [
    ...(cat.prosody_speed   || FALLBACK.prosody_speed),
    ...(cat.prosody_pause   || FALLBACK.prosody_pause),
    ...(cat.prosody_pitch   || FALLBACK.prosody_pitch),
    ...(cat.prosody_delivery|| FALLBACK.prosody_delivery),
  ];
  fillPills('#pills-prosody', prosody, 'prosody');
  // SFX
  fillPills('#pills-sfx', cat.sfx || FALLBACK.sfx, 'sfx', true);
}

function nameFromToken(tok) {
  const m = /<\|[a-z]+:([a-z_]+)\|>/i.exec(tok || '');
  return m ? m[1] : '';
}

function fillPills(sel, items, group, isSfx = false) {
  const row = $(sel);
  if (!row) return;
  row.innerHTML = '';
  items.forEach(it => {
    const name = it.name || nameFromToken(it.token);
    const ru = it.label_ru || RU[name] || name;
    const en = it.label_en || name;
    const emoji = EMOJI[name] || '◆';
    const pill = el('button', `pill${isSfx ? ' pill--sfx' : ''}`);
    pill.type = 'button';
    pill.dataset.tip = `${ru} · ${en}`;
    // group meta for insertion logic
    const realGroup = group === 'prosody'
      ? (/^pause|long_pause$/.test(name) ? 'prosody_pause' : 'prosody_other')
      : group;
    pill.dataset.group = realGroup;
    pill.dataset.token = it.token;
    if (it.onomatopoeia || ONO[name]) pill.dataset.ono = it.onomatopoeia || ONO[name];
    pill.innerHTML = `<span class="pill__emoji">${emoji}</span><span>${escapeHtml(ru)}</span>`;
    pill.addEventListener('click', () => insertToken(getActiveTextarea(), pill.dataset));
    row.appendChild(pill);
  });
}

/* Which textarea is the "current" one (Generate vs Clone tab) */
function getActiveTextarea() {
  return $('#cloneText').closest('.panel').classList.contains('is-active')
    ? $('#cloneText') : $('#genText');
}

/* Insert a control token: delivery → start; positional → at cursor. */
function insertToken(ta, data) {
  const token = data.token;
  const group = data.group;
  const ono = data.ono;
  const positional = POSITIONAL.has(group);
  const payload = ono ? `${token} ${ono} ` : `${token} `;

  if (positional) {
    const start = ta.selectionStart ?? ta.value.length;
    const end = ta.selectionEnd ?? ta.value.length;
    ta.value = ta.value.slice(0, start) + payload + ta.value.slice(end);
    const pos = start + payload.length;
    ta.setSelectionRange(pos, pos);
  } else {
    // delivery token → prepend at the very start (avoid duplicate same token)
    if (ta.value.startsWith(token)) { ta.focus(); return; }
    ta.value = payload + ta.value.replace(/^\s+/, '');
    ta.setSelectionRange(ta.value.length, ta.value.length);
  }
  ta.focus();
  ta.dispatchEvent(new Event('input'));
}

/* Insert a Russian stress mark. mark='́' (acute, AFTER vowel) or '+' (BEFORE vowel).
   With a selected vowel it wraps the selection on the correct side; with a bare
   cursor it inserts the mark at the caret. */
function insertStress(mark, before) {
  const ta = getActiveTextarea();
  const s = ta.selectionStart ?? ta.value.length;
  const e = ta.selectionEnd ?? ta.value.length;
  const sel = ta.value.slice(s, e);
  const ins = sel ? (before ? mark + sel : sel + mark) : mark;
  ta.value = ta.value.slice(0, s) + ins + ta.value.slice(e);
  const pos = s + ins.length;
  ta.setSelectionRange(pos, pos);
  ta.focus();
  ta.dispatchEvent(new Event('input'));
}

/* ------------------------------------------------------------
   7. Sliders (reusable factory)
   ------------------------------------------------------------ */
function buildSliders(containerSel, values, onChange) {
  const c = $(containerSel);
  if (!c) return {};
  c.innerHTML = '';
  const refs = {};
  PARAM_SCHEMA.forEach(p => {
    const wrap = el('div', 'slider');
    const off = p.offAt != null && Number(values[p.key]) === p.offAt;
    wrap.innerHTML = `
      <div class="slider__top">
        <span class="slider__label">${p.label}
          <span class="info" data-tip="${escapeHtml(p.tip)}">i</span>
        </span>
        <span class="slider__val${off ? ' is-off' : ''}"></span>
      </div>
      <input type="range" min="${p.min}" max="${p.max}" step="${p.step}" value="${values[p.key]}" aria-label="${p.label}">`;
    const input = $('input', wrap);
    const val = $('.slider__val', wrap);
    const render = () => {
      const v = Number(input.value);
      const isOff = p.offAt != null && v === p.offAt;
      val.textContent = isOff ? 'выкл' : (p.step < 1 ? v.toFixed(2) : v);
      val.classList.toggle('is-off', isOff);
      const pct = ((v - p.min) / (p.max - p.min)) * 100;
      input.style.setProperty('--fill', pct + '%');
    };
    input.addEventListener('input', () => { render(); onChange && onChange(p.key, Number(input.value)); });
    render();
    c.appendChild(wrap);
    refs[p.key] = input;
  });
  return refs;
}
function readSliders(refs) {
  const out = {};
  PARAM_SCHEMA.forEach(p => { if (refs[p.key]) out[p.key] = Number(refs[p.key].value); });
  return out;
}
function setSliders(refs, values) {
  PARAM_SCHEMA.forEach(p => {
    if (refs[p.key] && values[p.key] != null) {
      refs[p.key].value = values[p.key];
      refs[p.key].dispatchEvent(new Event('input'));
    }
  });
}

/* slider ref holders */
let genSliders = {}, cloneSliders = {}, defaultSliders = {};

/* ------------------------------------------------------------
   8. Voices dropdown
   ------------------------------------------------------------ */
function buildVoices() {
  const sel = $('#voiceSelect');
  sel.innerHTML = '';
  const zero = el('option');
  zero.value = '__zero__';
  zero.textContent = 'Без референса (zero-shot)';
  sel.appendChild(zero);
  const voices = (App.info && App.info.voices) || [];
  voices.forEach(v => {
    const o = el('option');
    o.value = v.id;
    o.textContent = v.name + (v.description ? '' : '');
    o.dataset.desc = v.description || '';
    sel.appendChild(o);
  });
  sel.addEventListener('change', updateVoiceDesc);
  updateVoiceDesc();
}
function updateVoiceDesc() {
  const sel = $('#voiceSelect');
  const opt = sel.options[sel.selectedIndex];
  $('#voiceDesc').textContent = sel.value === '__zero__'
    ? 'Без референса — модель сама выбирает тембр голоса.'
    : (opt && opt.dataset.desc) || '';
}

/* ------------------------------------------------------------
   9. Audio player + WebAudio waveform visualizer
   ------------------------------------------------------------ */
let audioCtx = null;
function getAudioCtx() {
  if (!audioCtx) {
    const AC = window.AudioContext || window.webkitAudioContext;
    if (AC) audioCtx = new AC();
  }
  return audioCtx;
}

/* A Player binds an <audio>, a play button, a canvas, and time labels. */
class Player {
  constructor({ audio, playBtn, canvas, curTime, totTime }) {
    this.audio = $(audio);
    this.btn = $(playBtn);
    this.canvas = $(canvas);
    this.cur = $(curTime);
    this.tot = $(totTime);
    this.peaks = null;       // Float32 normalized peaks
    this.raf = null;
    this.url = null;

    this.btn.addEventListener('click', () => this.toggle());
    this.audio.addEventListener('play',  () => { this.icon(true);  this.loop(); });
    this.audio.addEventListener('pause', () => { this.icon(false); cancelAnimationFrame(this.raf); this.draw(); });
    this.audio.addEventListener('ended', () => { this.icon(false); this.draw(); });
    this.audio.addEventListener('loadedmetadata', () => { this.tot.textContent = fmtTime(this.audio.duration); });
    this.audio.addEventListener('timeupdate', () => { this.cur.textContent = fmtTime(this.audio.currentTime); });
    // seek by clicking the waveform
    this.canvas.addEventListener('click', (e) => {
      const r = this.canvas.getBoundingClientRect();
      const ratio = clamp((e.clientX - r.left) / r.width, 0, 1);
      if (isFinite(this.audio.duration)) this.audio.currentTime = ratio * this.audio.duration;
      this.draw();
    });
    window.addEventListener('resize', () => this.draw());
  }
  icon(playing) {
    $('.ic-play', this.btn).hidden = playing;
    $('.ic-pause', this.btn).hidden = !playing;
    this.btn.setAttribute('aria-label', playing ? 'Пауза' : 'Воспроизвести');
  }
  toggle() {
    const ctx = getAudioCtx();
    if (ctx && ctx.state === 'suspended') ctx.resume();
    if (this.audio.paused) this.audio.play().catch(() => {}); else this.audio.pause();
  }
  /* Load from a Blob (or URL). Decodes peaks for the waveform. */
  async load(blob) {
    if (this.url) URL.revokeObjectURL(this.url);
    this.url = URL.createObjectURL(blob);
    this.audio.src = this.url;
    this.audio.load();
    this.peaks = null;
    this.draw(); // placeholder bars immediately
    try {
      const ctx = getAudioCtx();
      if (ctx) {
        const buf = await ctx.decodeAudioData(await blob.arrayBuffer());
        this.peaks = computePeaks(buf, 160);
      }
    } catch { /* fall back to flat bars */ }
    this.draw();
  }
  loop() {
    cancelAnimationFrame(this.raf);
    const step = () => { this.draw(); this.raf = requestAnimationFrame(step); };
    step();
  }
  draw() {
    const cv = this.canvas;
    const dpr = window.devicePixelRatio || 1;
    const w = cv.clientWidth || 600, h = cv.clientHeight || 64;
    if (cv.width !== w * dpr) { cv.width = w * dpr; cv.height = h * dpr; }
    const ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const N = 64;
    const peaks = this.peaks || flatPeaks(N);
    const bars = peaks.length;
    const gap = 2;
    const bw = (w - (bars - 1) * gap) / bars;
    const mid = h / 2;
    const progress = (isFinite(this.audio.duration) && this.audio.duration > 0)
      ? this.audio.currentTime / this.audio.duration : 0;

    for (let i = 0; i < bars; i++) {
      const p = peaks[i];
      const x = i * (bw + gap);
      const played = (i / bars) <= progress;
      // animated subtle bounce when playing
      let amp = p;
      if (!this.audio.paused && played) amp = p * (0.85 + 0.15 * Math.sin(Date.now() / 110 + i));
      const bh = Math.max(2, amp * (h - 8));
      const r = Math.min(bw / 2, 3);
      ctx.beginPath();
      roundRect(ctx, x, mid - bh / 2, bw, bh, r);
      if (played) {
        const g = ctx.createLinearGradient(0, mid - bh/2, 0, mid + bh/2);
        g.addColorStop(0, '#7BCFA3'); g.addColorStop(1, '#827DBD');
        ctx.fillStyle = g;
      } else {
        ctx.fillStyle = 'rgba(255,255,255,0.14)';
      }
      ctx.fill();
    }
    // playhead
    if (progress > 0 && progress < 1) {
      const px = progress * w;
      ctx.fillStyle = 'rgba(255,255,255,0.85)';
      ctx.fillRect(px - 0.5, 4, 1.5, h - 8);
    }
  }
}
function roundRect(ctx, x, y, w, h, r) {
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
}
function computePeaks(audioBuffer, count) {
  const ch = audioBuffer.getChannelData(0);
  const block = Math.floor(ch.length / count) || 1;
  const peaks = new Float32Array(count);
  let max = 0;
  for (let i = 0; i < count; i++) {
    let m = 0;
    const start = i * block;
    for (let j = 0; j < block; j++) { const v = Math.abs(ch[start + j] || 0); if (v > m) m = v; }
    peaks[i] = m; if (m > max) max = m;
  }
  if (max > 0) for (let i = 0; i < count; i++) peaks[i] = clamp(peaks[i] / max, 0.04, 1);
  return peaks;
}
function flatPeaks(n) {
  // pleasant static placeholder shape
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) a[i] = 0.18 + 0.12 * Math.abs(Math.sin(i * 0.5)) + 0.06 * Math.abs(Math.sin(i * 0.17));
  return a;
}

let genPlayer = null, clonePlayer = null;

/* base64 (no prefix) → Blob(audio/wav) */
function b64ToBlob(b64, mime = 'audio/wav') {
  const bin = atob(b64);
  const len = bin.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) bytes[i] = bin.charCodeAt(i);
  return new Blob([bytes], { type: mime });
}

/* ------------------------------------------------------------
   10. Generation flow
   ------------------------------------------------------------ */
function fmtDur(sec) {
  sec = Math.round(sec);
  if (sec < 60) return `${sec} с`;
  const m = Math.floor(sec / 60), s = sec % 60;
  return s ? `${m} мин ${s} с` : `${m} мин`;
}

async function runGeneration({ text, reference, referenceText, sliders, btn, progressId, resultCard, player, metaId, voiceLabel, statusId, statusTxtId, stopId }) {
  if (!text.trim()) { toast('Введите текст для озвучки', 'err'); return; }
  if (!App.ready) { toast('Модель ещё не готова', 'err'); openLoader(); return; }

  btn.classList.add('is-loading'); btn.disabled = true;
  setFooter('генерация…');

  // Live progress poller + Stop button (generation runs server-side).
  const statusEl = statusId ? $(statusId) : null;
  const statusTxt = statusTxtId ? $(statusTxtId) : null;
  const stopBtn = stopId ? $(stopId) : null;
  let pollTimer = null;
  const onStop = async () => {
    if (stopBtn) { stopBtn.disabled = true; stopBtn.textContent = '⏹ Останавливаю…'; }
    try { await api.post('/api/stop', {}); } catch {}
  };
  if (statusEl) statusEl.hidden = false;
  if (statusTxt) statusTxt.textContent = 'Генерация…';
  if (stopBtn) { stopBtn.disabled = false; stopBtn.textContent = '⏹ Стоп'; stopBtn.addEventListener('click', onStop); }
  const poll = async () => {
    try {
      const st = await api.get('/api/state');
      const g = st && st.generation;
      if (g && g.active && statusTxt) {
        const stage = g.stage || 'decode';
        const step = g.step || 0, total = g.total || 0, el = g.elapsed_s || 0;
        const chunk = g.chunk || 0, chunks = g.chunks || 0;
        const part = chunks > 1 ? `Часть ${chunk}/${chunks} · ` : '';
        let txt;
        if (stage === 'prefill') {
          txt = `${part}Подготовка голоса… ${step}/${total}`;
        } else {
          txt = `${part}Генерация… ${step}/${total} токенов · ${fmtDur(el)}`;
          if (step > 4 && el > 1) txt += ` · ≤ ${fmtDur((total - step) * (el / step))}`;
        }
        statusTxt.textContent = txt;
      }
    } catch {}
  };
  pollTimer = setInterval(poll, 800); poll();

  const seedRaw = $('#seedInput').value.trim();
  const params = readSliders(sliders);
  const payload = {
    text,
    reference,
    reference_text: referenceText || null,
    temperature: params.temperature,
    top_k: params.top_k,
    top_p: params.top_p,
    max_new_tokens: params.max_new_tokens,
    seed: seedRaw === '' ? null : parseInt(seedRaw, 10),
  };

  try {
    const res = await api.post('/api/generate', payload);
    if (!res || res.ok === false) throw new Error((res && res.error) || 'Ошибка генерации');

    const blob = b64ToBlob(res.audio_b64);
    await player.load(blob);

    // meta line
    const dur = res.duration_s != null ? `${res.duration_s.toFixed(2)} с` : '—';
    const eng = res.engine_s   != null ? `${res.engine_s.toFixed(2)} с`  : '—';
    const rtf = res.rtf        != null ? res.rtf.toFixed(2)              : '—';
    $(metaId).innerHTML =
      `<span>⏱ <b>${dur}</b></span><span>⚙ движок ${eng}</span><span>RTF <b>${rtf}</b></span>`;

    resultCard.hidden = false;
    resultCard.scrollIntoView({ behavior:'smooth', block:'nearest' });

    // history
    addHistory({
      id: Date.now(),
      text,
      voice: voiceLabel,
      duration_s: res.duration_s,
      output_file: res.output_file || null,
      audio_b64: res.audio_b64,           // kept for offline replay
      sample_rate: res.sample_rate || null,
      ts: Date.now(),
    });

    toast('Готово! Аудио сгенерировано', 'ok');
    setFooter('готово');
  } catch (e) {
    console.warn('generate failed:', e);
    const msg = String(e.message || e);
    if (/останов/i.test(msg)) { toast('Генерация остановлена', 'info'); setFooter('остановлено'); }
    else { toast(msg, 'err', 5000); setFooter('ошибка генерации'); }
  } finally {
    btn.classList.remove('is-loading'); btn.disabled = false;
    if (pollTimer) clearInterval(pollTimer);
    if (statusEl) statusEl.hidden = true;
    if (stopBtn) stopBtn.removeEventListener('click', onStop);
  }
}

function currentReferenceForGenerate() {
  const v = $('#voiceSelect').value;
  if (v === '__zero__') return null;
  return { preset: v };
}

function setupGenerate() {
  const ta = $('#genText');
  const updateCount = () => { $('#charCount').textContent = `${ta.value.length} символов`; };
  ta.addEventListener('input', updateCount); updateCount();
  $('#btnClearText').addEventListener('click', () => { ta.value = ''; ta.dispatchEvent(new Event('input')); ta.focus(); });

  // Stress marks: ́ (U+0301) after the vowel, or + before the vowel.
  $('#genStressAcute').addEventListener('click', () => insertStress('́', false));
  $('#genStressPlus').addEventListener('click', () => insertStress('+', true));

  $('#btnRandSeed').addEventListener('click', () => { $('#seedInput').value = Math.floor(Math.random() * 1e9); });
  $('#btnResetSampling').addEventListener('click', () => { setSliders(genSliders, effectiveDefaults()); toast('Параметры сброшены', 'info'); });

  // pointer glow on big button
  const gbtn = $('#generateBtn');
  gbtn.addEventListener('pointermove', (e) => {
    const r = gbtn.getBoundingClientRect();
    gbtn.style.setProperty('--mx', `${e.clientX - r.left}px`);
    gbtn.style.setProperty('--my', `${e.clientY - r.top}px`);
  });

  gbtn.addEventListener('click', () => {
    const sel = $('#voiceSelect');
    const voiceLabel = sel.value === '__zero__' ? 'zero-shot' : (sel.options[sel.selectedIndex]?.textContent || sel.value);
    runGeneration({
      text: ta.value,
      reference: currentReferenceForGenerate(),
      referenceText: null,
      sliders: genSliders,
      btn: gbtn,
      progressId: '#genProgress',
      statusId: '#genStatus',
      statusTxtId: '#genStatusTxt',
      stopId: '#genStopBtn',
      resultCard: $('#resultCard'),
      player: genPlayer,
      metaId: '#resultMeta',
      voiceLabel,
    });
  });

  $('#downloadBtn').addEventListener('click', () => downloadCurrent(genPlayer, 'higgs_audio'));
}

function setupClone() {
  const dz = $('#dropzone');
  const input = $('#refFile');
  const ta = $('#cloneText');
  const updateCount = () => { $('#cloneCharCount').textContent = `${ta.value.length} символов`; };
  ta.addEventListener('input', updateCount); updateCount();

  dz.addEventListener('click', () => input.click());
  dz.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); } });
  input.addEventListener('change', () => { if (input.files[0]) uploadReference(input.files[0]); });

  ['dragenter','dragover'].forEach(ev => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add('is-drag'); }));
  ['dragleave','drop'].forEach(ev => dz.addEventListener(ev, (e) => { e.preventDefault(); if (ev !== 'dragleave' || !dz.contains(e.relatedTarget)) dz.classList.remove('is-drag'); }));
  dz.addEventListener('drop', (e) => { const f = e.dataTransfer.files[0]; if (f) uploadReference(f); });

  $('#refFileRemove').addEventListener('click', clearReference);

  const cbtn = $('#cloneGenerateBtn');
  cbtn.addEventListener('pointermove', (e) => {
    const r = cbtn.getBoundingClientRect();
    cbtn.style.setProperty('--mx', `${e.clientX - r.left}px`);
    cbtn.style.setProperty('--my', `${e.clientY - r.top}px`);
  });
  cbtn.addEventListener('click', () => {
    if (!App.reference.ref_id) { toast('Сначала загрузите референс голоса', 'err'); return; }
    runGeneration({
      text: ta.value,
      reference: { ref_id: App.reference.ref_id },
      referenceText: $('#refText').value.trim() || null,
      sliders: cloneSliders,
      btn: cbtn,
      progressId: '#cloneGenProgress',
      statusId: '#cloneGenStatus',
      statusTxtId: '#cloneGenStatusTxt',
      stopId: '#cloneGenStopBtn',
      resultCard: $('#cloneResultCard'),
      player: clonePlayer,
      metaId: '#cloneResultMeta',
      voiceLabel: `клон · ${App.reference.name || 'reference'}`,
    });
  });

  $('#cloneDownloadBtn').addEventListener('click', () => downloadCurrent(clonePlayer, 'higgs_clone'));
}

async function uploadReference(file) {
  const ok = /\.(wav|mp3)$/i.test(file.name) || /audio\//.test(file.type);
  if (!ok) { toast('Поддерживаются .wav и .mp3', 'err'); return; }
  const info = $('#refFileInfo');
  $('#refFileName').textContent = file.name;
  $('#refFileMeta').textContent = 'загрузка…';
  info.hidden = false;
  $('#dropzoneInner').style.opacity = '.5';
  try {
    const fd = new FormData(); fd.append('file', file);
    const res = await api.post('/api/upload_reference', fd, true);
    if (!res || res.ok === false) throw new Error((res && res.error) || 'Не удалось загрузить');
    App.reference = { ref_id: res.ref_id, name: res.name || file.name, duration_s: res.duration_s };
    $('#refFileName').textContent = App.reference.name;
    $('#refFileMeta').textContent = res.duration_s != null
      ? `${res.duration_s.toFixed(1)} с · готов к клонированию`
      : 'готов к клонированию';
    toast('Референс загружен', 'ok');
  } catch (e) {
    console.warn('upload failed:', e);
    toast(String(e.message || e), 'err', 5000);
    clearReference();
  } finally {
    $('#dropzoneInner').style.opacity = '';
  }
}
function clearReference() {
  App.reference = { ref_id: null, name: null, duration_s: null };
  $('#refFileInfo').hidden = true;
  $('#refFile').value = '';
}

function downloadCurrent(player, base) {
  if (!player || !player.url) { toast('Нет аудио для скачивания', 'err'); return; }
  const a = el('a');
  a.href = player.url;
  a.download = `${base}_${new Date().toISOString().slice(0,19).replace(/[:T]/g,'-')}.wav`;
  document.body.appendChild(a); a.click(); a.remove();
}

/* ------------------------------------------------------------
   11. History (localStorage)
   ------------------------------------------------------------ */
const HIST_KEY = 'hg_history';
const HIST_MAX = 40;
function getHistory() { return LS.get(HIST_KEY, []); }
function addHistory(item) {
  const list = getHistory();
  list.unshift(item);
  // keep audio_b64 only for the most recent few (storage budget)
  list.forEach((it, i) => { if (i >= 6 && it.audio_b64) delete it.audio_b64; });
  while (list.length > HIST_MAX) list.pop();
  try { LS.set(HIST_KEY, list); }
  catch { list.forEach(it => delete it.audio_b64); LS.set(HIST_KEY, list); }
}
function renderHistory() {
  const list = getHistory();
  const host = $('#historyList');
  const empty = $('#historyEmpty');
  host.innerHTML = '';
  empty.classList.toggle('show', list.length === 0);
  list.forEach(item => host.appendChild(historyRow(item)));
}
function historyRow(item) {
  const row = el('div', 'histrow');
  const snippet = (item.text || '').replace(/<\|[^|]*\|>/g, '').trim().slice(0, 90) || '(без текста)';
  const dur = item.duration_s != null ? fmtTime(item.duration_s) : '—';
  const date = new Date(item.ts || item.id).toLocaleString('ru-RU', { day:'2-digit', month:'short', hour:'2-digit', minute:'2-digit' });
  row.innerHTML = `
    <button class="histrow__play" title="Воспроизвести" aria-label="Воспроизвести">
      <svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>
    </button>
    <div class="histrow__main">
      <span class="histrow__text">${escapeHtml(snippet)}</span>
      <span class="histrow__sub">
        <span class="chip">${escapeHtml(item.voice || '—')}</span>
        <span>⏱ ${dur}</span><span>${escapeHtml(date)}</span>
      </span>
    </div>
    <div class="histrow__actions">
      <button class="iconbtn iconbtn--sm" data-act="dl" title="Скачать WAV" aria-label="Скачать">
        <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="M7 10l5 5 5-5"/><path d="M12 15V3"/></svg>
      </button>
      <button class="iconbtn iconbtn--sm" data-act="regen" title="Сгенерировать снова" aria-label="Заново">
        <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v5h5"/><path d="M3.05 13A9 9 0 1 0 6 5.3L3 8"/></svg>
      </button>
    </div>`;

  // resolve a playable source: prefer stored b64, else /api/audio/{file}
  const sourceBlobUrl = async () => {
    if (item.audio_b64) return URL.createObjectURL(b64ToBlob(item.audio_b64));
    if (item.output_file) {
      const fname = String(item.output_file).split(/[\\/]/).pop();
      try {
        const r = await fetch(`/api/audio/${encodeURIComponent(fname)}`);
        if (r.ok) return URL.createObjectURL(await r.blob());
      } catch {}
    }
    return null;
  };

  const audio = new Audio();
  $('.histrow__play', row).addEventListener('click', async () => {
    if (!audio.paused) { audio.pause(); return; }
    if (!audio.src) {
      const url = await sourceBlobUrl();
      if (!url) { toast('Аудио недоступно (файл не найден)', 'err'); return; }
      audio.src = url;
    }
    audio.play().catch(() => toast('Не удалось воспроизвести', 'err'));
  });

  $('[data-act="dl"]', row).addEventListener('click', async () => {
    const url = await sourceBlobUrl();
    if (!url) { toast('Файл недоступен для скачивания', 'err'); return; }
    const a = el('a'); a.href = url; a.download = `higgs_${item.id}.wav`;
    document.body.appendChild(a); a.click(); a.remove();
  });

  $('[data-act="regen"]', row).addEventListener('click', () => {
    switchTab('generate');
    const ta = $('#genText'); ta.value = item.text || ''; ta.dispatchEvent(new Event('input'));
    toast('Текст загружен — нажмите «Сгенерировать»', 'info');
  });

  return row;
}

/* ------------------------------------------------------------
   12. Settings tab + persistence
   ------------------------------------------------------------ */
function effectiveDefaults() {
  return Object.assign({}, App.defaults, App.userDefaults || {});
}
function renderEngineKV() {
  const hw = (App.state && App.state.hardware) || {};
  const st = App.state || {};
  const info = App.info || {};
  const rows = [
    ['Устройство', hw.device === 'cuda' ? `GPU (CUDA)` : (hw.device === 'cpu' ? 'CPU' : '—'), hw.device === 'cuda'],
    ['Видеокарта', hw.gpu_name || (hw.device === 'cpu' ? 'не используется' : '—'), false],
    ['VRAM', hw.vram_total_gb != null ? `${hw.vram_free_gb != null ? hw.vram_free_gb.toFixed(1)+' / ' : ''}${hw.vram_total_gb.toFixed(1)} ГБ` : '—', false],
    ['Точность', (hw.precision || '—'), true],
    ['Статус', App.state ? phaseRu(st.phase) : '—', App.ready],
  ];
  if (st.phase === 'idle' && st.idle_remaining_s != null) {
    rows.push(['Модель', 'выгружена из VRAM · загрузится по запросу', false]);
  } else if (st.idle_remaining_s != null) {
    const remaining = Math.round(st.idle_remaining_s);
    if (remaining > 0) {
      rows.push(['До выгрузки', `${remaining} с (≈${Math.round(remaining / 60)} мин)`, false]);
    }
  }
  if (hw.reason) rows.push(['Почему так', hw.reason, false]);
  $('#engineKV').innerHTML = rows.map(([k, v, accent]) =>
    `<div class="kv__row"><span class="kv__k">${k}</span><span class="kv__v${accent ? ' accent':''}">${escapeHtml(String(v))}</span></div>`).join('');
}
function setupSettings() {
  defaultSliders = buildSliders('#defaultSamplingSliders', effectiveDefaults());
  $('#btnSaveDefaults').addEventListener('click', () => {
    App.userDefaults = readSliders(defaultSliders);
    LS.set('hg_defaults', App.userDefaults);
    setSliders(genSliders, effectiveDefaults());
    setSliders(cloneSliders, effectiveDefaults());
    toast('Параметры по умолчанию сохранены', 'ok');
  });
}

/* ------------------------------------------------------------
   13a. Voices manager (Settings tab)
   ------------------------------------------------------------ */
function setupPresets() {
  const dz = $('#presetDropzone');
  const input = $('#presetFile');
  if (!dz) return;

  dz.addEventListener('click', () => input.click());
  dz.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); } });
  input.addEventListener('change', () => { if (input.files[0]) uploadPreset(input.files[0]); });

  ['dragenter','dragover'].forEach(ev => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add('is-drag'); }));
  ['dragleave','drop'].forEach(ev => dz.addEventListener(ev, (e) => { e.preventDefault(); if (ev !== 'dragleave' || !dz.contains(e.relatedTarget)) dz.classList.remove('is-drag'); }));
  dz.addEventListener('drop', (e) => { const f = e.dataTransfer.files[0]; if (f) uploadPreset(f); });

  $('#btnRefreshVoices').addEventListener('click', renderVoicesList);

  // Re-render when presets tab becomes visible (in case list changed)
  const presetsTab = document.querySelector('[data-tab="presets"]');
  if (presetsTab) {
    presetsTab.addEventListener('click', () => {
      setTimeout(renderVoicesList, 100);
    });
  }
}

async function uploadPreset(file) {
  const ok = /\.(wav|mp3)$/i.test(file.name) || /audio\//.test(file.type);
  if (!ok) { toast('Поддерживаются .wav, .mp3, .ogg, .flac, .m4a', 'err'); return; }
  try {
    const fd = new FormData(); fd.append('file', file);
    fd.append('name', file.name.replace(/\.\w+$/, '').replace(/[_\s]+/g, ' ').trim());
    const res = await api.post('/api/upload_preset', fd, true);
    if (!res || res.ok === false) throw new Error((res && res.error) || 'Ошибка загрузки');
    toast(`Пресет «${res.name}» загружен (${res.duration_s ? res.duration_s.toFixed(1) + ' с' : 'OK'})`, 'ok');
    // Refresh voices everywhere
    if (res.voices) {
      App.info = App.info || {};
      App.info.voices = res.voices;
      buildVoices();
    }
    renderVoicesList();
  } catch (e) {
    toast(String(e.message || e), 'err', 5000);
  }
}

async function deletePreset(presetId) {
  try {
    const res = await api.delete('/api/voices/' + encodeURIComponent(presetId));
    if (res.voices) {
      App.info = App.info || {};
      App.info.voices = res.voices;
      buildVoices();
    }
    renderVoicesList();
    toast('Пресет удалён', 'info');
  } catch (e) {
    toast(String(e.message || e), 'err', 5000);
  }
}

function renderVoicesList() {
  const host = $('#voicesList');
  if (!host) return;
  const voices = (App.info && App.info.voices) || [];
  if (voices.length === 0) {
    host.innerHTML = '<p class="field__desc" style="padding:8px 0;text-align:center">Нет пресет-голосов. Загрузите .wav выше.</p>';
    return;
  }
  host.innerHTML = voices.map(v => `
    <div class="voice-row">
      <span class="voice-row__icon">
        <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
          <path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/>
        </svg>
      </span>
      <div class="voice-row__body">
        <div class="voice-row__name">${escapeHtml(v.name)}</div>
        <div class="voice-row__desc">${escapeHtml(v.description || 'Пресет-голос')}</div>
      </div>
      <button class="voice-row__del" data-preset-id="${escapeHtml(v.id)}" title="Удалить" aria-label="Удалить">
        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M18 6 6 18M6 6l12 12"/></svg>
      </button>
    </div>
  `).join('');
  // Wire delete buttons
  $$('.voice-row__del', host).forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      deletePreset(btn.dataset.presetId);
    });
  });
}

/* ------------------------------------------------------------
   13. Settings quick sheet (gear)
   ------------------------------------------------------------ */
function setupSheet() {
  const sheet = $('#settingsSheet');
  const open = () => { renderSheetHw(); sheet.hidden = false; };
  const close = () => { sheet.hidden = true; };
  $('#btnSettingsGear').addEventListener('click', open);
  $$('[data-close-sheet]').forEach(b => b.addEventListener('click', close));
  $$('[data-goto-tab]').forEach(b => b.addEventListener('click', () => { switchTab(b.dataset.gotoTab); close(); }));
  $('#sheetReload').addEventListener('click', () => { close(); kickLoad(); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && !sheet.hidden) close(); });
}
function renderSheetHw() {
  const hw = (App.state && App.state.hardware) || {};
  const st = App.state || {};
  const rows = [
    ['Устройство', hw.device === 'cuda' ? 'GPU' : (hw.device || '—')],
    ['Точность', hw.precision || '—'],
    ['Статус', st ? phaseRu(st.phase) : '—'],
  ];
  if (st.phase === 'idle') {
    rows.push(['Модель', 'выгружена · авто-загрузка']);
  } else if (st.phase === 'ready' && st.idle_remaining_s != null) {
    const remaining = Math.round(st.idle_remaining_s);
    if (remaining > 0) {
      rows.push(['До выгрузки', `${remaining} с`]);
    }
  }
  $('#sheetHw').innerHTML = rows.map(([k, v]) =>
    `<div class="kv__row"><span class="kv__k">${k}</span><span class="kv__v">${escapeHtml(String(v))}</span></div>`
  ).join('');
}

/* ------------------------------------------------------------
   14. Hardware badge + footer
   ------------------------------------------------------------ */
function renderHardware() {
  const hw = (App.state && App.state.hardware) || {};
  const st = App.state || {};
  const badge = $('#hwBadge');
  const text = $('#hwText');
  badge.dataset.device = hw.device || '';
  if (st.phase === 'idle' && st.idle_remaining_s != null) {
    // Модель выгружена — показываем статус
    const mins = Math.round(st.idle_remaining_s / 60);
    badge.dataset.device = 'idle';
    text.textContent = `Выгружена · авто-загрузка · простояла ${mins} мин`;
    badge.title = 'Модель выгружена из VRAM. При генерации загрузится автоматически.';
  } else if (hw.device === 'cuda') {
    const vram = hw.vram_total_gb != null ? `${Math.round(hw.vram_total_gb)}GB` : '';
    const name = (hw.gpu_name || 'GPU').replace(/NVIDIA\s*/i, '').trim();
    text.textContent = `GPU · ${name}${vram ? ' · ' + vram : ''} · ${hw.precision || 'bf16'}`;
    if (hw.reason) badge.title = hw.reason;
  } else if (hw.device === 'cpu') {
    text.textContent = `CPU · ${hw.precision || 'fp32'}`;
    if (hw.reason) badge.title = hw.reason;
  } else {
    text.textContent = 'Определение оборудования…';
    if (hw.reason) badge.title = hw.reason;
  }
  renderEngineKV();
}
function setFooter(stateTxt) { if (stateTxt) $('#footerState').textContent = stateTxt; }

/* ------------------------------------------------------------
   15. /api/info — wiring branding, voices, tokens, defaults
   ------------------------------------------------------------ */
async function loadInfo() {
  try {
    App.info = await api.get('/api/info');
  } catch (e) {
    console.warn('info unavailable, using fallback catalog:', e.message);
    App.info = null;
  }
  if (App.info && App.info.defaults) App.defaults = Object.assign({}, App.defaults, App.info.defaults);

  // version / model / branding
  const ver = (App.info && App.info.version) || '3.0';
  $('#aboutVersion').textContent = 'версия ' + ver;
  $('#footerVer').textContent = 'v' + ver;
  if (App.info && App.info.model_id) $('#aboutModel').textContent = App.info.model_id;

  buildPills();
  buildVoices();
  renderVoicesList();

  const eff = effectiveDefaults();
  genSliders   = buildSliders('#samplingSliders', eff);
  cloneSliders = buildSliders('#cloneSamplingSliders', eff);
  // default Top-P to 1.0 (off) explicitly
  setSliders(genSliders, eff); setSliders(cloneSliders, eff);
}

/* ------------------------------------------------------------
   16. Loading modal + SSE state stream
   ------------------------------------------------------------ */
function phaseRu(phase) {
  return ({
    idle:'ожидание', downloading:'скачивание весов', converting:'конвертация',
    loading:'загрузка в память', ready:'готов', error:'ошибка',
  })[phase] || phase || '—';
}
const PHASE_ORDER = ['downloading','converting','loading','ready'];

function openLoader() { const l = $('#loader'); l.hidden = false; l.classList.remove('is-out'); }
function closeLoader() {
  const l = $('#loader');
  l.classList.add('is-out');
  setTimeout(() => { l.hidden = true; }, 650);
}

function updateModelStatus(phase, st) {
  const update = (elId, txtId) => {
    const el = $(elId);
    const txt = $(txtId);
    if (!el || !txt) return;
    if (phase === 'idle') {
      el.className = 'model-status model-status--idle';
      txt.textContent = 'Модель выгружена из VRAM · начните генерацию — загрузится автоматически';
      el.hidden = false;
    } else if (phase === 'loading') {
      el.className = 'model-status model-status--loading';
      txt.textContent = 'Модель загружается в VRAM…';
      el.hidden = false;
    } else {
      el.hidden = true;
    }
  };
  update('#modelStatus', '#modelStatusTxt');
  update('#cloneModelStatus', '#cloneModelStatusTxt');
}

function applyState(st) {
  App.state = st;
  renderHardware();
  renderSheetHw();

  const phase = st.phase || 'idle';
  const progress = clamp(Number(st.progress || 0), 0, 1);

  // footer
  setFooter(phaseRu(phase));

  // model status bar (idle / loading)
  updateModelStatus(phase, st);

  // phase pills
  const idx = PHASE_ORDER.indexOf(phase);
  $$('#loaderPhases .phase').forEach(p => {
    const i = PHASE_ORDER.indexOf(p.dataset.phase);
    p.classList.toggle('is-active', p.dataset.phase === phase);
    p.classList.toggle('is-done', idx > -1 && i < idx);
  });

  // progress bar
  const fill = $('#loaderFill');
  const bar = $('#loaderFill').parentElement;
  const hasProgress = st.progress != null && progress > 0;
  bar.classList.toggle('is-determinate', hasProgress);
  fill.style.width = (hasProgress ? progress * 100 : (phase === 'ready' ? 100 : 8)) + '%';
  $('#loaderPct').textContent = hasProgress ? Math.round(progress * 100) + '%' : '';
  $('#loaderMsg').textContent = st.message || defaultPhaseMsg(phase);

  // error / ready handling
  const errBox = $('#loaderError');
  if (phase === 'error') {
    errBox.hidden = false;
    $('#loaderErrText').textContent = st.error || st.message || 'Не удалось подготовить модель.';
    openLoader();
    App.ready = false;
  } else {
    errBox.hidden = true;
  }

  if (phase === 'ready') {
    if (!App.ready) toast('Модель готова к работе', 'ok');
    App.ready = true;
    fill.style.width = '100%';
    closeLoader();
  } else if (phase !== 'idle') {
    App.ready = false;
    openLoader();
  }
}
function defaultPhaseMsg(phase) {
  return ({
    downloading:'Скачивание весов модели (~9.3 ГБ)…',
    converting:'Конвертация и подготовка весов…',
    loading:'Загрузка модели в память…',
    ready:'Готово!',
    idle:'Ожидание запуска…',
    error:'Произошла ошибка.',
  })[phase] || 'Подготовка движка…';
}

/* Subscribe to /api/events (SSE); fall back to polling /api/state. */
function subscribeState() {
  if (App.es) { try { App.es.close(); } catch {} App.es = null; }
  if (App.pollTimer) { clearInterval(App.pollTimer); App.pollTimer = null; }

  const startPolling = () => {
    if (App.pollTimer) return;
    const tick = async () => { try { applyState(await api.get('/api/state')); } catch {} };
    tick();
    App.pollTimer = setInterval(tick, 1000);
  };

  if (typeof window.EventSource === 'function') {
    try {
      const es = new EventSource('/api/events');
      App.es = es;
      es.onmessage = (ev) => {
        try { applyState(JSON.parse(ev.data)); } catch (e) { console.warn('bad SSE payload', e); }
      };
      es.onerror = () => {
        // SSE dropped — close and fall back to polling
        try { es.close(); } catch {}
        App.es = null;
        startPolling();
      };
    } catch {
      startPolling();
    }
  } else {
    startPolling();
  }
}

async function kickLoad() {
  openLoader();
  $('#loaderError').hidden = true;
  $('#loaderMsg').textContent = 'Запуск подготовки модели…';
  try { await api.post('/api/load', {}); }
  catch (e) {
    // If load endpoint fails AND we have no state stream, surface a soft error.
    console.warn('load failed:', e.message);
  }
}

/* ------------------------------------------------------------
   17. Boot
   ------------------------------------------------------------ */
async function init() {
  setupTabs();
  setupSheet();

  // initial pills/sliders/voices from fallback so UI is alive before /api/info
  buildPills();
  const eff = effectiveDefaults();
  genSliders   = buildSliders('#samplingSliders', eff);
  cloneSliders = buildSliders('#cloneSamplingSliders', eff);

  genPlayer = new Player({ audio:'#audioEl', playBtn:'#playBtn', canvas:'#waveform', curTime:'#curTime', totTime:'#totTime' });
  clonePlayer = new Player({ audio:'#cloneAudioEl', playBtn:'#clonePlayBtn', canvas:'#cloneWaveform', curTime:'#cloneCurTime', totTime:'#cloneTotTime' });

  setupGenerate();
  setupClone();
  setupSettings();
  setupPresets();

  $('#btnClearHistory').addEventListener('click', () => {
    LS.set(HIST_KEY, []); renderHistory(); toast('История очищена', 'info');
  });
  $('#loaderRetry').addEventListener('click', kickLoad);

  // Load info (branding, voices, tokens) then current state + stream
  await loadInfo();

  // Fetch one state snapshot to decide whether to show loader
  let st = null;
  try { st = await api.get('/api/state'); } catch (e) { console.warn('state unavailable:', e.message); }

  if (st) {
    applyState(st);
    subscribeState();
    // if engine idle, request preparation
    if (st.phase === 'idle') kickLoad();
  } else {
    // No backend reachable — offline preview mode. Hide loader, mark not-ready.
    closeLoader();
    setFooter('офлайн-режим (нет связи с движком)');
    toast('Нет связи с движком — превью-режим интерфейса', 'info', 5000);
    // still try to subscribe in case backend comes up
    subscribeState();
  }
}

document.addEventListener('DOMContentLoaded', init);
