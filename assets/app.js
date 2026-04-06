/* FAIRE — PraCha Labs Evolution Dashboard
 * Update tracking: edit JSON files in /data and /weekly, push to main.
 * GitHub Actions deploys automatically to GitHub Pages.
 */

// ── Config ──────────────────────────────────────────────────────────────────
const GITHUB_REPO = 'prabakaranc98/FAIRE';

// ── State ────────────────────────────────────────────────────────────────────
const D = { scorecard: null, okrs: null, pillars: null, velocity: null, weeks: {} };
let activeWeek = null;
let velocityChart = null;
let radarChart = null;

// ── Fetch helpers ─────────────────────────────────────────────────────────────
async function json(path) {
  try {
    const r = await fetch(path + '?_=' + Date.now());
    return r.ok ? await r.json() : null;
  } catch { return null; }
}

async function loadAll() {
  const [sc, okrs, pillars, vel, weeksMeta] = await Promise.all([
    json('data/scorecard.json'),
    json('data/okrs.json'),
    json('data/pillars.json'),
    json('data/velocity.json'),
    json('data/weeks.json'),
  ]);
  D.scorecard = sc;
  D.okrs      = okrs;
  D.pillars   = pillars;
  D.velocity  = vel;

  const weekList = weeksMeta?.weeks ?? [];
  await Promise.all(weekList.map(async w => {
    const d = await json(`weekly/${w}.json`);
    if (d) D.weeks[w] = d;
  }));
}

// ── Header ────────────────────────────────────────────────────────────────────
function renderHeader() {
  const now = new Date();
  const jan1 = new Date(now.getFullYear(), 0, 1);
  const weekNum = Math.ceil(((now - jan1) / 86400000 + jan1.getDay() + 1) / 7);
  const monthStr = now.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });

  document.getElementById('badge-week').textContent = `W${weekNum} · ${monthStr}`;

  // Phase
  const m = now.getMonth();
  let phase = 'Q2 Foundation Sprint';
  if (m >= 6 && m <= 8) phase = 'Q3 Ship & Scale';
  else if (m >= 9)      phase = 'Q4 Application Blitz';
  document.getElementById('badge-phase').textContent = phase;

  // Days remaining
  const end = new Date('2026-12-24');
  const days = Math.max(0, Math.round((end - now) / 86400000));
  document.getElementById('badge-days').textContent = `${days} days left`;
}

// ── Scorecard ─────────────────────────────────────────────────────────────────
function renderScorecard() {
  const el = document.getElementById('scorecard-grid');
  if (!D.scorecard) { el.innerHTML = '<p class="empty-state">No scorecard data.</p>'; return; }

  const metrics = D.scorecard.metrics;
  const totalCurrent = metrics.reduce((s, m) => s + m.current, 0);

  let html = '';
  for (const m of metrics) {
    const pct = m.target > 0 ? Math.min(Math.round((m.current / m.target) * 100), 100) : 0;
    const barClass = pct === 0 ? 'bar-none' : pct < 30 ? 'bar-low' : pct < 70 ? 'bar-mid' : 'bar-high';
    html += `
      <div class="metric-card">
        <div class="metric-header">
          <span class="metric-icon">${m.icon}</span>
          <span class="metric-label">${m.label}</span>
        </div>
        <div class="metric-nums">
          <span class="metric-current">${m.current}</span>
          <span class="metric-target">/ ${m.target}</span>
        </div>
        <div class="metric-pct">${pct}%</div>
        <div class="bar"><div class="bar-fill ${barClass}" style="width:${pct}%"></div></div>
      </div>`;
  }

  // Total card
  const totalPct = Math.min(Math.round((totalCurrent / 30) * 100), 100);
  html += `
    <div class="metric-card total-card">
      <div class="metric-header">
        <span class="metric-icon">🏆</span>
        <span class="metric-label">TOTAL ARTIFACTS</span>
      </div>
      <div class="metric-nums">
        <span class="metric-current">${totalCurrent}</span>
        <span class="metric-target">/ 30+ target</span>
      </div>
      <div class="metric-pct">${totalPct}% to target</div>
      <div class="bar"><div class="bar-fill bar-mid" style="width:${totalPct}%"></div></div>
    </div>`;

  el.innerHTML = html;
}

// ── OKRs ──────────────────────────────────────────────────────────────────────
function scoreClass(s) {
  if (s >= 0.7) return 'sc-green';
  if (s >= 0.4) return 'sc-yellow';
  if (s > 0)    return 'sc-red';
  return 'sc-gray';
}

function renderOKRs() {
  const el = document.getElementById('okr-grid');
  const osEl = document.getElementById('overall-score-fill');
  const osVal = document.getElementById('overall-score-val');

  if (!D.okrs) { el.innerHTML = '<p class="empty-state">No OKR data.</p>'; return; }

  // Overall score
  const overall = D.okrs.overallScore ?? 0;
  if (osEl) osEl.style.width = Math.round(overall * 100) + '%';
  if (osVal) {
    osVal.textContent = overall.toFixed(1);
    osVal.className = 'os-value ' + scoreClass(overall);
  }

  let html = '';
  for (const okr of D.okrs.okrs) {
    const sc = okr.score ?? 0;
    const pct = Math.round(sc * 100);
    const krsHtml = okr.krs.map(kr => {
      const done = kr.current >= kr.target;
      return `<li class="okr-kr ${done ? 'kr-done' : ''}">
        <span class="kr-id">${kr.id}</span>
        <span>${kr.text}</span>
        <span class="metric-target" style="margin-left:auto;white-space:nowrap">${kr.current}/${kr.target}</span>
      </li>`;
    }).join('');
    const blockerHtml = okr.blocker
      ? `<div class="okr-blocker">⚠ ${okr.blocker}</div>`
      : '';
    html += `
      <div class="okr-card clr-${okr.color}">
        <div class="okr-top">
          <div>
            <div class="okr-title">${okr.title}</div>
            <div class="okr-objective">${okr.objective}</div>
          </div>
          <span class="okr-score ${scoreClass(sc)}">${sc.toFixed(1)}</span>
        </div>
        <div class="okr-bar"><div class="okr-bar-fill" style="width:${pct}%"></div></div>
        <ul class="okr-krs">${krsHtml}</ul>
        ${blockerHtml}
      </div>`;
  }
  el.innerHTML = html;
}

// ── Pillars ───────────────────────────────────────────────────────────────────
function renderPillars() {
  const el = document.getElementById('pillar-grid');
  if (!D.pillars) { el.innerHTML = '<p class="empty-state">No pillar data.</p>'; return; }

  const heatEmoji = { hot: '🔥', warm: '🟡', cold: '⬜' };
  const modeLabel = { Lab: '🔬 Lab', 'Think Tank': '🧠 Think Tank', Startup: '🚀 Startup', Institute: '🏛️ Institute' };

  let html = '';
  for (const p of D.pillars.pillars) {
    const notionHref = p.notionUrl ? `href="${p.notionUrl}" target="_blank" rel="noopener"` : '';
    html += `
      <div class="pillar-card heat-${p.heat}" title="${p.name}">
        ${p.notionUrl ? `<a class="pillar-link" ${notionHref}></a>` : ''}
        <div class="pillar-icon">${p.icon}</div>
        <div class="pillar-name">${p.name}</div>
        <div class="pillar-tag">${p.tagline}</div>
        ${p.mode   ? `<div class="pillar-mode">${modeLabel[p.mode] ?? p.mode}</div>` : ''}
        ${p.focus  ? `<div class="pillar-focus">${p.focus}</div>` : ''}
        ${p.artifact ? `<div class="pillar-artifact">✓ ${p.artifact}</div>` : ''}
        <div class="pillar-heat">${heatEmoji[p.heat] ?? '⬜'}</div>
      </div>`;
  }
  el.innerHTML = html;
}

// ── Charts ────────────────────────────────────────────────────────────────────
const CHART_DEFAULTS = {
  color: '#8b949e',
  gridColor: 'rgba(48,54,61,0.8)',
  font: { family: 'inherit', size: 11 },
};

function renderVelocityChart() {
  const ctx = document.getElementById('velocity-chart');
  if (!ctx || !D.velocity) return;
  if (velocityChart) velocityChart.destroy();

  const months = D.velocity.months;
  const labels = months.map(m => m.month);

  // Compute cumulative
  const cumulate = key => {
    let sum = 0;
    return months.map(m => (sum += (m[key] || 0)));
  };

  const datasets = [
    { label: 'Papers',  data: cumulate('papers'),  borderColor: '#bc8cff', backgroundColor: 'rgba(188,140,255,0.15)', fill: true, tension: 0.35 },
    { label: 'Repos',   data: cumulate('repos'),   borderColor: '#3fb950', backgroundColor: 'rgba(63,185,80,0.12)',   fill: true, tension: 0.35 },
    { label: 'Blogs',   data: cumulate('blogs'),   borderColor: '#58a6ff', backgroundColor: 'rgba(88,166,255,0.12)', fill: true, tension: 0.35 },
    { label: 'Talks',   data: cumulate('talks'),   borderColor: '#ffa657', backgroundColor: 'rgba(255,166,87,0.12)', fill: true, tension: 0.35 },
    { label: 'Mastery', data: cumulate('mastery'), borderColor: '#d29922', backgroundColor: 'rgba(210,153,34,0.1)',  fill: true, tension: 0.35 },
  ];

  // Target line
  const target = D.velocity.target || 30;
  datasets.push({
    label: `Target (${target})`,
    data: months.map(() => target),
    borderColor: 'rgba(255,123,114,0.4)',
    borderDash: [6, 3],
    borderWidth: 1,
    pointRadius: 0,
    fill: false,
  });

  velocityChart = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { labels: { color: CHART_DEFAULTS.color, font: CHART_DEFAULTS.font, boxWidth: 12, padding: 10 } },
        tooltip: { backgroundColor: '#161b22', titleColor: '#e6edf3', bodyColor: '#8b949e', borderColor: '#30363d', borderWidth: 1 },
      },
      scales: {
        x: { ticks: { color: CHART_DEFAULTS.color, font: CHART_DEFAULTS.font }, grid: { color: CHART_DEFAULTS.gridColor } },
        y: { beginAtZero: true, ticks: { color: CHART_DEFAULTS.color, font: CHART_DEFAULTS.font }, grid: { color: CHART_DEFAULTS.gridColor } },
      },
    },
  });
}

function renderRadarChart() {
  const ctx = document.getElementById('okr-radar');
  if (!ctx || !D.okrs) return;
  if (radarChart) radarChart.destroy();

  const labels = D.okrs.okrs.map(o => o.title);
  const scores = D.okrs.okrs.map(o => o.score ?? 0);

  radarChart = new Chart(ctx, {
    type: 'radar',
    data: {
      labels,
      datasets: [{
        label: 'OKR Score',
        data: scores,
        borderColor: '#bc8cff',
        backgroundColor: 'rgba(188,140,255,0.15)',
        pointBackgroundColor: '#bc8cff',
        pointRadius: 4,
        borderWidth: 2,
      }],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: { backgroundColor: '#161b22', titleColor: '#e6edf3', bodyColor: '#8b949e', borderColor: '#30363d', borderWidth: 1 },
      },
      scales: {
        r: {
          min: 0, max: 1,
          ticks: { stepSize: 0.2, color: CHART_DEFAULTS.color, backdropColor: 'transparent', font: CHART_DEFAULTS.font },
          grid: { color: CHART_DEFAULTS.gridColor },
          angleLines: { color: CHART_DEFAULTS.gridColor },
          pointLabels: { color: '#e6edf3', font: { size: 11 } },
        },
      },
    },
  });
}

// ── Weekly Log ────────────────────────────────────────────────────────────────
const MODE_META = [
  { key: 'lab',        label: '🔬 Lab',        cls: 'mode-lab' },
  { key: 'think_tank', label: '🧠 Think Tank', cls: 'mode-think' },
  { key: 'startup',    label: '🚀 Startup',     cls: 'mode-startup' },
  { key: 'institute',  label: '🏛️ Institute',  cls: 'mode-institute' },
];

function renderWeeklyLog() {
  const weeks = Object.keys(D.weeks).sort().reverse();

  if (weeks.length === 0) {
    document.getElementById('week-nav').innerHTML   = '';
    document.getElementById('weekly-log').innerHTML = '<p class="empty-state">No weekly logs yet.</p>';
    return;
  }

  if (!activeWeek || !D.weeks[activeWeek]) activeWeek = weeks[0];

  // Tabs
  const tabsHtml = weeks.map(w => `
    <button class="week-tab${w === activeWeek ? ' active' : ''}" onclick="selectWeek('${w}')">${w}</button>
  `).join('');
  document.getElementById('week-nav').innerHTML = `<div class="week-tabs">${tabsHtml}</div>`;

  renderWeekEntry(activeWeek);
}

function renderWeekEntry(wid) {
  const w = D.weeks[wid];
  if (!w) return;

  const modesHtml = MODE_META.map(m => {
    const items = w.entries?.[m.key] ?? [];
    const bodyHtml = items.length > 0
      ? `<ul class="mode-items">${items.map(i => `<li class="mode-item"><span>${i}</span></li>`).join('')}</ul>`
      : `<span class="mode-empty">Nothing logged yet.</span>`;
    return `
      <div class="mode-section">
        <div class="mode-title ${m.cls}">${m.label}</div>
        ${bodyHtml}
      </div>`;
  }).join('');

  // Edit link
  const editHref = GITHUB_REPO && !GITHUB_REPO.includes('YOUR_GITHUB')
    ? `https://github.com/${GITHUB_REPO}/edit/main/weekly/${wid}.json`
    : '#';
  const editLink = editHref !== '#'
    ? `<a class="week-edit-link" href="${editHref}" target="_blank" rel="noopener">✏ Edit in GitHub</a>`
    : '';

  document.getElementById('weekly-log').innerHTML = `
    <div class="week-entry">
      <div class="week-entry-header">
        <div>
          <span class="week-entry-title">${w.week} · ${w.dates}</span>
          <div class="week-entry-sprint">${w.phase} · ${w.sprint}</div>
        </div>
        ${editLink}
      </div>
      <div class="week-modes">${modesHtml}</div>
      <div class="week-footer">
        <div>
          <div class="wf-label">Artifacts</div>
          <div class="wf-value">${w.artifacts || '—'}</div>
        </div>
        <div>
          <div class="wf-label">Key Decisions</div>
          <div class="wf-value">${w.key_decisions || '—'}</div>
        </div>
        <div>
          <div class="wf-label">Next Week</div>
          <div class="wf-value">${w.next_week || '—'}</div>
        </div>
      </div>
    </div>`;
}

// Exposed globally for onclick handlers
window.selectWeek = function(w) {
  activeWeek = w;
  renderWeeklyLog();
};

// ── Footer ────────────────────────────────────────────────────────────────────
function renderFooter() {
  const dates = [D.scorecard?.lastUpdated, D.okrs?.lastUpdated, D.pillars?.lastUpdated].filter(Boolean);
  if (!dates.length) return;
  const latest = dates.sort().reverse()[0];
  const el = document.getElementById('footer-sync');
  if (el) el.textContent = `Last synced: ${latest}`;
}

// ── Boot ──────────────────────────────────────────────────────────────────────
async function init() {
  await loadAll();
  renderHeader();
  renderScorecard();
  renderOKRs();
  renderPillars();
  renderVelocityChart();
  renderRadarChart();
  renderWeeklyLog();
  renderFooter();
}

document.addEventListener('DOMContentLoaded', init);
