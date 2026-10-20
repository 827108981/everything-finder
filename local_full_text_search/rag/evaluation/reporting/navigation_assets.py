"""离线报告的详情导航资源，生成时内嵌进单个 HTML 文件。"""

NAVIGATION_CSS = r"""
.report-viewer{display:grid;grid-template-columns:minmax(270px,310px) minmax(0,1fr);gap:18px;align-items:start}
.report-viewer.nav-hidden{grid-template-columns:56px minmax(0,1fr)}
.report-nav{position:sticky;top:calc(var(--report-header-height, 132px) + 12px);max-height:calc(100vh - var(--report-header-height, 132px) - 28px);overflow:auto;align-self:start;background:#fff;border:1px solid #d9e2ef;border-radius:12px;padding:12px;box-shadow:0 8px 24px #17203312;z-index:4}
.report-nav .nav-body{display:grid;gap:12px}
.nav-hidden .nav-body{display:none}
.report-nav .nav-toggle{width:100%;margin-bottom:8px;font-size:12px}
.report-nav label{font-weight:700;font-size:12px;color:#475467;display:block;margin-bottom:5px}
.report-nav select,.report-nav input{width:100%}
.nav-case-head{font-weight:700;line-height:1.45;overflow-wrap:anywhere}
.nav-stage-list{display:grid;grid-template-columns:1fr 1fr;gap:5px}
.nav-stage-list button{min-width:0;text-align:left;font-size:12px;padding:7px;line-height:1.35}
.nav-stage-list button.active{background:#e9f1ff;border-color:#2563eb;color:#154caa;font-weight:700}
.nav-stage-list button small{display:block;color:#667085}
.nav-stats{background:#f5f7fb;border-radius:8px;padding:9px;font-size:12px;line-height:1.7}
.nav-ranks{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:5px}
.nav-ranks button{padding:5px 0;font-size:11px;font-weight:700;min-width:0}
.nav-ranks button.gold{background:#e8f5e9;border-color:#8bc79a;color:#2e7d32}
.nav-ranks button.negative{background:#ffebee;border-color:#ef9a9a;color:#c62828}
.nav-ranks button.mixed{background:linear-gradient(90deg,#e8f5e9,#ffebee);border-left:3px solid #2e7d32;border-right:3px solid #c62828;color:#5e3152}
.nav-ranks button.unlabelled{background:#fffde7;border-color:#ecd986;color:#8a6500}
.nav-ranks button.partial_gold,.nav-ranks button.partial_negative{background:#fff8cf;border-color:#e8d575;color:#8a6500}
.nav-ranks button.active{outline:2px solid #2563eb;outline-offset:1px}
.nav-jump{display:flex;gap:6px}
.nav-jump input{min-width:0}
.nav-jump button{white-space:nowrap}
.nav-message{min-height:18px;color:#9b3b19;font-size:12px}
.report-stage-content{min-width:0}
.report-stage-content h3{margin:4px 0 10px}
.report-stage-content .evidence{background:#f8fafc}
.report-stage-content .evidence:not(.miss) button{border:0;background:transparent;color:#1756aa;padding:2px 3px;text-decoration:underline;font-size:12px}
.report-stage-content .evidence:not(.miss) button:hover{color:#0b3474}
.report-stage-content .candidate{background:#fff;scroll-margin-top:calc(var(--report-header-height, 132px) + 18px)}
.report-stage-content .candidate>summary{cursor:pointer;list-style:none;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.report-stage-content .candidate>summary::-webkit-details-marker{display:none}
.report-stage-content .candidate>summary:before{content:'▾';color:#667085}
.report-stage-content .candidate:not([open])>summary:before{content:'▸'}
.report-stage-content .candidate.gold{border-left-color:#2e7d32}
.report-stage-content .candidate.negative{border-left-color:#c62828}
.report-stage-content .candidate.mixed{border-left-color:#2e7d32;border-right:5px solid #c62828}
.report-stage-content .candidate.unlabelled{border-left-color:#f9a825}
.report-stage-content .candidate.partial_gold,.report-stage-content .candidate.partial_negative{border-left-color:#d7a324}
.report-stage-content .candidate-body{margin:10px 0 2px}
.report-stage-content .candidate-body button{border:0;background:#edf4ff;color:#1756aa;font-size:12px;padding:3px 5px;margin:2px}
.report-stage-content .candidate-body pre{max-height:none}
.report-stage-content .candidate.jump-focus{outline:3px solid #2563eb;box-shadow:0 0 0 6px #2563eb24}
.report-stage-content .stage-summary{padding:12px;background:#f5f7fb;border-radius:9px;margin:8px 0 18px}
.report-stage-content .stage-actions{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0}
.report-stage-content .stage-actions button{font-size:12px}
.report-stage-content .stage-empty{padding:20px;background:#f8fafc;border-radius:8px;color:#667085}
.report-stage-content .inline-caption{color:#667085;font-size:12px}
@media(max-width:900px){.report-viewer,.report-viewer.nav-hidden{grid-template-columns:1fr}.report-nav{max-height:40vh}.nav-ranks{grid-template-columns:repeat(10,minmax(0,1fr))}}
"""

NAVIGATION_JS = r"""
// 详情区只渲染当前题目的当前阶段；导航保持在视野里。
const ReportNav = (() => {
  const caseStages = new Map();
  const lastRanks = new Map();
  const bodyOpen = new Map();
  let currentCase = (DATA.cases.find(c => c.stage_data) || DATA.cases[0] || {}).id;
  let navHidden = false;
  let focusedRank = null;

  const key = (caseId, stage, rankValue) => JSON.stringify([caseId, stage, rankValue]);
  const current = () => caseById(currentCase);
  const stageOf = () => caseStages.get(currentCase) || DATA.config.default_stage;
  const candidatesOf = (c, stage) => (c.stage_data?.[stage]?.candidates || []);
  const ranked = (c, stage) => candidatesOf(c, stage)
    .filter(x => Number.isInteger(x.stage_rank))
    .sort((a, b) => a.stage_rank - b.stage_rank);
  const textOrDash = value => value == null ? '—' : String(value);

  function captureReadingRank() {
    const c = current();
    if (!c) return;
    // 阶段切换前同步记录折叠状态；toggle 事件可能还在浏览器队列里。
    $('caseDetails').querySelectorAll('.report-stage-content .candidate[data-body-key]')
      .forEach(card => bodyOpen.set(card.dataset.bodyKey, card.open));
    const cards = [...$('caseDetails').querySelectorAll('.report-stage-content .candidate[data-rank]')]
      .filter(card => card.dataset.rank !== '');
    if (!cards.length) return;
    if (focusedRank != null && cards.some(card => Number(card.dataset.rank) === focusedRank)) {
      lastRanks.set(key(c.id, stageOf(), 'reading'), focusedRank);
      return;
    }
    const closest = cards.reduce((best, card) => {
      const headerHeight = Number.parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--report-header-height')) || 132;
      const distance = Math.abs(card.getBoundingClientRect().top - headerHeight - 18);
      return distance < best.distance ? {card, distance} : best;
    }, {card: null, distance: Infinity}).card;
    if (closest) lastRanks.set(key(c.id, stageOf(), 'reading'), Number(closest.dataset.rank));
  }

  function evidenceHtml(e) {
    const isGold = e.kind === 'positive';
    const ranks = ORDER.map(stage => {
      const found = e.stages?.[stage] || [];
      return `<span>${esc(STAGES[stage] || stage)} ${found.length ? found.map(n => `<button type="button" data-action="rank-stage" data-stage="${esc(stage)}" data-rank="${n}" title="跳转到${esc(STAGES[stage] || stage)}第 ${n} 名">#${n}</button>`).join('、') : '未出现'}</span>`;
    }).join('　');
    return `<div class="evidence ${isGold ? 'ok' : 'neg'} ${e.first_rank == null ? 'miss' : ''}"><b>${esc(e.evidence_id)}</b> · ${esc(e.file || '')} · 第 ${esc(e.page ?? '?')} 页<br>${esc(e.quote || '')}${!isGold && e.reason ? `<br><span class="inline-caption">负例原因：${esc(e.reason)}</span>` : ''}<br><span class="inline-caption">各阶段名次：${ranks}</span></div>`;
  }

  function candidateHtml(c, stage, caseId) {
    const n = c.stage_rank;
    const bodyKey = key(caseId, stage, n == null ? `rejected:${c.chunk_id}` : n);
    const isOpen = bodyOpen.get(bodyKey) !== false;
    const labels = {gold:'金标证据',negative:'指定负例',mixed:'同块混合',unlabelled:'相似未标注',partial_gold:'部分金标',partial_negative:'部分负例'};
    const category = c.category || 'unlabelled';
    const text = String(c.included_text ?? c.text ?? '');
    // 旧结果可能只记录完整块的高亮区间；截断上下文时避免错位高亮。
    const segments = c.included_text != null && c.included_text !== c.text ? [{text, kinds:[]}] : (c.highlight_segments || [{text, kinds:[]}]);
    const content = segments.map(s => {
      const kinds = s.kinds || [];
      const color = kinds.includes('positive') && kinds.includes('hard_negative') ? 'mixed'
        : kinds.includes('positive') ? (kinds.includes('partial') ? 'partial-gold' : 'gold')
        : kinds.includes('hard_negative') ? (kinds.includes('partial') ? 'partial-negative' : 'negative') : '';
      return `<span class="hl-${color}">${esc(s.text)}</span>`;
    }).join('');
    const history = (c.rank_history || []).map(h => `<button type="button" data-action="rank-stage" data-stage="${esc(h.stage)}" data-rank="${h.rank}" title="查看同一内容块在${esc(STAGES[h.stage] || h.stage)}的名次">${esc(STAGES[h.stage] || h.stage)} #${h.rank}</button>`).join(' → ');
    const scores = `关键词 ${textOrDash(c.lexical_score)} · 向量 ${textOrDash(c.vector_score)} · 融合 ${textOrDash(c.fused_score)} · 重排 ${textOrDash(c.rerank_score)} · 约束 ${textOrDash(c.constraint_score)}`;
    const location = c.location || (c.page_number != null ? `第 ${c.page_number} 页` : '位置未记录');
    return `<details class="candidate ${esc(category)}" data-rank="${n == null ? '' : n}" data-body-key="${esc(bodyKey)}" ${isOpen ? 'open' : ''}>
      <summary><strong class="rank">${n == null ? '未入选' : '#' + n}</strong><b>${esc(c.filename || '')}</b><span class="pill">${esc(location)}</span><span class="badge ${esc(category)}">${esc(labels[category] || category)}</span><span class="score">${esc(scores)}</span></summary>
      <div class="candidate-body"><div class="inline-caption">状态：${esc(c.status || '')}；原因：${esc(c.reason || c.constraint_decision?.reason || '')}${c.truncated ? '；上下文已截断' : ''}</div>
      <div class="inline-caption">同块排名轨迹：${history || '未进入其他已记录阶段'}</div><pre>${content}</pre></div></details>`;
  }

  function navHtml(c, stage) {
    const rows = ranked(c, stage);
    const metrics = c.stage_summaries?.[stage] || {};
    const options = DATA.cases.map(x => `<option value="${esc(x.id)}" ${x.id === c.id ? 'selected' : ''}>${esc(x.id)} · ${esc(x.question.slice(0, 45))}</option>`).join('');
    const stageButtons = ORDER.map(s => `<button type="button" data-action="stage" data-stage="${esc(s)}" class="${s === stage ? 'active' : ''}" aria-pressed="${s === stage}">${esc(STAGES[s] || s)}<small>${c.stage_data?.[s]?.candidates?.length || 0} 条</small></button>`).join('');
    const rankButtons = rows.map(x => `<button type="button" class="${esc(x.category || 'unlabelled')} ${focusedRank === x.stage_rank ? 'active' : ''}" data-action="rank" data-rank="${x.stage_rank}" title="${esc(STAGES[stage] || stage)}第 ${x.stage_rank} 名：${esc(x.filename || '')}">#${x.stage_rank}</button>`).join('');
    return `<aside class="report-nav" aria-label="逐题阶段导航"><button type="button" class="nav-toggle" data-action="toggle-nav" aria-expanded="${!navHidden}">${navHidden ? '显示导航 ›' : '‹ 隐藏导航'}</button><div class="nav-body">
      <div><label for="navCase">当前题目</label><select id="navCase">${options}</select><div class="nav-case-head">${esc(c.question)}</div></div>
      <div><label>阶段导航</label><div class="nav-stage-list">${stageButtons}</div></div>
      <div class="nav-stats"><b>${esc(STAGES[stage] || stage)}</b><br>金标覆盖 ${metrics.gold_found ?? '—'}/${metrics.gold_total ?? '—'} · 首条金标 ${rank(metrics.first_gold_rank)}<br>首个负例 ${rank(metrics.hard_negative_first_rank)}</div>
      <div><label for="navRank">直接跳到名次</label><form class="nav-jump" id="navJump"><input id="navRank" type="number" min="1" step="1" placeholder="如 20" aria-label="目标名次"><button type="submit">跳转</button></form><div class="nav-message" id="navMessage" aria-live="polite"></div></div>
      <div><label>本阶段排名 · ${rows.length} 个可跳转位置</label><div class="nav-ranks">${rankButtons || '<span class="inline-caption">没有已记录的排名</span>'}</div></div>
    </div></aside>`;
  }

  function contentHtml(c, stage) {
    const rows = candidatesOf(c, stage);
    const sorted = [...rows].sort((a, b) => (a.stage_rank ?? Infinity) - (b.stage_rank ?? Infinity));
    const positives = c.gold_evidence || [];
    const negatives = c.hard_negatives || [];
    return `<section class="report-stage-content" aria-label="当前阶段详情"><h3>${esc(c.id)} · ${esc(c.question)}</h3>
      <div class="stage-summary"><b>${esc(STAGES[stage] || stage)}</b> · ${rows.length} 条候选 · <span class="inline-caption">${esc(c.expected_behavior || '')} / ${esc(c.group_id || '')}</span><br>
      金标 ${c.stage_summaries?.[stage]?.gold_found ?? '—'}/${c.stage_summaries?.[stage]?.gold_total ?? '—'} · 首条金标 ${rank(c.stage_summaries?.[stage]?.first_gold_rank)} · 首个负例 ${rank(c.stage_summaries?.[stage]?.hard_negative_first_rank)}
      <div class="inline-caption">点击下方证据名次或卡片中的排名轨迹，可以直接切换阶段并定位。</div></div>
      <div class="detail-grid"><div><h3>金标证据</h3>${positives.map(evidenceHtml).join('') || '<div class="inline-caption">本题没有正向金标</div>'}</div><div><h3>指定负例</h3>${negatives.map(evidenceHtml).join('') || '<div class="inline-caption">没有指定负例</div>'}</div></div>
      <div class="stage-actions"><button type="button" data-action="expand-all">展开本阶段全部正文</button><button type="button" data-action="collapse-all">收起本阶段全部正文</button></div>
      <h3>${esc(STAGES[stage] || stage)}排名 · ${sorted.length} 条</h3>${sorted.length ? sorted.map(x => candidateHtml(x, stage, c.id)).join('') : '<div class="stage-empty">本阶段没有记录候选。</div>'}
      <details><summary>约束、缺失条件和生成评测参考</summary><pre>${esc(JSON.stringify({constraints:c.constraints,missing_fields:c.missing_fields,required_facts:c.required_facts,forbidden_claims:c.forbidden_claims},null,2))}</pre></details>
    </section>`;
  }

  function render() {
    const c = current();
    if (!c) { $('caseDetails').innerHTML = '<div class="stage-empty">没有可显示的题目。</div>'; return; }
    document.documentElement.style.setProperty('--report-header-height', `${Math.ceil(document.querySelector('.top').getBoundingClientRect().height)}px`);
    const stage = stageOf();
    $('caseDetails').innerHTML = `<div class="report-viewer ${navHidden ? 'nav-hidden' : ''}">${navHtml(c, stage)}${contentHtml(c, stage)}</div>`;
    const root = $('caseDetails');
    root.onclick = handleClick;
    root.onchange = event => { if (event.target.id === 'navCase') openCase(event.target.value); };
    root.onsubmit = event => {
      if (event.target.id !== 'navJump') return;
      event.preventDefault();
      jumpRank(Number($('navRank').value));
    };
    if (!root.dataset.navEventsBound) {
      root.addEventListener('toggle', event => {
        const card = event.target.closest?.('.candidate[data-body-key]');
        if (card && event.target === card) bodyOpen.set(card.dataset.bodyKey, card.open);
      }, true);
      root.dataset.navEventsBound = 'true';
    }
  }

  function focusRank(n, smooth = true, expand = true) {
    const card = [...$('caseDetails').querySelectorAll('.report-stage-content .candidate[data-rank]')]
      .find(x => Number(x.dataset.rank) === n);
    if (!card) return false;
    if (expand) {
      card.open = true;
      bodyOpen.set(card.dataset.bodyKey, true);
    }
    focusedRank = n;
    lastRanks.set(key(currentCase, stageOf(), 'reading'), n);
    $('caseDetails').querySelectorAll('.nav-ranks button').forEach(x => x.classList.toggle('active', Number(x.dataset.rank) === n));
    card.classList.add('jump-focus');
    card.scrollIntoView({behavior:smooth ? 'smooth' : 'instant', block:'start'});
    setTimeout(() => card.classList.remove('jump-focus'), 1700);
    return true;
  }

  function jumpRank(n, stage = stageOf()) {
    const c = current();
    if (!c || !Number.isInteger(n) || n < 1 || !ranked(c, stage).some(x => x.stage_rank === n)) {
      const msg = $('navMessage');
      if (msg) msg.textContent = `第 ${Number.isFinite(n) ? n : '—'} 名在该阶段没有记录；请选择有颜色的排名。`;
      return;
    }
    if (stage !== stageOf()) {
      captureReadingRank();
      caseStages.set(currentCase, stage);
      render();
    }
    requestAnimationFrame(() => focusRank(n));
  }

  function chooseStage(stage) {
    if (!ORDER.includes(stage)) return;
    captureReadingRank();
    caseStages.set(currentCase, stage);
    focusedRank = null;
    render();
    const saved = lastRanks.get(key(currentCase, stage, 'reading'));
    requestAnimationFrame(() => saved ? focusRank(saved, false, false) : $('caseDetails').scrollIntoView({behavior:'smooth',block:'start'}));
  }

  function openCase(id) {
    if (!caseById(id)) return;
    captureReadingRank();
    currentCase = id;
    focusedRank = null;
    render();
    const saved = lastRanks.get(key(id, stageOf(), 'reading'));
    requestAnimationFrame(() => saved ? focusRank(saved, false, false) : $('caseDetails').scrollIntoView({behavior:'smooth',block:'start'}));
  }

  function handleClick(event) {
    const button = event.target.closest('button[data-action]');
    if (!button || !$('caseDetails').contains(button)) return;
    const action = button.dataset.action;
    if (action === 'toggle-nav') {
      navHidden = !navHidden;
      $('caseDetails').querySelector('.report-viewer').classList.toggle('nav-hidden', navHidden);
      button.textContent = navHidden ? '显示导航 ›' : '‹ 隐藏导航';
      button.setAttribute('aria-expanded', String(!navHidden));
      return;
    }
    if (action === 'stage') { chooseStage(button.dataset.stage); return; }
    if (action === 'rank' || action === 'rank-stage') { jumpRank(Number(button.dataset.rank), button.dataset.stage || stageOf()); return; }
    if (action === 'expand-all' || action === 'collapse-all') {
      const expanded = action === 'expand-all';
      $('caseDetails').querySelectorAll('.report-stage-content .candidate').forEach(card => {card.open = expanded; bodyOpen.set(card.dataset.bodyKey, expanded);});
    }
  }

  return {renderDetails: render, openCase};
})();
renderDetails = ReportNav.renderDetails;
openCase = ReportNav.openCase;
"""
