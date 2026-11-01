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
.report-stage-content .query-panel{padding:12px 14px;margin:10px 0 16px;border:1px solid #b9d0fb;background:#f4f8ff;border-radius:10px}
.report-stage-content .query-panel h4{margin:0 0 8px}
.report-stage-content .query-panel code{white-space:pre-wrap;overflow-wrap:anywhere}
.report-stage-content .query-chip{display:inline-block;padding:3px 8px;margin:3px;border-radius:7px;background:#fff;border:1px solid #b9d0fb}
.report-stage-content .answer-panel{padding:14px 16px;margin:10px 0 16px;border:1px solid #9fd3b3;background:#f0faf4;border-radius:10px}
.report-stage-content .answer-panel.failed{border-color:#ef9a9a;background:#fff3f3}
.report-stage-content .answer-panel h4{margin:0 0 8px}
.report-stage-content .answer-panel pre{background:#fff;margin:8px 0;max-height:340px;overflow:auto}
.report-stage-content .answer-reason{color:#9b1c1c;font-weight:700;overflow-wrap:anywhere}
.report-stage-content .answer-panel.judge-incorrect{border-color:#ef9a9a;background:#fff3f3}
.report-stage-content .answer-panel.judge-incomplete,.report-stage-content .answer-panel.judge-needs_clarification,.report-stage-content .answer-panel.judge-behavior_mismatch{border-color:#e8d575;background:#fffbea}
.report-stage-content .answer-panel.judge-safety_gap{border-color:#f0aa60;background:#fff4e8}
.report-stage-content .judge-rubric{border-top:1px solid #dce5e4;padding:8px 0}
.report-stage-content .judge-rubric.context{color:#667085}
.report-stage-content .judge-rubric small{display:block;color:#667085}
.report-stage-content .exit-note{margin-top:5px;color:#9b3b19;font-size:12px;font-weight:700}
.report-stage-content .group-sources{margin:8px 0;padding:7px;background:#f4f7fb;border-radius:7px;font-size:12px}
.report-stage-content .vector-unit{margin:8px 0;padding:9px;border-left:3px solid #6b7fd7;background:#f7f8ff;border-radius:6px}
.report-stage-content .vector-unit pre{margin:6px 0 0;background:#fff}
.report-stage-content .stage-funnel{display:grid;grid-template-columns:repeat(8,minmax(0,1fr));gap:5px;margin:12px 0}
.report-stage-content .stage-funnel button{padding:7px 2px;font-size:11px;min-width:0;white-space:normal}
.report-stage-content .stage-funnel b{display:block;font-size:15px}
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
  function groupedRows(c, stage) {
    const groups = new Map();
    for (const row of candidatesOf(c, stage)) {
      const id = row.content_group_id || row.chunk_id;
      if (!groups.has(id)) groups.set(id, []);
      groups.get(id).push(row);
    }
    return [...groups.values()].sort((a, b) => (a[0].stage_rank ?? Infinity) - (b[0].stage_rank ?? Infinity));
  }
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
    if (focusedRank != null && cards.some(card => (card.dataset.memberRanks || '').split(',').map(Number).includes(focusedRank))) {
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
    const exit = e.exit || {};
    const exitLabel = exit.from ? `${STAGES[exit.from] || exit.from} → ${STAGES[exit.to] || exit.to}：${exit.reason || ''}` : exit.reason || '';
    const suspected = ORDER.map(stage => (e.suspected_stages?.[stage] || []).map(n => `<button type="button" data-action="rank-stage" data-stage="${esc(stage)}" data-rank="${n}" title="疑似来源，不计入正式指标">${esc(STAGES[stage] || stage)} #${n}</button>`).join('、')).filter(Boolean).join(' → ');
    const mapping = e.mapping || {};
    const mappingNote = isGold ? `指定来源 ${mapping.declared_chunk_ids?.length || 0} 块 · 自动等价 ${mapping.equivalent_chunk_ids?.length || 0} 块 · 疑似 ${mapping.suspected_chunk_ids?.length || 0} 块` : '';
    return `<div class="evidence ${isGold ? 'ok' : 'neg'} ${e.first_rank == null ? 'miss' : ''}"><b>${esc(e.evidence_id)}</b> · ${esc(e.file || '')} · 第 ${esc(e.page ?? '?')} 页<br>${esc(e.quote || '')}${!isGold && e.reason ? `<br><span class="inline-caption">负例原因：${esc(e.reason)}</span>` : ''}${mappingNote ? `<br><span class="inline-caption">${esc(mappingNote)}</span>` : ''}<br><span class="inline-caption">各阶段完整证据名次：${ranks}</span>${suspected ? `<div class="inline-caption">疑似金标（不计分）：${suspected}</div>` : ''}${exitLabel ? `<div class="exit-note">去向：${esc(exitLabel)}</div>` : ''}</div>`;
  }

  function candidateHtml(c, stage, caseId, members = [c]) {
    const n = c.stage_rank;
    const bodyKey = key(caseId, stage, n == null ? `rejected:${c.chunk_id}` : n);
    const isOpen = bodyOpen.get(bodyKey) !== false;
    const labels = {gold:'金标证据',negative:'指定负例',mixed:'同块混合',unlabelled:'相似未标注',partial_gold:'部分金标',partial_negative:'部分负例'};
    const categories = new Set(members.map(row => row.category || 'unlabelled'));
    const category = categories.has('mixed') || (categories.has('gold') && categories.has('negative'))
      ? 'mixed' : categories.has('gold') ? 'gold' : categories.has('negative') ? 'negative' : c.category || 'unlabelled';
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
    const scores = `关键词 ${textOrDash(c.lexical_score)}${c.lexical_score_kind ? '（' + c.lexical_score_kind + '）' : ''} · 向量 ${textOrDash(c.vector_score)} · 融合 ${textOrDash(c.fused_score)} · 重排 ${textOrDash(c.rerank_score)} · 约束 ${textOrDash(c.constraint_score)}`;
    const lexicalDetail = (c.lexical_contributions || []).map(x => `<tr><td>${esc(x.term)}</td><td>${esc(x.document_frequency)}</td><td>${esc(x.body_frequency)}</td><td>${esc(x.title_frequency)}</td><td>${esc(Number(x.body_score || 0).toFixed(4))}</td><td>${esc(Number(x.title_score || 0).toFixed(4))}</td><td>${esc((x.match_methods || []).join('、'))}</td></tr>`).join('');
    const vectorUnits = (c.vector_unit_hits || []).map(x => `<div class="vector-unit"><b>原始单元 #${esc(x.raw_rank)}</b> · ${esc(x.kind || '')} · 分数 ${esc(x.score)} · ${esc(x.token_count)} tokens<br><span class="inline-caption">${esc(x.unit_id || '')}</span><pre>${esc(x.text || '')}</pre></div>`).join('');
    const location = c.location || (c.page_number != null ? `第 ${c.page_number} 页` : '位置未记录');
    const memberRanks = members.map(row => row.stage_rank).filter(Number.isInteger);
    const sourceRows = members.flatMap(row => row.sources?.length ? row.sources.map(source => ({...source, rank: row.stage_rank, text: source.chunk_id === row.chunk_id ? row.text : null}))
      : [{filename:row.filename, location:row.location, rank:row.stage_rank, text:row.text}]);
    const sourceHtml = `<details class="group-sources"><summary>同一正文的原始来源 ${sourceRows.length} 处 · 原始名次 ${memberRanks.map(x => '#' + x).join('、') || '未入选'}</summary>${sourceRows.map(source => `<div><b>${esc(source.filename || '')}</b> · ${esc(source.location || '')} · ${source.rank == null ? '未入选' : '#' + source.rank}${source.text == null ? ' · 原始块可在关键词/向量/融合阶段按 ID 追溯' : `<details><summary>原始正文</summary><pre>${esc(source.text)}</pre></details>`}</div>`).join('')}</details>`;
    return `<details class="candidate ${esc(category)}" data-rank="${n == null ? '' : n}" data-member-ranks="${esc(memberRanks.join(','))}" data-body-key="${esc(bodyKey)}" ${isOpen ? 'open' : ''}>
      <summary><strong class="rank">${n == null ? '未入选' : '#' + n}</strong><b>${esc(c.filename || '')}</b><span class="pill">${esc(location)}</span><span class="badge ${esc(category)}">${esc(category === 'mixed' ? '同块混合' : c.formal_gold_type || labels[category] || category)}</span><span class="score">${esc(scores)}</span></summary>
      <div class="candidate-body"><div class="inline-caption">状态：${esc(c.status || '')}；${c.formal_gold_type ? `证据判定：${esc(c.formal_gold_type)}；` : ''}原因：${esc(c.reason || c.constraint_decision?.reason || '')}${c.truncated ? '；上下文已截断' : ''}</div>
      <div class="inline-caption">同块排名轨迹：${history || '未进入其他已记录阶段'}</div>${lexicalDetail ? `<details><summary>关键词逐词得分</summary><table><thead><tr><th>词项</th><th>全库 df</th><th>正文 tf</th><th>标题 tf</th><th>正文贡献</th><th>标题原始贡献</th><th>定位方式</th></tr></thead><tbody>${lexicalDetail}</tbody></table></details>` : ''}${vectorUnits ? `<details><summary>查看命中的细粒度向量单元（父证据取最高分）</summary>${vectorUnits}</details>` : ''}${sourceHtml}<pre>${content}</pre></div></details>`;
  }

  function queryHtml(c) {
    const q = c.query || {};
    const words = q.lexical_terms || [];
    const spans = (q.spans || []).map(x => `<span class="query-chip" title="原文位置 ${esc(x.start)}–${esc(x.end)}；${esc((x.features || []).join('、'))}">${esc(x.text)} <small>${esc(x.start)}–${esc(x.end)}</small></span>`).join('');
    const changes = (q.transformations || []).map(x => `<li>${esc(x.operation)}：${esc(x.source_text)} → ${esc(x.result_text)}；${esc(x.reason)}</li>`).join('');
    const lexicalRoutes = (q.executed_lexical_routes || []).map(x => `<li><b>${esc(x.route_id)}</b> · 后端 ${esc(x.backend || 'legacy')} · 输出 ${esc(x.candidate_count)} 条 · 定位/过滤/计分 ${esc(x.located_candidate_count ?? x.scanned_count ?? '—')}/${esc(x.filtered_candidate_count ?? '—')}/${esc(x.scored_candidate_count ?? '—')} · ${esc(x.duration_ms)} ms${x.candidate_pool_exhausted ? ' · <strong class="warning">扫描到上限，候选池可能尚未穷尽</strong>' : ''}${(x.complete_match_terms || []).length ? ' · 完整词补充：' + esc(x.complete_match_terms.join('、')) : ''} · <code>${esc(x.expression || x.query_text || '')}</code></li>`).join('');
    const semanticRoutes = (q.executed_vector_routes || []).map(x => `<li><b>${esc(x.route_id)}</b> · 后端 ${esc(x.backend || 'legacy')} · 原始命中 ${esc(x.candidate_count)} 条 · ${esc(x.duration_ms)} ms · ${esc(x.query_text || '')}${x.vector_index_fingerprint ? ' · 索引 ' + esc(x.vector_index_fingerprint.slice(0, 12)) : ''}${x.status === 'degraded_route' ? '（失败后降级）' : ''}</li>`).join('');
    return `<section class="query-panel"><h4>本题实际检索关键词</h4><div>后端：${esc(q.lexical_backend || 'legacy')}；关键词：${words.length ? words.map(word => `<span class="query-chip">${esc(word)}</span>`).join('') : '未记录或未提取'}</div>
      <div class="inline-caption">实际全文查询：<code>${esc(q.lexical_expression || '未记录')}</code></div>
      <div class="inline-caption">向量检索原文：${esc(q.semantic_query || '未记录')}</div>
      <div class="inline-caption">保留的原文片段：${spans || '旧运行未记录'}</div>
      ${changes ? `<details><summary>查看问句清理过程</summary><ul>${changes}</ul></details>` : ''}
      ${lexicalRoutes ? `<details open><summary>实际执行的关键词查询路线</summary><ul>${lexicalRoutes}</ul></details>` : ''}
      ${semanticRoutes ? `<details><summary>实际执行的向量查询路线</summary><ul>${semanticRoutes}</ul></details>` : ''}
      <div class="inline-caption">阶段上限：${esc(JSON.stringify(q.limits || {}))} · 解析版本 ${esc(q.analyzer_version || '未记录')}</div></section>`;
  }

  function answerHtml(c) {
    const answer = c.answer || {};
    const generation = c.generation || {};
    const finalText = String(answer.text || '').trim();
    const draft = String(generation.draft_answer || '').trim();
    const failure = c.error?.message || '';
    const refused = /本服务不支持|无法协助|不能协助|无法提供|不能提供/.test(draft);
    if (finalText) {
      const original = draft && draft !== finalText
        ? `<details><summary>查看生成初稿</summary><pre>${esc(draft)}</pre></details>` : '';
      return `<section class="answer-panel"><h4>本题最终答案</h4><pre>${esc(finalText)}</pre>${original}</section>`;
    }
    if (failure) {
      const label = refused ? '模型／服务商返回的拒答说明' : '模型返回的原始答复（未通过校验）';
      const raw = draft ? `<h4>${label}</h4><pre>${esc(draft)}</pre>` : '<h4>本题没有可展示的模型答复</h4>';
      return `<section class="answer-panel failed">${raw}<div class="answer-reason">评测失败原因：${esc(failure)}</div>${generation.failed_stage ? `<div class="inline-caption">失败阶段：${esc(generation.failed_stage)}</div>` : ''}</section>`;
    }
    const note = DATA.run?.mode === 'retrieval-only' ? '本次只评测检索，没有调用 LLM 生成答案。' : '本题尚未执行生成，因而没有答案。';
    return `<section class="answer-panel"><h4>本题答案</h4><div>${note}</div></section>`;
  }

  function judgeHtml(c) {
    const j = c.gold_judgement || {};
    const labels = {consistent:'符合金标', incomplete:'必答内容缺失', safety_gap:'安全提示缺失', needs_clarification:'需要澄清', behavior_mismatch:'回答方式不符', incorrect:'明确错误', uncertain:'待人工复核', unavailable:'裁判未完成', inconsistent:'旧版：与金标不一致'};
    const actions = {consistent:'核心要求已覆盖；可选背景未提不扣分。', incomplete:'核对必答项是否确属本题，再补全答案。', safety_gap:'答案涉及维修操作，应补充断电与防静电提示。', needs_clarification:'先请用户确认影响唯一答案的条件。', behavior_mismatch:'核对回答、部分回答或拒答方式是否符合题目要求。', incorrect:'核对矛盾事实或被命中的禁止主张，并检查证据。', uncertain:'需人工核查裁判无法确认的内容。'};
    const roleLabels = {core:'本题必答', safety:'安全要求', context:'可选背景', forbidden:'禁止主张'};
    const statusLabels = {covered:'已表达', missing:'未表达', contradicted:'表达相反', uncertain:'无法判清', present:'错误主张出现', absent:'未出现'};
    if (!j.verdict) return '<section class="answer-panel"><h4>答案与金标</h4><div class="inline-caption">本次未运行金标答案裁判；不能据此认定答案正确。</div></section>';
    const rows = (j.rubrics || []).map(r => {const role=r.role || (r.kind === 'forbidden_claims' ? 'forbidden' : 'core'); return `<div class="judge-rubric ${esc(role)}"><b>${esc(r.id)} · ${esc(roleLabels[role] || role)} · ${esc(statusLabels[r.status] || r.status)}</b><div>${esc(r.gold_text)}</div><small>${esc(r.reason || '未提供理由')}</small></div>`}).join('');
    const comparison = j.rejudged_from_saved_answer ? `<div class="inline-caption">沿用原运行已保存的答案重新判分：${esc(labels[j.previous_verdict] || j.previous_verdict || '未运行')} → ${esc(labels[j.verdict] || j.verdict)}；检索与答案生成未重跑。</div>` : '';
    return `<section class="answer-panel judge-${esc(j.verdict)}"><h4>答案与金标：${esc(labels[j.verdict] || j.verdict)}</h4>
      <div>行为核查：${esc(j.behavior || '未判定')} · ${esc(j.behavior_reason || j.reason || '')}</div>
      <div><b>处理方向：</b>${esc(actions[j.verdict] || j.reason || '请结合逐项结果人工复核。')}</div>
      <div class="inline-caption">独立 LLM 裁判 ${esc(j.model || '')} · 判分版本 ${esc(j.rubric_version || '旧版')}；逐项理由供人工复核，不作为绝对正确性证明。</div>${comparison}${rows}</section>`;
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
    const groups = groupedRows(c, stage);
    const positives = c.gold_evidence || [];
    const negatives = c.hard_negatives || [];
    return `<section class="report-stage-content" aria-label="当前阶段详情"><h3>${esc(c.id)} · ${esc(c.question)}</h3>
      <div class="stage-summary"><b>${esc(STAGES[stage] || stage)}</b> · ${rows.length} 条候选 · <span class="inline-caption">${esc(c.expected_behavior || '')} / ${esc(c.group_id || '')}</span><br>
      金标 ${c.stage_summaries?.[stage]?.gold_found ?? '—'}/${c.stage_summaries?.[stage]?.gold_total ?? '—'} · 首条金标 ${rank(c.stage_summaries?.[stage]?.first_gold_rank)} · 首个负例 ${rank(c.stage_summaries?.[stage]?.hard_negative_first_rank)}
      <div class="inline-caption">点击下方证据名次或卡片中的排名轨迹，可以直接切换阶段并定位。</div></div>${answerHtml(c)}${judgeHtml(c)}${caseTimingHtml(c)}${queryHtml(c)}
      <div class="stage-funnel" aria-label="各阶段金标和负例数量">${ORDER.map(s => `<button type="button" data-action="stage" data-stage="${esc(s)}" title="切换到${esc(STAGES[s] || s)}"><span>${esc(STAGES[s] || s)}</span><b>${c.stage_summaries?.[s]?.gold_found ?? '—'}/${c.stage_summaries?.[s]?.gold_total ?? '—'}</b><small>负例 ${c.stage_summaries?.[s]?.hard_negative_found ?? '—'}</small></button>`).join('')}</div>
      <div class="detail-grid"><div><h3>金标证据</h3>${positives.map(evidenceHtml).join('') || '<div class="inline-caption">本题没有正向金标</div>'}</div><div><h3>指定负例</h3>${negatives.map(evidenceHtml).join('') || '<div class="inline-caption">没有指定负例</div>'}</div></div>
      ${(c.acceptable_alternative_evidence || []).length ? `<details><summary>人工记录的可接受替代证据（未自动并入 Gold 计分）</summary><pre>${esc(JSON.stringify(c.acceptable_alternative_evidence,null,2))}</pre></details>` : ''}
      <div class="stage-actions"><button type="button" data-action="expand-all">展开本阶段全部正文</button><button type="button" data-action="collapse-all">收起本阶段全部正文</button></div>
      <h3>${esc(STAGES[stage] || stage)}排名 · ${rows.length} 条 / ${groups.length} 个内容组</h3>${groups.length ? groups.map(items => candidateHtml(items[0], stage, c.id, items)).join('') : '<div class="stage-empty">本阶段没有记录候选。</div>'}
      <details><summary>约束、缺失条件和答案判分项</summary><pre>${esc(JSON.stringify({constraints:c.constraints,missing_fields:c.missing_fields,answer_checks:c.answer_checks,required_facts:c.required_facts,forbidden_claims:c.forbidden_claims},null,2))}</pre><div class="inline-caption">新版答案裁判使用 answer_checks 和 forbidden_claims；required_facts 保留供历史对照。</div></details>
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
      .find(x => (x.dataset.memberRanks || '').split(',').map(Number).includes(n));
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
