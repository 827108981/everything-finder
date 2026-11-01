"""离线报告的耗时图表资源。"""

TIMING_CSS = r"""
.timing-cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin-bottom:12px}
.timing-cards>div{padding:10px 12px;border:1px solid #dce5f2;border-radius:9px;background:#f8fbff}
.timing-cards b{display:block;font-size:19px;color:#173c76}
.timing-row{display:grid;grid-template-columns:minmax(110px,180px) minmax(90px,1fr) 85px;gap:10px;align-items:center;margin:7px 0}
.timing-track{height:15px;background:#edf2f7;border-radius:8px;overflow:hidden}
.timing-fill{height:100%;background:#4479d4;border-radius:8px}
.timing-row.important .timing-fill{background:#7b48b9}
.case-timing{margin:10px 0;padding:12px 14px;border:1px solid #dce5f2;border-radius:10px;background:#f8fbff}
.case-timing h4{margin:0 0 7px}
.case-timing .timing-row{font-size:12px;margin:4px 0}
@media(max-width:680px){.timing-row{grid-template-columns:115px 1fr 68px;gap:5px}}
"""

TIMING_JS = r"""
const TimingView = (() => {
  const ms = value => value == null ? '未记录' : value >= 60000
    ? `${(value / 60000).toFixed(1)} 分` : `${(value / 1000).toFixed(2)} 秒`;
  function bars(modules, labels, total) {
    const rows = Object.entries(modules || {}).filter(([,value]) => value > 0)
      .sort((a,b) => b[1] - a[1]);
    const max = Math.max(1, ...rows.map(([,value]) => value));
    return rows.map(([key,value]) => `<div class="timing-row ${key === 'generation' || key === 'gold_judge' ? 'important' : ''}">
      <span>${esc(labels?.[key] || key)}</span><span class="timing-track"><span class="timing-fill" style="display:block;width:${Math.max(2, value / max * 100)}%"></span></span>
      <b>${ms(value)}</b></div>`).join('') || '<div class="muted">没有阶段计时记录。</div>';
  }
  function render() {
    const target = $('timingOverview');
    if (!target) return;
    const timing = DATA.summary?.timing || {};
    const labels = timing.module_labels || {};
    target.innerHTML = `<div class="timing-cards">
      <div><small>已记录整体耗时</small><b>${ms(timing.overall_recorded_ms ?? timing.wall_ms)}</b><small>${timing.overall_recorded_ms == null ? '旧记录仅包含评测循环' : '预检准备 + 评测循环'}</small></div>
      <div><small>预检与本地模型准备</small><b>${ms(timing.preparation_ms)}</b><small>旧运行可能未记录</small></div>
      <div><small>逐题耗时合计</small><b>${ms(timing.case_total_ms)}</b><small>已运行 ${timing.measured_case_count || 0} 题</small></div>
      <div><small>题间及汇总处理</small><b>${ms(timing.outside_cases_ms)}</b><small>循环耗时减去逐题耗时</small></div>
      <div><small>LLM 生成及重试</small><b>${ms(timing.modules_ms?.generation || 0)}</b><small>含网络等待和自动重试</small></div>
    </div>${bars(timing.modules_ms, labels, timing.case_total_ms)}
    <p class="muted">${esc(timing.definition || '没有计时说明。')}</p>`;
  }
  function forCase(c) {
    const timing = c.timing || {};
    if (timing.total_ms == null) return '';
    const labels = DATA.summary?.timing?.module_labels || {};
    return `<section class="case-timing"><h4>本题耗时 ${ms(timing.total_ms)}</h4>
      ${bars(timing.modules_ms, labels, timing.total_ms)}</section>`;
  }
  return {render, forCase};
})();
function renderTiming(){TimingView.render()}
function caseTimingHtml(c){return TimingView.forCase(c)}
"""
