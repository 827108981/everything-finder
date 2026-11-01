"""离线报告中的召回路线、Gold 退出原因和证据类型图。"""

OVERVIEW_JS = r'''
function renderDiagnosticOverview() {
  const overview = DATA.summary?.diagnostic_overview || {};
  const groups = [
    ['routeHits', 'Gold 在各查询路线出现的题数', '#2563eb'],
    ['lossReasons', 'Gold 候选退出原因', '#c62828'],
    ['evidenceTypes', '证据映射类型', '#2e7d32'],
  ];
  const bar = (name, value, max, color) => `<div style="display:grid;grid-template-columns:minmax(120px,1fr) 3fr 36px;gap:8px;align-items:center;margin:7px 0"><span>${esc(name)}</span><span style="height:13px;background:#edf1f7;border-radius:7px"><i style="display:block;height:100%;width:${Math.max(2,value/max*100)}%;border-radius:7px;background:${color}"></i></span><b>${value}</b></div>`;
  $('diagnosticOverview').innerHTML = groups.map(([key,title,color]) => {
    const entries = Object.entries(overview[key] || {}).sort((a,b) => b[1]-a[1]);
    const max = Math.max(1,...entries.map(([,n]) => Number(n)));
    return `<div class="card"><h3>${esc(title)}</h3>${entries.length ? entries.map(([name,n]) => bar(name,Number(n),max,color)).join('') : '<span class="muted">本次未记录</span>'}</div>`;
  }).join('');
}
'''
