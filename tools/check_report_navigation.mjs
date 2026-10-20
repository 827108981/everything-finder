// 手工浏览器回归：启动 Chrome 远程调试后运行 `node tools/check_report_navigation.mjs`。
import fs from 'node:fs';

const port = Number(process.argv[2] || 9223);
const targets = await fetch(`http://127.0.0.1:${port}/json/list`).then(response => response.json());
const target = targets.find(item => item.type === 'page' && item.url.endsWith('/report.html'));
if (!target) throw new Error('没有找到打开 report.html 的 Chrome 页面');

const socket = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((resolve, reject) => {
  socket.addEventListener('open', resolve, {once:true});
  socket.addEventListener('error', reject, {once:true});
});

let nextId = 0;
const pending = new Map();
socket.addEventListener('message', event => {
  const message = JSON.parse(event.data);
  if (!message.id || !pending.has(message.id)) return;
  const {resolve, reject} = pending.get(message.id);
  pending.delete(message.id);
  message.error ? reject(new Error(message.error.message)) : resolve(message.result);
});
function send(method, params = {}) {
  const id = ++nextId;
  return new Promise((resolve, reject) => {
    pending.set(id, {resolve, reject});
    socket.send(JSON.stringify({id, method, params}));
  });
}
async function evaluate(expression) {
  const result = await send('Runtime.evaluate', {expression, returnByValue:true, awaitPromise:true});
  if (result.exceptionDetails) throw new Error(result.exceptionDetails.text);
  return result.result.value;
}
async function check(label, expression) {
  const value = await evaluate(expression);
  if (!value) throw new Error(`${label} 验证失败`);
  console.log(`PASS ${label}`);
}

await send('Runtime.enable');
await send('Page.enable');
await send('Emulation.setDeviceMetricsOverride', {width:1600, height:900, deviceScaleFactor:1, mobile:false});
await send('Page.reload', {ignoreCache:true});
for (let attempt = 0; attempt < 30; attempt++) {
  if (await evaluate(`Boolean(document.querySelector('.report-viewer .report-nav'))`)) break;
  await new Promise(resolve => setTimeout(resolve, 100));
}
await check('详情导航初始渲染', `Boolean(document.querySelector('.report-viewer .report-nav'))`);
await evaluate(`ReportNav.openCase('rag-biz-006')`);
await check('切换题目', `document.querySelector('#navCase').value === 'rag-biz-006'`);
await evaluate(`document.querySelector('[data-action="stage"][data-stage="lexical"]').click()`);
await check('切换关键词阶段', `Boolean(document.querySelector('.nav-stage-list [data-stage="lexical"].active'))`);
await evaluate(`document.querySelector('#navRank').value = '20'; document.querySelector('#navJump').requestSubmit()`);
await new Promise(resolve => setTimeout(resolve, 200));
await check('直达关键词第 20 名', `Boolean(document.querySelector('.report-stage-content .candidate[data-rank="20"].jump-focus'))`);
await check('目标正文自动展开', `document.querySelector('.report-stage-content .candidate[data-rank="20"]').open`);
await check('固定导航可见', `getComputedStyle(document.querySelector('.report-nav')).position === 'sticky'`);

const vectorLink = await evaluate(`(() => {const x=document.querySelector('.evidence button[data-stage="vector"][data-rank="22"]'); if(x) x.click(); return Boolean(x)})()`);
if (!vectorLink) throw new Error('没有找到金标的向量第 22 名链接');
await new Promise(resolve => setTimeout(resolve, 200));
await check('金标跨阶段跳转', `Boolean(document.querySelector('.nav-stage-list [data-stage="vector"].active')) && Boolean(document.querySelector('.candidate[data-rank="22"].jump-focus'))`);
await evaluate(`document.querySelector('.candidate[data-rank="22"]').scrollIntoView({behavior:'instant',block:'start'})`);
await check('导航未被顶部栏遮住', `(() => {const header=document.querySelector('.top').getBoundingClientRect(); const nav=document.querySelector('.report-nav').getBoundingClientRect(); return nav.top >= header.bottom && nav.bottom <= innerHeight + 1})()`);

await evaluate(`document.querySelector('[data-action="toggle-nav"]').click()`);
await check('导航可隐藏', `document.querySelector('.report-viewer').classList.contains('nav-hidden')`);
await evaluate(`document.querySelector('[data-action="toggle-nav"]').click()`);
await check('导航可再次显示', `!document.querySelector('.report-viewer').classList.contains('nav-hidden')`);

await evaluate(`document.querySelector('[data-action="collapse-all"]').click()`);
await check('正文可由用户收起', `[...document.querySelectorAll('.report-stage-content .candidate')].every(x => !x.open)`);
await evaluate(`document.querySelector('[data-action="expand-all"]').click()`);
await check('正文可由用户展开', `[...document.querySelectorAll('.report-stage-content .candidate')].every(x => x.open)`);

await evaluate(`document.querySelector('.candidate[data-rank="22"]').open = false; document.querySelector('[data-action="stage"][data-stage="lexical"]').click()`);
await check('切回关键词阶段并记住名次', `Boolean(document.querySelector('.nav-stage-list [data-stage="lexical"].active')) && document.querySelector('.candidate[data-rank="20"]').open`);
await evaluate(`document.querySelector('[data-action="stage"][data-stage="vector"]').click()`);
await check('返回阶段后保持手动收起状态', `Boolean(document.querySelector('.nav-stage-list [data-stage="vector"].active')) && !document.querySelector('.candidate[data-rank="22"]').open`);

const screenshot = await send('Page.captureScreenshot', {format:'png', captureBeyondViewport:false});
const output = process.argv[3];
if (output) {
  fs.writeFileSync(output, Buffer.from(screenshot.data, 'base64'));
  console.log(`SCREENSHOT ${output}`);
}
socket.close();
