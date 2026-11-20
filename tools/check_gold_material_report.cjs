const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('C:/Users/50263194/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

(async () => {
  const out = path.resolve('reports/rag_gold_material_repair_20261116');
  const browser = await chromium.launch({ headless: true, executablePath: 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe' });
  const page = await browser.newPage({ viewport: { width: 1500, height: 1000 } });
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  await page.goto(process.argv[2], { waitUntil: 'domcontentloaded', timeout: 120000 });
  await page.waitForSelector('#navCase', { timeout: 120000 });
  const count = await page.locator('#caseTable tbody tr').count();
  if (count !== 3) throw Error('Expected 3 freshly rerun cases, got ' + count);
  const body = await page.locator('body').innerText();
  for (const expected of ['实际Context', '491', '924', '不是召回算法改善', '6题标注修订']) {
    if (!body.includes(expected)) throw Error('Missing report section: ' + expected);
  }
  await page.screenshot({ path: path.join(out, 'report_overview.png') });
  const sources = JSON.parse(fs.readFileSync(path.join(out, 'report_sources.json'), 'utf8'));
  const keys = await page.locator('[data-original-source]').evaluateAll(nodes => nodes.map(n => n.dataset.originalSource));
  for (const key of keys) {
    const s = sources[key];
    if (!s || !fs.existsSync(s.path)) throw Error('Missing physical source ' + key);
    if (/\.(pdf|pptx?)$/i.test(s.path) && !(s.location.page_number > 0 || s.location.slide_number > 0)) throw Error('Missing page ' + key);
  }
  let requested;
  await page.route('**/open-source', async route => {
    requested = route.request().postDataJSON();
    if (!sources[requested.id]) throw Error('Unresolved source button');
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ message: 'UI test: original source page resolved' }) });
  });
  await page.locator('#materialAudit > details > summary').first().click();
  await page.locator('#materialAudit [data-original-source]:visible').first().click();
  await page.waitForFunction(() => document.getElementById('sourceStatus').textContent.startsWith('UI test'));
  await page.locator('#navCase').selectOption('evidence-flow-independent-06');
  await page.locator('[data-action="stage"][data-stage="context"]').first().click();
  if (!(await page.locator('#caseDetails').innerText()).includes('BS-2800M')) throw Error('Missing actual case');
  await page.locator('#caseDetails').scrollIntoViewIfNeeded();
  await page.screenshot({ path: path.join(out, 'report_case.png') });
  await page.locator('#search').fill('evidence-flow-independent-06');
  if (await page.locator('#caseTable tbody tr').count() !== 1) throw Error('Search filter failed');
  if (errors.length) throw Error(JSON.stringify(errors));
  const result = { cases: count, annotation_revisions: 6, before_after_context: true, stage_navigation: true,
    source_buttons_checked: keys.length, source_request_resolves: !!requested, page_errors: errors,
    note: 'Headless Edge; native source launch mocked, all physical paths and page fields checked; source server separately unit tested.' };
  fs.writeFileSync(path.join(out, 'report_ui_audit.json'), JSON.stringify(result, null, 2));
  console.log(JSON.stringify(result));
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
