const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('C:/Users/50263194/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

(async () => {
  const out = path.resolve(process.argv[3] || 'reports/rag_selection_factorial_20261113');
  const browser = await chromium.launch({ headless: true, executablePath: 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe' });
  const page = await browser.newPage({ viewport: { width: 1500, height: 1000 } });
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  await page.goto(process.argv[2], { waitUntil: 'domcontentloaded', timeout: 120000 });
  await page.waitForSelector('#navCase', { timeout: 120000 });
  const count = await page.locator('#caseTable tbody tr').count();
  if (count !== 91) throw Error(`Expected 91 cases, got ${count}`);
  await page.screenshot({ path: path.join(out, 'report_overview.png') });
  const verified = [];
  const arms = JSON.parse(fs.readFileSync(path.join(out, 'report_format_audit.json'), 'utf8')).arms;
  for (const arm of arms) {
    await page.locator('#armSelect').selectOption(arm);
    await page.waitForFunction(a => document.getElementById('runLine').textContent.includes('selection_factorial_' + a), arm);
    await page.locator('#navCase').selectOption('rag-biz-023');
    await page.locator('[data-action="stage"][data-stage="context"]').first().click();
    const text = await page.locator('#caseDetails').innerText();
    if (!text.includes('实际 Context 前后对照') || !text.includes('本轮未运行（不调用API）')) throw Error('Missing actual-Context/no-API panels');
    verified.push(arm);
  }
  const focusArm = arms.includes('combined') ? 'combined' : arms.includes('AC') ? 'AC' : arms.includes('R') ? 'R' : arms.at(-1);
  await page.locator('#armSelect').selectOption(focusArm);
  await page.waitForFunction(a => document.getElementById('runLine').textContent.includes('selection_factorial_' + a), focusArm);
  await page.locator('#navCase').selectOption('rag-biz-022');
  await page.locator('#caseDetails').scrollIntoViewIfNeeded();
  await page.screenshot({ path: path.join(out, 'report_case.png') });
  const sources = JSON.parse(fs.readFileSync(path.join(out, 'report_sources.json'), 'utf8'));
  let requested;
  await page.route('**/open-source', async route => {
    requested = route.request().postDataJSON();
    if (!sources[requested.id]) throw Error('Unresolved source button');
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ message: 'UI test: original source and page resolved' }) });
  });
  await page.locator('#caseDetails .detail-grid details > summary').first().click();
  await page.locator('[data-original-source]:visible').first().click();
  await page.waitForFunction(() => document.getElementById('sourceStatus').textContent.startsWith('UI test'));
  await page.locator('#search').fill('rag-biz-021');
  if (await page.locator('#caseTable tbody tr').count() !== 1) throw Error('Search filter failed');
  let pendingPagesChecked = 0;
  if (await page.locator('#freshPending').count()) {
    const keys = await page.locator('#freshPending [data-original-source]').evaluateAll(nodes => nodes.map(n => n.dataset.originalSource));
    for (const key of keys) {
      const source = sources[key];
      const paged = source && /\.(pdf|pptx?|ppsx?)$/i.test(source.path);
      if (!source || (paged && !(source.location.page_number > 0 || source.location.slide_number > 0)) || !source.location.location_text) {
        throw Error('Pending-case source button has no page or slide: ' + key);
      }
    }
    pendingPagesChecked = keys.length;
    await page.locator('#freshPending > details > summary').first().click();
    await page.locator('#freshPending').scrollIntoViewIfNeeded();
    await page.screenshot({ path: path.join(out, 'report_pending.png') });
  }
  if (errors.length) throw Error(JSON.stringify(errors));
  const result = { cases: count, verified_arms: verified, paired_actual_context: true, no_old_answers: true,
    stage_navigation: true, source_request_resolves: !!requested, pending_page_buttons_checked: pendingPagesChecked, page_errors: errors,
    note: 'Headless Edge; source launch mocked to avoid opening dozens of user windows. Native resolver separately tested.' };
  fs.writeFileSync(path.join(out, 'report_ui_audit.json'), JSON.stringify(result, null, 2));
  console.log(JSON.stringify(result));
  await browser.close();
})().catch(e => { console.error(e); process.exit(1); });
