const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('C:/Users/50263194/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

(async () => {
  const out = path.resolve('reports/rag_vector_stage_comparison_20261116');
  const browser = await chromium.launch({headless: true, executablePath: 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
  const page = await browser.newPage({viewport: {width: 1500, height: 1000}});
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  await page.goto(process.argv[2], {waitUntil: 'domcontentloaded', timeout: 120000});
  await page.waitForSelector('#navCase', {timeout: 120000});
  if (await page.locator('#caseTable tbody tr').count() !== 91) throw Error('Expected 91 cases');
  const arms = ['baseline', 'exact', 'deeper', 'no_prefix', 'top2'];
  for (const arm of arms) {
    await page.locator('#armSelect').selectOption(arm);
    await page.waitForFunction(a => document.getElementById('runLine').textContent.includes('vector_stage_' + a), arm);
    if (await page.locator('#caseTable tbody tr').count() !== 91) throw Error('Arm lost cases: ' + arm);
  }
  await page.locator('#armSelect').selectOption('baseline');
  await page.waitForFunction(() => document.getElementById('runLine').textContent.includes('vector_stage_baseline'));
  await page.screenshot({path: path.join(out, 'report_overview.png')});
  const sources = JSON.parse(fs.readFileSync(path.join(out, 'report_sources.json'), 'utf8'));
  for (const [key, s] of Object.entries(sources)) {
    if (!fs.existsSync(s.archive || s.path)) throw Error('Missing source ' + key);
    if (/\.(pdf|pptx?)$/i.test(s.path) && !(s.location.page_number > 0 || s.location.slide_number > 0)) throw Error('Missing page ' + key);
  }
  let requested;
  await page.route('**/open-source', async route => {
    requested = route.request().postDataJSON();
    if (!sources[requested.id]) throw Error('Unresolved source button');
    await route.fulfill({status: 200, contentType: 'application/json', body: JSON.stringify({message: 'UI test: original source page resolved'})});
  });
  await page.locator('#localControls > summary').click();
  await page.locator('#localControls > details > summary').first().click();
  await page.locator('#localControls [data-original-source]:visible').first().click();
  await page.waitForFunction(() => document.getElementById('sourceStatus').textContent.startsWith('UI test'));
  await page.locator('#navCase').selectOption('evidence-flow-independent-06');
  await page.locator('[data-action="stage"][data-stage="vector"]').first().click();
  const detail = await page.locator('#caseDetails').innerText();
  if (!detail.includes('BS-2800M')) throw Error('Wrong case detail');
  await page.locator('#caseDetails').scrollIntoViewIfNeeded();
  await page.screenshot({path: path.join(out, 'report_case.png')});
  await page.locator('#search').fill('evidence-flow-independent-06');
  if (await page.locator('#caseTable tbody tr').count() !== 1) throw Error('Search filter failed');
  if (errors.length) throw Error(JSON.stringify(errors));
  const result = {cases: 91, arms, stage_navigation: true, source_manifest_checked: Object.keys(sources).length,
    source_request_resolves: !!requested, page_errors: errors,
    note: 'Headless Edge; native source launch mocked; physical source paths and 1-based page fields checked. This UI test does not launch PowerPoint/PDF readers.'};
  fs.writeFileSync(path.join(out, 'report_ui_audit.json'), JSON.stringify(result, null, 2));
  console.log(JSON.stringify(result));
  await browser.close();
})().catch(e => {console.error(e); process.exit(1);});
