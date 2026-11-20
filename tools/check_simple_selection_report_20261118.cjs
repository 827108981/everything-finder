const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require('C:/Users/50263194/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
(async () => {
  const out = path.resolve('reports/rag_simple_selection_20261118');
  const sources = JSON.parse(fs.readFileSync(path.join(out, 'report_sources.json'), 'utf8'));
  const browser = await chromium.launch({headless:true, executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
  const page = await browser.newPage({viewport:{width:1650,height:1000}});
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  await page.goto(process.argv[2], {waitUntil:'domcontentloaded', timeout:120000});
  await page.waitForSelector('#navCase', {timeout:120000});
  if (await page.locator('#caseTable tbody tr').count() !== 111) throw Error('Expected 111 cases');
  if (await page.locator('#comparisonTable tbody tr').count() !== 4) throw Error('Expected four controls');
  if (await page.locator('#armSelect').inputValue() !== 'simple') throw Error('Incorrect selected arm');
  await page.locator('#compareCase').selectOption('vector-source-independent-003');
  if (!(await page.locator('#simpleDetails').innerText()).includes('vector-source-independent-003:positive:1')) throw Error('Missing restored Gold');
  for (const arm of ['baseline','A','simple_legacy_budget','simple']) {
    await page.locator('#armSelect').selectOption(arm);
    await page.waitForFunction(a => document.getElementById('runLine').textContent.includes('vector_all_stages_'+a), arm);
    if (await page.locator('#caseTable tbody tr').count() !== 111) throw Error('Missing cases '+arm);
  }
  await page.locator('#compareCase').selectOption('rag-biz-018');
  if (!(await page.locator('#simpleDetails').innerText()).includes('rag-biz-018:hard_negative:2')) throw Error('Hidden negative evidence change');
  await page.locator('#armSelect').selectOption('simple');
  await page.locator('#navCase').selectOption('rag-biz-018');
  await page.locator('[data-action="stage"][data-stage="context"]').first().click();
  let clicked;
  await page.route('**/open-source', async route => {
    clicked = route.request().postDataJSON();
    if (!sources[clicked.id]) throw Error('Unknown source ID');
    await route.fulfill({status:200,contentType:'application/json',body:JSON.stringify({message:'UI test: source resolved'})});
  });
  await page.locator('#caseDetails [data-original-source]:visible').first().click();
  await page.waitForFunction(() => document.getElementById('sourceStatus').textContent.includes('UI test'));
  await page.locator('#search').fill('rag-biz-019');
  if (await page.locator('#caseTable tbody tr').count() !== 1) throw Error('Search failed');
  if (errors.length) throw Error(JSON.stringify(errors));
  await page.screenshot({path:path.join(out,'report_detail.png'),fullPage:false});
  await page.locator('#search').fill('');
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({path:path.join(out,'report_overview.png'),fullPage:false});
  const audit = {cases:111,arms:4,stages:8,source_request_resolves:!!clicked,page_errors:errors,
    no_gold_negative_change_visible:true,initial_arm:'simple',
    actual_source_launch:'mocked to avoid opening user applications; manifest binding checked'};
  fs.writeFileSync(path.join(out,'ui_audit.json'),JSON.stringify(audit,null,2));
  console.log(JSON.stringify(audit));
  await browser.close();
})().catch(e => {console.error(e);process.exit(1)});
