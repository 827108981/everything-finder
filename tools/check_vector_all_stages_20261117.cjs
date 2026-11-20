const {chromium} = require('C:/Users/50263194/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

(async () => {
  const browser = await chromium.launch({headless:true, executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
  const page = await browser.newPage({viewport:{width:1500,height:1000}});
  const errors=[];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(process.argv[2], {waitUntil:'domcontentloaded'});
  if (await page.locator('#case option').count() !== 111) throw Error('Expected 111 questions');
  if (await page.locator('#comparison tbody tr').count() !== 9) throw Error('Expected 9 schemes');
  if (await page.locator('#summary th').count() !== 9) throw Error('Expected 8 stages');
  await page.locator('#search').fill('rag-biz-019');
  if (await page.locator('#case option').count() !== 1) throw Error('Search failed');
  await page.locator('[data-arm="W0"][data-stage="rerank"]').click();
  const detail=await page.locator('#detailPanel').innerText();
  if (!detail.includes('金标 2') || !detail.includes('未命中')) throw Error('Missing individual evidence ranks');
  await page.locator('#cohort').selectOption('source_holdout');
  await page.locator('#metric').selectOption('mrr');
  if (!(await page.locator('#summary').innerText()).includes('W1')) throw Error('Summary not updated');
  let clicked=false;
  await page.route('**/open-source', async route => {clicked=true; await route.fulfill({status:200,contentType:'application/json',body:JSON.stringify({message:'source resolved'})});});
  await page.locator('#detailPanel [data-source]').first().click();
  await page.waitForFunction(() => document.getElementById('message').textContent === 'source resolved');
  if (!clicked || errors.length) throw Error(JSON.stringify(errors));
  console.log(JSON.stringify({cases:111,arms:9,stages:8,individual_gold:true,source_button:true,page_errors:errors}));
  await browser.close();
})().catch(error => {console.error(error); process.exit(1)});
