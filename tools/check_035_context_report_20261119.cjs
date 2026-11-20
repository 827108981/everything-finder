const fs=require('node:fs');
const path=require('node:path');
const crypto=require('node:crypto');
const {chromium}=require('C:/Users/50263194/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
(async()=>{
 const out=path.resolve('reports/rag_035_context_ablation_20261119');
 const rows=fs.readFileSync(path.join(out,'answers.jsonl'),'utf8').trim().split('\n').map(JSON.parse);
 const sources=JSON.parse(fs.readFileSync(path.join(out,'report_sources.json'),'utf8'));
 const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
 try{
  const page=await browser.newPage({viewport:{width:1720,height:1100}});const errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto(process.argv[2],{waitUntil:'domcontentloaded'});
  if(await page.locator('#summary tbody tr').count()!==4)throw Error('Wrong arm count');
  if(await page.locator('.answer').count()!==20)throw Error('Missing answers');
  for(const r of rows){
   const shown=await page.locator(`td[data-arm="${r.arm}"][data-round="${r.round}"] .answer`).innerText();
   if(shown!==r.answer)throw Error('Answer mismatch '+r.arm+r.round);
  }
  const ids=await page.locator('[data-original-source]').evaluateAll(bs=>bs.map(b=>b.dataset.originalSource));
  if(!ids.length||ids.some(id=>!sources[id]))throw Error('Unregistered source');
  const source=page.locator('td[data-arm="A"][data-round="1"] details').filter({has:page.locator('[data-original-source]')});
  await source.locator('summary').click();
  let clicked;
  await page.route('**/open-source',async route=>{
   clicked=route.request().postDataJSON();
   if(!sources[clicked.id])throw Error('Wrong source request');
   await route.fulfill({status:200,contentType:'application/json',body:JSON.stringify({message:'source position verified'})});
  });
  await source.locator('[data-original-source]').first().click();
  await page.waitForFunction(()=>document.querySelector('#sourceStatus').textContent.includes('source position verified'));
  await page.evaluate(()=>scrollTo(0,0));
  await page.screenshot({path:path.join(out,'report_overview.png')});
  if(errors.length)throw Error(JSON.stringify(errors));
  const audit={arms:4,answer_texts_verified:20,registered_source_buttons:ids.length,source_request_valid:!!clicked,desktop_opener:'mocked; no native application opened',page_errors:errors,report_sha256:crypto.createHash('sha256').update(fs.readFileSync(path.join(out,'report.html'))).digest('hex')};
  fs.writeFileSync(path.join(out,'ui_audit.json'),JSON.stringify(audit,null,2));console.log(JSON.stringify(audit));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
