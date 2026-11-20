const fs=require('node:fs');
const path=require('node:path');
const crypto=require('node:crypto');
const {chromium}=require('C:/Users/50263194/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
(async()=>{
 const out=path.resolve('reports/rag_unified_prompt_20261119');
 const rows=fs.readFileSync(path.join(out,'report_answers.jsonl'),'utf8').trim().split('\n').map(JSON.parse);
 const manifest=JSON.parse(fs.readFileSync(path.join(out,'report_sources.json'),'utf8'));
 const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
 try{
  const page=await browser.newPage({viewport:{width:1700,height:1100}}), errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto(process.argv[2],{waitUntil:'domcontentloaded',timeout:120000});
  await page.waitForSelector('#navCase',{timeout:120000});
  if(await page.locator('#caseTable tbody tr').count()!==111)throw Error('Missing cases');
  if(!(await page.locator('#experimentSummary').innerText()).includes(`${rows.length}/250`))throw Error('Wrong task denominator');
  if(!(await page.locator('#experimentSummary').evaluate(e=>e.scrollWidth<=e.clientWidth+2)))throw Error('Summary table overflows');
  if(await page.locator('#supplementalAudit tbody tr').count()!==14)throw Error('Missing real entry comparisons');
  const checked=[];
  for(const id of ['rag-biz-014','rag-biz-016','rag-biz-021','rag-biz-035','rag-row-holdout2-001']){
   const available=rows.filter(r=>r.id===id&&r.round===1);
   if(!available.length)continue;
   await page.locator('#promptCase').selectOption(id);
   for(const r of available){
    const actual=await page.locator(`#promptCompare [data-arm="${r.arm}"] .answer-text`).first().innerText();
    if(actual!==(r.answer||r.error||''))throw Error('Wrong displayed answer '+id+' '+r.arm);
    checked.push(id+':'+r.arm);
   }
  }
  for(const arm of ['baseline','unified']){
   await page.locator('#promptArm').selectOption(arm);
   if(!(await page.locator('#runLine').innerText()).includes('unified_prompt_'+arm))throw Error('Wrong native arm');
   const r=rows.find(r=>r.arm===arm&&r.round===1&&r.answer);
   if(!r)continue;
   await page.locator('#navCase').selectOption(r.id);
   if(!(await page.locator('#caseDetails').innerText()).includes(r.answer))throw Error('Native answer missing '+r.id);
  }
  const r=rows.find(r=>r.round===1&&r.citation_check?.cited_ids?.length);
  await page.locator('#promptCase').selectOption(r.id);
  const panel=page.locator(`#promptCompare [data-arm="${r.arm}"]`);
  await panel.locator('summary').filter({hasText:'答案引用的原资料'}).first().click();
  let sourceRequest;
  await page.route('**/open-source',async route=>{
   sourceRequest=route.request().postDataJSON();
   if(!manifest[sourceRequest.id])throw Error('Unregistered source');
   await route.fulfill({status:200,contentType:'application/json',body:JSON.stringify({message:'source position verified'})});
  });
  await panel.locator('[data-original-source]').first().click();
  await page.waitForFunction(()=>document.querySelector('#sourceStatus').textContent.includes('source position verified'));
  await page.evaluate(()=>scrollTo(0,0));
  await page.screenshot({path:path.join(out,'report_overview.png')});
  if(errors.length)throw Error(JSON.stringify(errors));
  const audit={cases:111,tasks:rows.length,real_entry_rows:14,answers_checked:checked,both_native_arms_checked:true,original_source_manifest_valid:!!sourceRequest,desktop_opener:'mocked',page_errors:errors,report_sha256:crypto.createHash('sha256').update(fs.readFileSync(path.join(out,'report.html'))).digest('hex')};
  fs.writeFileSync(path.join(out,'ui_audit.json'),JSON.stringify(audit,null,2));
  console.log(JSON.stringify(audit));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
