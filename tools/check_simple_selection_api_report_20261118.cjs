const fs=require('node:fs');
const path=require('node:path');
const {chromium}=require('C:/Users/50263194/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
(async()=>{
 const out=path.resolve('reports/rag_simple_selection_20261118');
 const rows=fs.readFileSync(path.join(out,'api_answers.jsonl'),'utf8').trim().split('\n').map(JSON.parse);
 const byId=Object.fromEntries(rows.map(r=>[r.id,r]));
 const manifest=JSON.parse(fs.readFileSync(path.join(out,'report_sources.json'),'utf8'));
 const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
 try {
  const page=await browser.newPage({viewport:{width:1650,height:1000}});const errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto(process.argv[2],{waitUntil:'domcontentloaded',timeout:120000});
  await page.waitForSelector('#navCase',{timeout:120000});
  if(await page.locator('#caseTable tbody tr').count()!==111)throw Error('Missing cases');
  if(!(await page.locator('#apiAnswerSummary').innerText()).includes('一致 40'))throw Error('Missing reviewed results');
  for(const id of ['rag-biz-019','rag-biz-021','rag-biz-050','rag-row-holdout2-001','vector-source-independent-003']){
   await page.locator('#navCase').selectOption(id);
   const displayed=await page.locator('#caseDetails .answer-panel').first().innerText();
   if(!displayed.includes(byId[id].draft_answer))throw Error('Wrong native answer '+id);
   if(!(await page.locator('#caseDetails').innerText()).includes('独立 LLM 裁判'))throw Error('Missing native judgement '+id);
   await page.locator('#compareCase').selectOption(id);
   if(await page.locator('#apiAnswerPanel .api-answer').innerText()!==byId[id].draft_answer)throw Error('Wrong comparison answer '+id);
  }
  await page.locator('#armSelect').selectOption('A');
  await page.waitForFunction(()=>document.querySelector('#runLine').textContent.includes('vector_all_stages_A'));
  if(!(await page.locator('#caseDetails').innerText()).includes('本题尚未执行生成'))throw Error('Copied answer to unrun A');
  await page.locator('#armSelect').selectOption('simple');
  await page.waitForFunction(()=>document.querySelector('#runLine').textContent.includes('vector_all_stages_simple'));
  await page.locator('#compareCase').selectOption('rag-biz-019');
  await page.locator('#apiAnswerPanel summary').filter({hasText:'答案引用的原资料'}).click();
  let sourceRequest;
  await page.route('**/open-source',async route=>{
   sourceRequest=route.request().postDataJSON();
   if(!manifest[sourceRequest.id])throw Error('Unregistered API citation source');
   await route.fulfill({status:200,contentType:'application/json',body:JSON.stringify({message:'API UI source verified'})});
  });
  await page.locator('#apiAnswerPanel [data-original-source]').first().click();
  await page.waitForFunction(()=>document.querySelector('#sourceStatus').textContent.includes('API UI source verified'));
  await page.locator('#navCase').selectOption('rag-biz-019');
  await page.screenshot({path:path.join(out,'api_report_detail.png'),fullPage:false});
  await page.evaluate(()=>window.scrollTo(0,0));
  await page.screenshot({path:path.join(out,'api_report_overview.png'),fullPage:false});
  if(errors.length)throw Error(JSON.stringify(errors));
  const audit={cases:111,native_answers_and_judgements_checked:5,api_summary_reviewed_40_of_50:true,unrun_A_still_unrun:true,original_source_link_resolves:!!sourceRequest,desktop_opener:'mocked',page_errors:errors};
  fs.writeFileSync(path.join(out,'api_ui_audit.json'),JSON.stringify(audit,null,2));console.log(JSON.stringify(audit));
 } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
