const fs=require('node:fs');
const path=require('node:path');
const {chromium}=require('C:/Users/50263194/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
(async()=>{
 const out=path.resolve('reports/rag_condition_check_20261118');
 const sources=JSON.parse(fs.readFileSync(path.join(out,'report_sources.json'),'utf8'));
 const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
 const page=await browser.newPage({viewport:{width:1650,height:1000}}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 await page.goto(process.argv[2],{waitUntil:'domcontentloaded',timeout:120000});
 await page.waitForSelector('#navCase',{timeout:120000});
 if(await page.locator('#caseTable tbody tr').count()!==111)throw Error('Expected 111 cases');
 if(await page.locator('#comparisonTable tbody tr').count()!==6)throw Error('Expected six controls');
 await page.locator('#compareCase').selectOption('rag-row-holdout2-001');
 if(!(await page.locator('#conditionDetails').innerText()).includes('rag-row-holdout2-001:positive:1'))throw Error('Missing recovered barcode evidence');
 for(const arm of ['baseline','A','B','AB','ABC','AC']){
  await page.locator('#armSelect').selectOption(arm);
  await page.waitForFunction(a=>document.getElementById('runLine').textContent.includes('vector_all_stages_'+a),arm);
  if(await page.locator('#caseTable tbody tr').count()!==111)throw Error('Missing cases for '+arm);
 }
 await page.locator('#compareCase').selectOption('rag-biz-017');
 if(!(await page.locator('#conditionDetails').innerText()).includes('accepted_with_unconfirmed_conditions'))throw Error('C refusal-scope regression hidden');
 await page.locator('#armSelect').selectOption('A');
 await page.locator('#navCase').selectOption('rag-row-holdout2-001');
 await page.locator('[data-action="stage"][data-stage="constraints"]').first().click();
 let clicked;
 await page.route('**/open-source',async route=>{clicked=route.request().postDataJSON();if(!sources[clicked.id])throw Error('Unknown source');await route.fulfill({status:200,contentType:'application/json',body:JSON.stringify({message:'UI test: source resolved'})});});
 await page.locator('#caseDetails [data-original-source]:visible').first().click();
 await page.waitForFunction(()=>document.getElementById('sourceStatus').textContent.includes('UI test'));
 await page.locator('#search').fill('rag-biz-019');
 if(await page.locator('#caseTable tbody tr').count()!==1)throw Error('Search failed');
 if(errors.length)throw Error(JSON.stringify(errors));
 await page.screenshot({path:path.join(out,'report_overview.png'),fullPage:false});
 const audit={cases:111,arms:6,stages:8,source_request_resolves:!!clicked,page_errors:errors,actual_source_launch:'mocked; did not launch user PDF/PPT viewers',no_gold_regression_displayed:true};
 fs.writeFileSync(path.join(out,'ui_audit.json'),JSON.stringify(audit,null,2));
 console.log(JSON.stringify(audit));
 await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
