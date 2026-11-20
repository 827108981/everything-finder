const fs=require('node:fs');
const path=require('node:path');
const {chromium}=require('C:/Users/50263194/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
(async()=>{
 const out=path.resolve('reports/rag_vector_packaging_20261116/all_stages_20261117');
 const sources=JSON.parse(fs.readFileSync(path.join(out,'report_sources.json'),'utf8'));
 const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
 const page=await browser.newPage({viewport:{width:1500,height:1000}}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 await page.goto(process.argv[2],{waitUntil:'domcontentloaded',timeout:120000});
 await page.waitForSelector('#navCase',{timeout:120000});
 if(await page.locator('#caseTable tbody tr').count()!==111)throw Error('Expected 111 cases');
 if(await page.locator('#comparisonTable tbody tr').count()!==9)throw Error('Expected 9 arms');
 await page.locator('#compareCase').selectOption('rag-biz-019');
 if(!(await page.locator('#comparisonTable').innerText()).includes('1/2'))throw Error('Missing per-case comparison');
 for(const arm of ['W0','W1','W2','W3','W4','W5','F10','F11','F10_count_matched']){
  await page.locator('#armSelect').selectOption(arm);
  await page.waitForFunction(a=>document.getElementById('runLine').textContent.includes('vector_all_stages_'+a),arm);
  if(await page.locator('#caseTable tbody tr').count()!==111)throw Error('Missing cases in '+arm);
 }
 await page.locator('#navCase').selectOption('rag-biz-019');
 await page.locator('[data-action="stage"][data-stage="rerank"]').first().click();
 let clicked;
 await page.route('**/open-source',async route=>{clicked=route.request().postDataJSON();if(!sources[clicked.id])throw Error('Unknown source');await route.fulfill({status:200,contentType:'application/json',body:JSON.stringify({message:'UI test: source resolved'})});});
 await page.locator('#caseDetails [data-original-source]:visible').first().click();
 await page.waitForFunction(()=>document.getElementById('sourceStatus').textContent.includes('UI test'));
 await page.locator('#search').fill('rag-biz-019');
 if(await page.locator('#caseTable tbody tr').count()!==1)throw Error('Search failed');
 if(errors.length)throw Error(JSON.stringify(errors));
 await page.screenshot({path:path.join(out,'native_report_overview.png'),fullPage:false});
 console.log(JSON.stringify({cases:111,arms:9,stages:8,source_request_resolves:!!clicked,page_errors:errors,actual_source_launch:'mocked'}));
 await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
