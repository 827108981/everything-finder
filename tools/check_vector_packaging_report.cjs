const fs=require('node:fs');
const path=require('node:path');
const {chromium}=require('C:/Users/50263194/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
(async()=>{
 const out=path.resolve('reports/rag_vector_packaging_20261116');
 const audit=JSON.parse(fs.readFileSync(path.join(out,'report_format_audit.json'),'utf8'));
 const sources=JSON.parse(fs.readFileSync(path.join(out,'report_sources.json'),'utf8'));
 const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
 const page=await browser.newPage({viewport:{width:1500,height:1000}});const errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 await page.goto(process.argv[2],{waitUntil:'domcontentloaded',timeout:120000});
 await page.waitForSelector('#navCase',{timeout:120000});
 if(await page.locator('#caseTable tbody tr').count()!==111)throw Error('Expected 111 cases including 20 pending');
 for(const arm of audit.arms){
  await page.locator('#armSelect').selectOption(arm);
  await page.waitForFunction(a=>document.getElementById('runLine').textContent.includes('vector_packaging_'+a),arm);
  if(await page.locator('#caseTable tbody tr').count()!==111)throw Error('Missing cases in '+arm);
 }
 for(const [key,s] of Object.entries(sources)){
  if(!fs.existsSync(s.archive||s.path))throw Error('Missing source '+key);
  if(/\.(pdf|pptx?)$/i.test(s.path)&&!(s.location.page_number>0||s.location.slide_number>0))throw Error('Missing page '+key);
 }
 let clicked;
 await page.route('**/open-source',async route=>{
  clicked=route.request().postDataJSON();if(!sources[clicked.id])throw Error('Unresolved source button');
  await route.fulfill({status:200,contentType:'application/json',body:JSON.stringify({message:'UI test: original source position resolved'})});
 });
 await page.locator('#holdoutSources > summary').click();
 await page.locator('#holdoutSources > details > summary').first().click();
 await page.locator('#holdoutSources [data-original-source]:visible').first().click();
 await page.waitForFunction(()=>document.getElementById('sourceStatus').textContent.startsWith('UI test'));
 await page.locator('#navCase').selectOption('vector-source-independent-001');
 await page.locator('[data-action="stage"][data-stage="vector"]').first().click();
 const text=await page.locator('#caseDetails').innerText();
 if(!text.includes('302340300'))throw Error('Incorrect source holdout navigation');
 await page.screenshot({path:path.join(out,'report_overview.png'),fullPage:false});
 await page.locator('#search').fill('vector-source-independent-001');
 if(await page.locator('#caseTable tbody tr').count()!==1)throw Error('Search broken');
 if(errors.length)throw Error(JSON.stringify(errors));
 const result={cases:111,arms:audit.arms,source_manifest_checked:Object.keys(sources).length,source_request_resolves:!!clicked,page_errors:errors,
  complete:audit.complete,note:'Headless Edge; actual source launching mocked. Physical source paths and indexed page/paragraph fields checked.'};
 fs.writeFileSync(path.join(out,'report_ui_audit.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result));await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
