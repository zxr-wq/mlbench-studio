"""Check rendered chart geometry, not just the presence of an SVG."""
import asyncio
import base64
import importlib.util
import json
from pathlib import Path
import urllib.request

import websockets

spec=importlib.util.spec_from_file_location("browserqa",Path(__file__).with_name("browser-delivery-qa.py"))
qa=importlib.util.module_from_spec(spec)
spec.loader.exec_module(qa)
OUT=qa.ROOT/"tmp"/"chart-layout-qa"

CARD="[...document.querySelectorAll('.evidence-chart')].find(c=>c.querySelector('h3')?.textContent==='单样本类别概率')"


async def capture(browser,expression,filename):
    await browser.evaluate(f"({expression}).scrollIntoView({{block:'center'}})")
    await asyncio.sleep(.2)
    bounds=await browser.evaluate(f"(()=>{{let r=({expression}).getBoundingClientRect();return {{x:r.x+scrollX,y:r.y+scrollY,width:r.width,height:r.height,scale:1}}}})()")
    result=await browser.call("Page.captureScreenshot",{"format":"png","captureBeyondViewport":True,"clip":bounds})
    (OUT/filename).write_bytes(base64.b64decode(result["data"]))


async def check_plot(browser):
    result=await browser.evaluate(f"""(()=>{{
      const card={CARD},svg=card.querySelector('svg');
      const bars=[...svg.querySelectorAll('.plot-bar')].map(b=>({{x:+b.getAttribute('x'),y:+b.getAttribute('y'),width:+b.getAttribute('width'),height:+b.getAttribute('height')}}));
      const axes=svg.querySelector('.plot-axes'),data=svg.querySelector('.plot-data');
      const firstTick=svg.querySelector('.plot-y-tick').getBBox();
      const dataClip=data.getAttribute('clip-path');
      return {{bars,clip:dataClip,clipExists:!!svg.querySelector('clipPath'),axesAboveData:!!(data.compareDocumentPosition(axes)&Node.DOCUMENT_POSITION_FOLLOWING),tickRight:firstTick.x+firstTick.width}};
    }})()""")
    assert result["clipExists"] and result["clip"]
    assert result["axesAboveData"], result
    assert result["tickRight"]<66, result
    for bar in result["bars"]:
        assert bar["x"]>=66-1e-6 and bar["x"]+bar["width"]<=606+1e-6,bar
        assert bar["y"]>=30-1e-6 and bar["y"]+bar["height"]<=266+1e-6,bar
    return result


async def main():
    OUT.mkdir(parents=True,exist_ok=True)
    pages=json.load(urllib.request.urlopen("http://127.0.0.1:9230/json/list"))
    page=next(p for p in pages if p.get("type")=="page" and p.get("url","").startswith(qa.BASE))
    async with websockets.connect(page["webSocketDebuggerUrl"],max_size=30_000_000) as socket:
        browser=qa.Browser(socket)
        await browser.call("Runtime.enable")
        await browser.call("Page.enable")
        await browser.call("Emulation.setDeviceMetricsOverride",{"width":1440,"height":1000,"deviceScaleFactor":1,"mobile":False})
        results={}
        for dataset in ("moons","iris","digits"):
            await browser.experiment("knn",dataset)
            results[dataset]=await check_plot(browser)
            sample=await browser.evaluate("(async()=>{let id=location.hash.split('/').at(-1).replace('EXP-','');let record=await fetch('/api/experiments/'+id).then(r=>r.json());let rows=record.result.visualization.probabilities.rows;let index=rows.findIndex(r=>JSON.stringify(r.values)!==JSON.stringify(rows[0].values));return {index:Math.max(1,index),values:rows[Math.max(1,index)].values}})()")
            await browser.evaluate(f"(()=>{{let input=({CARD}).querySelector('input[type=number]');input.value='{sample['index']+1}';input.dispatchEvent(new Event('change',{{bubbles:true}}))}})()")
            await asyncio.sleep(.15)
            after=await check_plot(browser)
            for bar,value in zip(after["bars"],sample["values"]):
                assert abs(bar["height"]-236*value)<.001,(bar,value)
            await capture(browser,CARD,f"{dataset}-probabilities-100.png")
            await browser.evaluate(f"({CARD}).querySelector('button[aria-label=\"概率图测试样本：上一个样本\"]').click()")
            assert await browser.evaluate(f"+({CARD}).querySelector('input[type=number]').value==={sample['index']}")
            await browser.evaluate(f"({CARD}).querySelector('button[aria-label=\"概率图测试样本：下一个样本\"]').click()")
            assert await browser.evaluate(f"+({CARD}).querySelector('input[type=number]').value==={sample['index']+1}")
            await browser.call("Emulation.setDeviceMetricsOverride",{"width":1152,"height":800,"deviceScaleFactor":1.25,"mobile":False})
            await check_plot(browser)
            await capture(browser,CARD,f"{dataset}-probabilities-125.png")
            await browser.call("Emulation.setDeviceMetricsOverride",{"width":390,"height":844,"deviceScaleFactor":1,"mobile":False})
            await check_plot(browser)
            assert await browser.evaluate("document.documentElement.scrollWidth<=innerWidth+2")
            await capture(browser,CARD,f"{dataset}-probabilities-mobile.png")
            await browser.call("Emulation.setDeviceMetricsOverride",{"width":1440,"height":1000,"deviceScaleFactor":1,"mobile":False})
        # Reproduce the supplied six-column table (seed, fold, four metrics).
        await browser.navigate('/experiments/new?dataset=iris&model=knn')
        await browser.wait("document.querySelectorAll('.step-nav li').length===3")
        await browser.evaluate("document.querySelectorAll('.step-nav li')[2].click()")
        await browser.evaluate("[...document.querySelectorAll('.metric-choice-list button')].filter(b=>/Precision|Recall/.test(b.textContent)&&!b.classList.contains('selected')).forEach(b=>b.click())")
        await browser.click('运行实验')
        await browser.wait("location.hash.includes('/runs/') && !!document.querySelector('.evaluation-details')")
        await browser.evaluate("document.querySelector('.evaluation-details').open=true")
        centers=await browser.evaluate("""(()=>{
            const table=document.querySelector('.evaluation-details table');
            const headers=[...table.querySelectorAll('thead th')];
            const cells=[...table.querySelectorAll('tbody tr:first-child td')];
            return headers.map((h,i)=>{
                const a=h.getBoundingClientRect(),b=cells[i].getBoundingClientRect();
                return {header:(a.left+a.right)/2,cell:(b.left+b.right)/2,
                        headerAlign:getComputedStyle(h).textAlign,cellAlign:getComputedStyle(cells[i]).textAlign};
            });
        })()""")
        assert all(abs(pair["header"]-pair["cell"])<.5 and pair["headerAlign"]==pair["cellAlign"]=="center" for pair in centers),centers
        assert len(centers)==6,centers
        await capture(browser,"document.querySelector('.evaluation-details')","evaluation-alignment.png")
        await browser.navigate('/benchmark')
        before=await browser.evaluate("document.querySelector('.benchmark-table')?.innerText")
        assert before
        headings=await browser.evaluate("[...document.querySelectorAll('.benchmark-table thead th')].slice(1).map(h=>getComputedStyle(h).textAlign)")
        assert all(value=="center" for value in headings),headings
        await capture(browser,"document.querySelector('.benchmark-matrix')","benchmark-alignment.png")
        assert before==await browser.evaluate("document.querySelector('.benchmark-table').innerText")
        assert not browser.errors,browser.errors
        (OUT/'verification.json').write_text(json.dumps({"probability_geometry":results,"table_centers":centers,"checks":["2-3-10-class-bars","data-clipping","axes-layer-order","sample-jump-and-prev-next","actual-model-probability-values","100-125-percent-and-mobile","evaluation-column-centering","benchmark-header-centering-and-values"],"console_errors":browser.errors},ensure_ascii=False,indent=2),encoding='utf-8')
        print('PASS: chart geometry, alignment, navigation, and actual values',flush=True)


asyncio.run(main())
