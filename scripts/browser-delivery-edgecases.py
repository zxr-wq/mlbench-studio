"""Failure and recovery acceptance checks against the isolated local browser."""
import asyncio
import importlib.util
import json
from pathlib import Path
import urllib.request

import websockets

spec = importlib.util.spec_from_file_location("browserqa", Path(__file__).with_name("browser-delivery-qa.py"))
qa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qa)


async def prepare(browser, model="knn", dataset="iris"):
    await browser.navigate(f"/experiments/new?dataset={dataset}&model={model}")
    await browser.wait("document.querySelectorAll('.step-nav li').length===3")
    await browser.evaluate("document.querySelectorAll('.step-nav li')[2].click()")
    await asyncio.sleep(.2)


async def main():
    pages = json.load(urllib.request.urlopen("http://127.0.0.1:9230/json/list"))
    page = next(p for p in pages if p.get("type") == "page" and p.get("url", "").startswith(qa.BASE))
    async with websockets.connect(page["webSocketDebuggerUrl"], max_size=30_000_000) as socket:
        browser = qa.Browser(socket)
        await browser.call("Runtime.enable")
        await browser.call("Page.enable")
        await browser.call("Emulation.setDeviceMetricsOverride", {"width":1440,"height":1050,"deviceScaleFactor":1,"mobile":False})
        checks = []

        # Only F1 is requested, with three seeds. It must not invent Accuracy=0.
        await prepare(browser)
        await browser.evaluate("[...document.querySelectorAll('.metric-choice-list button')].find(b=>b.textContent.includes('Accuracy')).click()")
        await browser.evaluate("(()=>{const s=document.querySelector('.evaluation-fields select');s.value='3';s.dispatchEvent(new Event('change',{bubbles:true}))})()")
        await browser.click("运行实验")
        await browser.wait("location.hash.includes('/runs/') && !!document.querySelector('.evidence-grid')")
        await asyncio.sleep(.4)
        assert await browser.evaluate("document.querySelector('.detail-metrics').textContent.includes('Macro F1') && !document.querySelector('.detail-metrics').textContent.includes('Accuracy')")
        assert await browser.evaluate("document.querySelector('.evidence-provenance').textContent.includes('3 次运行的均值')")
        await browser.screenshot("three-seeds-f1-only",height=1300)
        await browser.evaluate("[...document.querySelectorAll('a')].find(a=>a.textContent.trim()==='按此配置重跑').click()")
        await browser.wait("!!document.querySelector('.metric-choice-list')")
        assert await browser.evaluate("[...document.querySelectorAll('.metric-choice-list .selected')].length===1 && document.querySelector('.metric-choice-list .selected').textContent.includes('Macro F1')")
        checks.append("f1-only-and-three-seed-provenance-and-exact-rerun")

        # A WebSocket which never delivers any event must trigger HTTP fallback.
        await prepare(browser)
        await browser.evaluate("window.WebSocket=class{close(){}}")
        await browser.click("运行实验")
        await browser.wait("location.hash.includes('/runs/')", timeout=20)
        checks.append("silent-websocket-http-fallback")

        await prepare(browser)
        await browser.evaluate("window.WebSocket=class{constructor(){throw new Error('blocked')}}")
        await browser.click("运行实验")
        await browser.wait("location.hash.includes('/runs/')", timeout=20)
        checks.append("websocket-constructor-failure-http-fallback")

        # Refresh after submission, before the client receives completion.
        await prepare(browser)
        await browser.evaluate("window.WebSocket=class{close(){}}")
        await browser.click("运行实验")
        await browser.wait("!!localStorage.getItem('mlbench-pending')")
        pending = await browser.evaluate("JSON.parse(localStorage.getItem('mlbench-pending')).id")
        await browser.call("Page.reload")
        await browser.wait("!localStorage.getItem('mlbench-pending') && JSON.parse(localStorage.getItem('mlbench-studio-runs')||'[]').some(r=>r.backendId===" + json.dumps(pending) + ")", timeout=30)
        checks.append("pending-experiment-refresh-recovery")

        # Summary storage quota failures must not fail a completed experiment.
        await prepare(browser, "naive_bayes")
        await browser.evaluate("(()=>{const original=Storage.prototype.setItem;Storage.prototype.setItem=function(k,v){if(k==='mlbench-studio-runs')throw new DOMException('quota','QuotaExceededError');return original.call(this,k,v)}})()")
        await browser.click("运行实验")
        await browser.wait("location.hash.includes('/runs/') && !!document.querySelector('.evidence-grid')")
        saved_id = await browser.evaluate("location.hash.split('/').at(-1)")
        assert await browser.evaluate("document.body.innerText.includes('存储空间不足')")
        await browser.call("Page.reload")
        await browser.wait("!!document.querySelector('.evidence-grid')")
        assert await browser.evaluate("location.hash.endsWith(" + json.dumps(saved_id) + ")")
        checks.append("quota-failure-still-saves-full-result-in-indexeddb")

        # Export a vector-metric run and a figure; neither export may be empty.
        await browser.experiment("pca", "wine")
        await browser.call("Browser.setDownloadBehavior", {"behavior":"allow","downloadPath":str(qa.OUT)})
        await browser.click("下载指标 CSV")
        await browser.evaluate("document.querySelector('button[aria-label=\"下载图表\"]').click()")
        await asyncio.sleep(.5)
        csv_path = sorted(qa.OUT.glob("*pca-result.csv"),key=lambda p:p.stat().st_mtime)[-1]
        assert "explained_variance_ratio[1]" in csv_path.read_text(encoding="utf-8-sig")
        assert next(qa.OUT.glob("mlbench-chart*.svg")).stat().st_size > 200
        checks.append("pca-vector-csv-and-svg-image-export")

        for model in ("decision_tree", "random_forest", "gradient_boosting", "logistic_regression"):
            await browser.experiment(model,"iris")
            if model=="decision_tree":
                assert await browser.evaluate("document.querySelector('.tree-svg').textContent.includes('setosa')")
            if model in {"random_forest","gradient_boosting"}:
                assert await browser.evaluate("document.body.innerText.includes('集成规模与测试表现')")
            if model=="logistic_regression":
                await browser.screenshot("logistic-loss",height=1600)
        checks.append("tree-real-class-names-and-ensemble-curves-and-softmax-loss")

        await prepare(browser)
        await browser.evaluate("(()=>{let n=document.querySelector('.seed-field input');n.value='-3';n.dispatchEvent(new Event('input',{bubbles:true}))})()")
        await browser.click("运行实验")
        assert await browser.evaluate("document.querySelector('.experiment-error').textContent.includes('随机种子')")
        await prepare(browser)
        await browser.evaluate("(()=>{const original=window.fetch;window.fetch=(input,options)=>String(input).endsWith('/api/experiments')?Promise.resolve(new Response(JSON.stringify({detail:'服务暂时不可用'}),{status:503,headers:{'Content-Type':'application/json'}})):original(input,options)})()")
        await browser.click("运行实验")
        await browser.wait("!!document.querySelector('.experiment-error')")
        assert await browser.evaluate("document.querySelector('.experiment-error').textContent.includes('服务暂时不可用')")
        checks.append("invalid-seed-and-api-error-visible")

        await browser.navigate("/runs")
        malformed=qa.OUT/"invalid-results.json"
        malformed.write_text(json.dumps({"experiments":[{"id":"invalid"}]}),encoding="utf-8")
        root=await browser.call("DOM.getDocument")
        node=await browser.call("DOM.querySelector",{"nodeId":root["root"]["nodeId"],"selector":".import-control input"})
        await browser.call("DOM.setFileInputFiles",{"nodeId":node["nodeId"],"files":[str(malformed)]})
        await browser.wait("document.body.innerText.includes('无效记录')")
        checks.append("invalid-json-import-rejected")
        await browser.navigate("/")
        assert not browser.errors, browser.errors
        (qa.OUT/"edgecases.json").write_text(json.dumps({"checks":checks,"console_errors":browser.errors},ensure_ascii=False,indent=2),encoding="utf-8")
        print("PASS",json.dumps(checks),flush=True)


asyncio.run(main())
