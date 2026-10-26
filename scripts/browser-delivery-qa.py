"""Local Edge CDP delivery checks. Uses real UI experiments, isolated browser profile."""
import asyncio
import base64
import json
import time
from pathlib import Path
import urllib.request

import websockets

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp" / "delivery-qa"
BASE = "http://127.0.0.1:5173/mlbench-studio/"


class Browser:
    def __init__(self, socket):
        self.socket = socket
        self.serial = 0
        self.errors = []

    async def call(self, method, params=None):
        self.serial += 1
        identity = self.serial
        await self.socket.send(json.dumps({"id": identity, "method": method, "params": params or {}}))
        while True:
            message = json.loads(await self.socket.recv())
            if message.get("method") == "Runtime.exceptionThrown":
                self.errors.append(message["params"]["exceptionDetails"])
            if message.get("id") == identity:
                if "error" in message:
                    raise RuntimeError(message["error"])
                return message.get("result", {})

    async def evaluate(self, expression):
        result = await self.call("Runtime.evaluate", {"expression": expression, "awaitPromise": True, "returnByValue": True})
        if "exceptionDetails" in result:
            raise AssertionError(result["exceptionDetails"])
        return result.get("result", {}).get("value")

    async def wait(self, expression, timeout=120):
        for _ in range(int(timeout * 4)):
            if await self.evaluate(expression):
                return
            await asyncio.sleep(.25)
        raise AssertionError("Timed out waiting for " + expression + "\n" + str(await self.evaluate("document.body.innerText.slice(-3000)")))

    async def navigate(self, route):
        origin = await self.evaluate("performance.timeOrigin")
        # A unique search parameter forces a full navigation. Hash-only
        # navigation followed immediately by reload can reload the old page.
        await self.call("Page.navigate", {"url": BASE + "?_qa=" + str(time.monotonic_ns()) + "#" + route})
        await self.wait("performance.timeOrigin!==" + str(origin) + " && document.readyState==='complete' && location.hash===" + json.dumps("#"+route) + " && !!document.querySelector('.app-nav')")
        await asyncio.sleep(.6)

    async def click(self, text):
        value = await self.evaluate("(()=>{const button=[...document.querySelectorAll('button')].find(b=>b.textContent.trim()===" + json.dumps(text) + ");if(!button||button.disabled)return false;button.click();return true})()")
        assert value, "Button unavailable: " + text
        await asyncio.sleep(.15)

    async def screenshot(self, name, width=1440, height=1050):
        await self.call("Emulation.setDeviceMetricsOverride", {"width": width, "height": height, "deviceScaleFactor": 1, "mobile": False})
        await asyncio.sleep(.35)
        result = await self.call("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": True})
        (OUT / (name + ".png")).write_bytes(base64.b64decode(result["data"]))

    async def experiment(self, model, dataset, implementation="scratch"):
        print("UI experiment", model, dataset, implementation, flush=True)
        await self.navigate(f"/experiments/new?dataset={dataset}&model={model}")
        await self.wait("document.querySelectorAll('.step-nav li').length===3")
        if implementation == "sklearn":
            await self.evaluate("(()=>{const select=document.querySelector('.implementation-panel select');select.value='sklearn';select.dispatchEvent(new Event('change',{bubbles:true}))})()")
        await self.evaluate("document.querySelectorAll('.step-nav li')[2].click()")
        await asyncio.sleep(.2)
        await self.click("运行实验")
        await self.wait("location.hash.includes('/runs/')")
        await self.wait("document.querySelectorAll('.evidence-chart').length>0")
        await asyncio.sleep(.6)
        assert await self.evaluate("!document.querySelector('.experiment-error')")
        return await self.evaluate("location.hash.split('/').at(-1)")


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pages = json.load(urllib.request.urlopen("http://127.0.0.1:9230/json/list"))
    page = next(p for p in pages if p.get("type") == "page" and p.get("url", "").startswith(BASE))
    async with websockets.connect(page["webSocketDebuggerUrl"], max_size=30_000_000) as socket:
        browser = Browser(socket)
        await browser.call("Runtime.enable")
        await browser.call("Page.enable")
        await browser.call("Browser.setDownloadBehavior", {"behavior": "allow", "downloadPath": str(OUT)})
        await browser.call("Emulation.setDeviceMetricsOverride", {"width":1440,"height":1050,"deviceScaleFactor":1,"mobile":False})
        records = {}
        records["iris-knn"] = await browser.experiment("knn", "iris")
        await browser.wait("!!document.querySelector('.neighbor-list')")
        await browser.evaluate("(()=>{let input=document.querySelector('input[aria-label=\"演示近邻数量 K\"]');input.value='9';input.dispatchEvent(new Event('input',{bubbles:true}))})()")
        assert await browser.evaluate("document.querySelectorAll('.neighbor-list li').length===9")
        await browser.screenshot("iris-knn", height=1800)
        await browser.click("下载完整 JSON")
        await browser.click("下载指标 CSV")
        await asyncio.sleep(.4)
        json_file = next(OUT.glob("*knn-result.json"))
        exported = json.loads(json_file.read_text(encoding="utf-8"))
        assert exported["experiment"]["config"]["preprocessing"] == ["standard_scaler"]
        assert exported["experiment"]["visualization"]["knn"]["examples"]
        await browser.call("Page.reload")
        await browser.wait("!!document.querySelector('.neighbor-list')")
        await browser.evaluate("document.querySelector('button[aria-label=\"放大图表\"]').click()")
        assert await browser.evaluate("!!document.querySelector('.chart-expanded')")
        await browser.evaluate("window.dispatchEvent(new KeyboardEvent('keydown',{key:'Escape'}))")
        assert await browser.evaluate("!document.querySelector('.chart-expanded')")

        records["digits-knn"] = await browser.experiment("knn", "digits")
        await browser.click("按真实类别归一化")
        assert await browser.evaluate("document.querySelector('.matrix-svg').textContent.includes('%')")
        await browser.screenshot("digits-desktop", height=1200)
        await browser.screenshot("digits-mobile", width=390, height=844)
        assert await browser.evaluate("document.documentElement.scrollWidth<=window.innerWidth+2"), "mobile viewport overflow"
        await browser.call("Emulation.setDeviceMetricsOverride", {"width":1440,"height":1050,"deviceScaleFactor":1,"mobile":False})

        records["moons-svm"] = await browser.experiment("svm", "moons", "sklearn")
        assert await browser.evaluate("!!document.querySelector('.boundary-plot')")
        await browser.screenshot("moons-svm",height=1900)
        records["iris-kmeans"] = await browser.experiment("kmeans", "iris")
        await browser.click("重置")
        await browser.click("下一步")
        await browser.click("播放")
        await asyncio.sleep(1)
        await browser.screenshot("iris-kmeans",height=1300)
        records["wine-pca"] = await browser.experiment("pca", "wine")
        assert await browser.evaluate("document.body.innerText.includes('累计解释方差')")
        await browser.screenshot("wine-pca",height=1500)
        records["diabetes-linear"] = await browser.experiment("linear_regression", "diabetes")
        assert await browser.evaluate("document.body.innerText.includes('残差诊断')")
        await browser.screenshot("diabetes-linear",height=1600)

        await browser.navigate("/benchmark")
        await browser.evaluate("[...document.querySelectorAll('button')].find(b=>b.textContent.includes('运行 6 组实验')).click()")
        await browser.wait("document.body.innerText.includes('批量实验完成')", timeout=150)
        assert not await browser.evaluate("!!document.querySelector('.batch-errors')")
        assert await browser.evaluate("document.querySelectorAll('.benchmark-table tbody tr').length>=3")
        await browser.screenshot("benchmark",height=1450)
        await browser.click("导出比较 CSV")

        await browser.navigate("/runs")
        root = await browser.call("DOM.getDocument")
        node = await browser.call("DOM.querySelector", {"nodeId": root["root"]["nodeId"], "selector": ".import-control input"})
        await browser.call("DOM.setFileInputFiles", {"nodeId": node["nodeId"], "files": [str(json_file)]})
        await browser.wait("document.body.innerText.includes('已成功导入')")
        await browser.screenshot("history",height=1000)
        await browser.navigate("/runs/unknown-experiment")
        assert await browser.evaluate("document.body.innerText.includes('没有找到')")
        await browser.navigate("/algorithms")
        assert await browser.evaluate("document.querySelectorAll('.algorithm-card').length===10")
        await browser.navigate("/datasets")
        assert await browser.evaluate("document.querySelectorAll('.dataset-row').length===8")
        await browser.navigate("/")
        await browser.screenshot("overview",height=1050)
        assert not browser.errors, browser.errors
        (OUT / "verification.json").write_text(json.dumps({"ui_experiments": records,"console_errors":browser.errors,"checks":["real-ui-training","indexeddb-reload","knn-interaction","normalized-matrix","mobile-overflow","chart-expand-escape","full-json-csv-download","kmeans-playback","benchmark-six-runs","json-import","missing-run","10-algorithms","8-datasets"]},ensure_ascii=False,indent=2), encoding="utf-8")
        print("PASS", json.dumps(records), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
