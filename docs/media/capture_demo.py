# Records the README screenshot and GIF of the VeriSQL demo.
#
#   python verisql/app.py --replay --port 7861        (in another terminal)
#   python docs/media/capture_demo.py                 (writes docs/media/verisql-*.png and verisql-demo.gif)
#
# Needs: pip install playwright pillow && playwright install chromium
import io
import os
import sys
import time

from PIL import Image
from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7861/"
OUT = os.path.dirname(os.path.abspath(__file__))
SIZE = {"width": 1280, "height": 800}


def shot(page):
    return Image.open(io.BytesIO(page.screenshot())).convert("RGB")


with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport=SIZE, device_scale_factor=1)
    page.goto(URL)
    page.wait_for_selector("button.vs-chip", timeout=60000)
    time.sleep(1.5)
    frames = [(shot(page), 1500)]
    frames[0][0].save(os.path.join(OUT, "verisql-home.png"))

    page.locator("button.vs-chip").first.click()
    start = time.time()
    while time.time() - start < 30:  # one frame every 0.4 s while the steps appear
        frames.append((shot(page), 400))
        if page.locator(".vs-status summary").count():
            break
        time.sleep(0.4)
    page.wait_for_selector(".vs-answer", timeout=30000)
    time.sleep(1.5)
    answer = shot(page)
    answer.save(os.path.join(OUT, "verisql-answer.png"))
    frames.append((answer, 3500))
    browser.close()

small = [f.resize((960, 600), Image.LANCZOS) for f, _ in frames]
small[0].save(os.path.join(OUT, "verisql-demo.gif"), save_all=True, append_images=small[1:],
              duration=[d for _, d in frames], loop=0, optimize=True)
print(f"{len(frames)} frames -> {OUT}/verisql-demo.gif")
