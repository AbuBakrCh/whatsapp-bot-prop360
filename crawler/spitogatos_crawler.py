import asyncio
import random
import re
from typing import List

from bs4 import BeautifulSoup
from curl_cffi import CurlMime
from curl_cffi.requests import AsyncSession

class AuthExpiredError(Exception):
    pass

class ServerError(Exception):
    pass

class SpitogatosCrawler:
    DOMAIN = "https://www.spitogatos.gr"
    IMAGE_UPLOAD_URL = "https://valid.prop360.pro/check_images"
    PROPERTY_API_URL = "https://solomon.realestate/api/merchant/form_data"

    def __init__(self, collection, db, reese84: str, token: str):
        self.collection = collection
        self.db = db
        self.reese84 = reese84
        self.token = token

    # =========================
    # FETCH HTML
    # =========================
    async def fetch_html(self, session, url: str) -> str:
        response = await session.get(
            url,
            impersonate="chrome120",
            timeout=60,
            headers={
                "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "accept-language": "en",
                "referer": self.DOMAIN,
                "user-agent": "Mozilla/5.0",
            },
            cookies={"reese84": self.reese84},
        )

        if response.status_code >= 500:
            raise ServerError(f"Spitogatos server error: {response.status_code}")

        if response.status_code in (401, 403, 405):
            raise AuthExpiredError("Spitogatos session expired (cookie invalid)")

        return response.text

    # =========================
    # EXTRACT LINKS
    # =========================
    def extract_property_urls(self, html: str) -> List[str]:
        soup = BeautifulSoup(html, "lxml")
        links = set()

        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/property/" in href:
                if not href.startswith("http"):
                    href = self.DOMAIN + href
                links.add(href.rstrip("/"))

        return list(links)

    # =========================
    # EXTRACT PROPERTY DATA
    # =========================
    def extract_property_data(self, html: str) -> dict:
        soup = BeautifulSoup(html, "lxml")

        result = {
            "title": None,
            "price": None,
            "address": None,
            "features": {},
            "amenities": [],
            "images": [],
        }

        title_el = soup.select_one("h1.property__title")
        if title_el:
            result["title"] = title_el.get_text(strip=True)

        price_el = soup.select_one(".property__price__text")
        if price_el:
            result["price"] = price_el.get_text(strip=True)

        address_el = soup.select_one(".property__address")
        if address_el:
            result["address"] = address_el.get_text(strip=True)

        details = soup.select("dl.property__details dt, dl.property__details dd")
        key = None
        for el in details:
            if el.name == "dt":
                key = el.get_text(strip=True)
            elif el.name == "dd" and key:
                result["features"][key] = el.get_text(strip=True)
                key = None

        for li in soup.select("ul.property__features li"):
            svg = li.find("svg")
            text_el = li.find("span")
            if svg and text_el:
                result["amenities"].append({
                    "name": text_el.get_text(strip=True),
                    "available": "i-correct" in str(svg)
                })

        for img in soup.select(".property__gallery__item img"):
            src = img.get("src") or img.get("data-src")
            if src:
                result["images"].append({
                    "url": src.replace("_300x220", "_900x675"),
                    "alt": img.get("alt", "")
                })

        return result

    # =========================
    # DESCRIPTION BUILDER
    # =========================
    def build_description_html(self, data: dict) -> str:
        html = []

        if data.get("title"):
            html.append(f"<h2>{data['title']}</h2>")

        if data.get("features"):
            html.append("<h3>Features</h3><ul>")
            for k, v in data["features"].items():
                html.append(f"<li><b>{k}:</b> {v}</li>")
            html.append("</ul>")

        if data.get("amenities"):
            html.append("<h3>Amenities</h3><ul>")
            for item in data["amenities"]:
                if item["available"]:
                    html.append(f"<li>{item['name']}</li>")
            html.append("</ul>")

        return "".join(html)

    # =========================
    # IMAGE UPLOAD
    # =========================
    async def upload_image(self, session, image_url: str, merchant_id: str, folder: str):
        img_resp = await session.get(image_url)
        if img_resp.status_code != 200:
            return None

        form = CurlMime()
        form.addpart(
            name="files",
            filename="image.png",
            data=img_resp.content,
            content_type="image/png",
        )
        form.addpart(name="folder", data=folder)
        form.addpart(name="merchantId", data=merchant_id)

        resp = await session.post(
            self.IMAGE_UPLOAD_URL,
            multipart=form,
            headers={
                "authorization": f"Bearer {self.token}",
                "origin": "https://solomon.realestate",
                "referer": "https://solomon.realestate/",
            }
        )

        if resp.status_code >= 500:
            raise ServerError(f"Image upload server error: {resp.status_code}")

        if resp.status_code in (401, 403):
            raise AuthExpiredError("Solomon token expired during image upload")

        if resp.status_code == 200:
            result = resp.json()
            return result[0] if isinstance(result, list) else result


    # =========================
    # PAYLOAD BUILDER
    # =========================
    def build_payload(self, data, images_meta, property_url):
        return {
            "formId": "67cdbbb067127a1afc8154f0",
            "indicator": "properties",
            "isPublic": False,
            "destructive": False,
            "data": {
                "field-1741536181001-wd8it2quy": data["title"],
                "field-1741536272085-yi74oirib": re.sub(r"[^\d]", "", data["price"]),
                "field-1744021392093-03a295o25": data["address"],
                "field-1741536304675-8m3nlhbmy": self.build_description_html(data),
                "field-1741536446663-7s5bcmilv": images_meta,
                "spitogatos_url": property_url
            }
        }

    async def is_already_processed(self, property_url: str) -> bool:
        doc = await self.collection.find_one({
            "data.spitogatos_url": property_url
        })
        return doc is not None

    # =========================
    # SUBMIT PROPERTY
    # =========================
    async def submit_property(self, session, payload):
        resp = await session.post(
            self.PROPERTY_API_URL,
            json=payload,
            headers={
                "authorization": f"Bearer {self.token}",
                "content-type": "application/json",
            }
        )

        if resp.status_code >= 500:
            raise ServerError(f"Property submission server error: {resp.status_code}")

        if resp.status_code in (401, 403):
            raise AuthExpiredError("Solomon token expired during property submission")

        print("Submit status:", resp.status_code)
        print(resp.text)

    async def should_stop(self):
        control = await self.db.job_control.find_one(
            {"_id": "spitogatos_crawler_job"}
        )
        return control and control.get("status") == "stop"

    # =========================
    # MAIN CRAWL
    # =========================
    async def crawl(self, base_url: str, total_pages: int, delay: float = 3.0):
        total = 0

        await self.db.job_control.update_one(
            {"_id": "spitogatos_crawler_job"},
            {"$set": {"status": "run"}},
            upsert=True
        )

        async with AsyncSession() as session:
            try:
                for page in range(1, total_pages + 1):
                    if await self.should_stop():
                        print("[CRAWLER] Stopped by user")
                        return total

                    url = base_url if page == 1 else f"{base_url}/page_{page}"
                    print(f"[CRAWLER] Page {page}: {url}")

                    html = await self.fetch_html(session, url)

                    if not html:
                        continue

                    property_urls = self.extract_property_urls(html)

                    for property_url in property_urls:
                        if await self.should_stop():
                            print("[CRAWLER] Stopped mid-processing")
                            return total

                        print(f"[CRAWLER] Fetching property: {property_url}")

                        if await self.is_already_processed(property_url):
                            print(f"[CRAWLER] Skipping duplicate: {property_url}")
                            continue

                        property_html = await self.fetch_html(session, property_url)

                        if not property_html:
                            continue

                        data = self.extract_property_data(property_html)

                        images_meta = []
                        for img in data["images"][:10]:
                            meta = await self.upload_image(
                                session,
                                img["url"],
                                merchant_id="3124d713-067b-427c-9672-1cfee6058246", #invest greece research merchant id
                                folder="field-1741536446663-7s5bcmilv"
                            )
                            if meta:
                                images_meta.append(meta)

                        payload = self.build_payload(data, images_meta, property_url)
                        await self.submit_property(session, payload)

                        total += 1

                        await asyncio.sleep(random.uniform(2, 4))

                    await asyncio.sleep(delay + random.uniform(3, 6))

            except AuthExpiredError as e:
                print(f"[CRAWLER STOPPED] {e}")
                raise e  # propagate to API

        print(f"[CRAWLER] DONE → {total} properties fetched")
        return total