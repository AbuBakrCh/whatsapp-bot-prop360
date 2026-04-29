import asyncio
import random
from typing import List
import logging
import re
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from bson import ObjectId
from curl_cffi import CurlMime
from curl_cffi.requests import AsyncSession

logger = logging.getLogger("spitogatos_crawler")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class AuthExpiredError(Exception):
    pass

class ServerError(Exception):
    pass

class SpitogatosCrawler:
    DOMAIN = "https://www.spitogatos.gr"
    BASE_URL = "https://prop360.pro/api/integration"

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

        logger.info(f"[FETCH HTML] {url} → {response.status_code}")
        preview = (response.text[:300] or "").replace("\n", " ")
        logger.info(f"[FETCH HTML BODY PREVIEW] {preview}")

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
    def extract_property_data(self, url: str, html: str) -> dict:
        soup = BeautifulSoup(html, "lxml")

        result = {
            "title": None,
            "price": None,
            "address": None,
            "features": {},
            "amenities": [],
            "images": [],
            "agency": None,
            "document_folder_number": None
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

        agency_info = self.extract_agency_info(soup)
        agency_info["phone"] = self.extract_phone(html)
        result["agency"] = self.format_agency(agency_info)

        result["document_folder_number"] = url
        return result

    def extract_agency_info(self, soup: BeautifulSoup) -> dict:
        agency = {
            "name": None,
            "website": None,
            "agent_name": None,
        }

        # Agency name
        name_el = soup.select_one(".property__agency__info h3 a")
        if name_el:
            agency["name"] = name_el.get_text(strip=True)

        # Website
        website_el = soup.select_one(".property__agency__website")
        if website_el:
            agency["website"] = website_el.get("href")

        # Agent name (person)
        agent_el = soup.select_one(".property__agency__contact")
        if agent_el:
            agency["agent_name"] = agent_el.get_text(strip=True)

        return agency

    def extract_phone(self, html: str) -> str | None:
        # Match international phone numbers like +302111995637
        match = re.search(r"\+\d{8,15}", html)
        if match:
            return match.group(0)
        return None

    def format_agency(self, agency: dict) -> str:
        parts = []

        if agency.get("name"):
            parts.append(agency["name"])

        if agency.get("agent_name"):
            parts.append(f"Agent: {agency['agent_name']}")

        if agency.get("phone"):
            parts.append(f"Phone: {agency['phone']}")

        if agency.get("website"):
            parts.append(f"Website: {agency['website']}")

        return " | ".join(parts)

    # =========================
    # DESCRIPTION
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
    # FIELD MAPPING
    # =========================
    def map_fields(self, data):
        features = data.get("features", {})

        def clean_price(price):
            return int(re.sub(r"[^\d]", "", price)) if price else None

        def clean_number(val):
            if not val:
                return None
            return int(re.sub(r"[^\d]", "", val))

        def parse_construction_year(val):
            if not val:
                return None
            val = val.strip()
            if re.search(r"\d{4}", val):
                return int(re.search(r"\d{4}", val).group())
            return val

        return {
            "Title": f"{data.get('title')} - {data.get('address')}",
            "Description": self.build_description_html(data),
            "Address": data.get("address"),

            "Price": clean_price(data.get("price")),
            "Square Meters": clean_number(features.get("Surface")),
            "Construction Year": parse_construction_year(features.get("Construction year")),

            "Bathrooms": clean_number(features.get("Bathrooms")),
            "Bedroom": clean_number(features.get("Bedrooms") or features.get("Rooms")),

            "Floor": features.get("Floor"),
            "Heating System": features.get("Heating system"),

            "isPublic": False,
            "Property Owner Real Estate Agent": data.get("agency"),
            "Document Folder Number": data.get("document_folder_number"),
        }

    # =========================
    # CREATE PROPERTY
    # =========================
    async def create_property(self, session, mapped_data):
        url = f"{self.BASE_URL}/properties"

        resp = await session.post(
            url,
            json={"data": mapped_data},
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            }
        )

        if resp.status_code >= 500:
            raise ServerError("Property API server error")

        if resp.status_code in (401, 403):
            raise AuthExpiredError("Token expired")

        if resp.status_code != 200:
            print("Create failed:", resp.text)
            return None

        result = resp.json()
        return result.get("property", {}).get("id")

    # =========================
    # UPLOAD IMAGES
    # =========================
    async def upload_images(self, session, property_id, images):
        url = f"{self.BASE_URL}/properties/{property_id}/images"

        form = CurlMime()

        added = False

        for i, img in enumerate(images[:10]):
            resp = await session.get(img["url"])
            if resp.status_code != 200:
                continue

            form.addpart(
                name="images",
                filename=f"image{i}.jpg",
                data=resp.content,
                content_type="image/jpeg",
            )
            added = True

        if not added:
            return

        resp = await session.post(
            url,
            multipart=form,
            headers={
                "Authorization": f"Bearer {self.token}",
            }
        )

        print("Image upload:", resp.status_code, resp.text)

    # =========================
    # DUPLICATE CHECK
    # =========================
    async def is_already_processed(self, property_url: str) -> bool:
        doc = await self.collection.find_one({
            "spitogatos.propertyUrl": property_url,
            "status": "active"
        })
        return doc is not None

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
                    print(f"[PAGE] {url}")

                    html = await self.fetch_html(session, url)

                    if not html:
                        continue

                    property_urls = self.extract_property_urls(html)

                    for property_url in property_urls:
                        if await self.should_stop():
                            print("[CRAWLER] Stopped mid-processing")
                            return total

                        print(f"[PROPERTY] {property_url}")

                        try:
                            if await self.is_already_processed(property_url):
                                print("Skipping duplicate")
                                continue

                            property_html = await self.fetch_html(session, property_url)
                            if not property_html:
                                continue

                            data = self.extract_property_data(property_url, property_html)

                            data = {
                                k: v for k, v in data.items()
                                if v is not None and v != "" and v != "NaN"
                            }

                            mapped = self.map_fields(data)

                            property_id = await self.create_property(session, mapped)
                            if not property_id:
                                continue

                            characteristics = extract_spitogatos_characteristics(base_url, property_url, mapped)
                            await self.collection.update_one(
                                {"_id": ObjectId(property_id)},
                                {
                                    "$set": {
                                        "spitogatos": characteristics
                                    }
                                }
                            )

                            await self.upload_images(session, property_id, data.get("images", []))

                            total += 1

                            await asyncio.sleep(random.uniform(2, 4))

                        except Exception as e:
                            print(f"[ERROR] Property failed: {property_url} → {e}")
                            continue

                    await asyncio.sleep(delay + random.uniform(1, 3))

            except AuthExpiredError as e:
                print(f"[CRAWLER STOPPED] {e}")
                raise e

            except ServerError as e:
                print(f"[SERVER ERROR] {e}")
                raise e

            except Exception as e:
                print(f"[UNEXPECTED ERROR] {e}")
                raise e

        print(f"[DONE] {total} properties")
        return total

def extract_spitogatos_characteristics(url: str, property_url: str, mapping):
    parsed = urlparse(url)
    path = parsed.path.strip("/")
    parts = path.split("/")

    characteristics = {
        "baseUrl": url,
        "propertyUrl": property_url
    }

    try:
        # -------------------------------------
        # Skip language prefix (en, el, etc.)
        # -------------------------------------
        if parts and len(parts[0]) == 2:
            parts = parts[1:]

        # -------------------------------------
        # Extract purpose + category
        # Example: for_sale-homes
        # -------------------------------------
        if len(parts) >= 1:
            first = parts[0].lower()

            if "-" in first:
                transaction_part, category_part = first.split("-", 1)

                # Purpose
                if "sale" in transaction_part:
                    characteristics["purpose"] = "sale"
                elif "rent" in transaction_part:
                    characteristics["purpose"] = "rent"

                # Category (dictionary-based)
                category_map = {
                    "home": "home",
                    "commercial": "commercial",
                    "land": "land",
                    "new-development": "new-development",
                    "student-housing": "student-housing",
                    "other": "other"
                }

                mapped_category = None
                for key, value in category_map.items():
                    if key in category_part:
                        mapped_category = value
                        break

                characteristics["category"] = mapped_category or category_part

        # -------------------------------------
        # Area
        # -------------------------------------
        if len(parts) >= 2:
            characteristics["area"] = parts[1]

        # -------------------------------------
        # Numeric characteristics
        # -------------------------------------
        price = mapping.get("Price")
        surface = mapping.get("Square Meters")

        if price is not None:
            characteristics["price"] = price

        if surface is not None:
            characteristics["surface"] = surface

    except Exception:
        pass

    return characteristics