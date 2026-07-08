"""Build the knowledge corpus from three public sources -> data/corpus.jsonl.

  A. arXiv abstracts (free API, no key).
  B. Patents by CPC domain. Preference order (each skipped cleanly if unconfigured):
       1. SerpApi Google Patents (SERPAPI_API_KEY) — real titles + abstracts.
       2. Google Patents Public Data on BigQuery (GCP_PROJECT) — real abstracts.
       3. USPTO Open Data Portal metadata API (USPTO_API_KEY) — titles only.
  C. 24 hardcoded technical seed abstracts (3 per single defect type).

Each record is {"id", "title", "text"}. All sources are de-duplicated by id and
the combined corpus is capped at 100 documents.

NOTE on USPTO: the brief assumed a POST `/patents/search` endpoint returning
`abstractText`, queried by `cpcSectionSymbol`. The real public ODP key only
serves the GET `/patent/applications/search` endpoint, which exposes
bibliographic metadata (invention title, classification, inventor) but NOT
abstract text, and does not cleanly filter by CPC. We therefore query each CPC
domain by its full-text description and build each patent record from the
available verified metadata. Patents are capped so seeds and arXiv are preserved.
"""
import json
import os
import sys
import time
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

from config import settings

ARXIV_ENDPOINT = "https://export.arxiv.org/api/query"
ARXIV_MAX_RESULTS = 100  # per query (de-duplicated across queries)
ARXIV_QUERIES = [
    "wafer defect root cause analysis semiconductor",
    "semiconductor manufacturing process defect mechanism",
    "chemical mechanical planarization CMP wafer defect",
    "photolithography defect wafer semiconductor",
    "semiconductor yield improvement wafer bin map",
    "wafer map defect pattern classification deep learning",
    "plasma etching semiconductor process defect",
    "thin film deposition semiconductor defect uniformity",
    "semiconductor metrology inspection wafer defect detection",
    "rapid thermal annealing semiconductor wafer process",
    "wafer surface contamination particle semiconductor",
    "semiconductor lithography overlay critical dimension control",
    "wafer edge defect semiconductor manufacturing yield",
    "semiconductor process variation statistical yield analysis",
    "scratch defect wafer handling robotic semiconductor",
    "wafer bin map spatial defect pattern recognition",
    "semiconductor fault detection classification fab equipment",
    "chemical vapor deposition semiconductor film defect",
    "ion implantation semiconductor wafer defect damage",
    "semiconductor wafer cleaning residue defect reduction",
]

USPTO_ENDPOINT = "https://api.uspto.gov/api/v1/patent/applications/search"
# CPC code -> full-text query describing that classification domain. (The ODP key
# cannot filter by CPC directly, so we approximate each domain with a text query.)
USPTO_CPC_QUERIES = {
    "H01L21": "semiconductor device manufacturing process wafer",
    "H01L22": "testing measuring semiconductor manufacture wafer inspection",
    "H01L21/02": "semiconductor wafer surface chemical mechanical treatment",
    "H01L21/67": "apparatus manufacture treatment semiconductor device wafer handling",
    "H01L21/66": "in-line testing measuring semiconductor manufacture defect",
}

MAX_PATENTS = 200  # cap on patent docs (the rest of the budget is arXiv)

# BigQuery (Google Patents Public Data) — preferred patent source when a GCP
# project is configured, because it provides real patent ABSTRACTS that the ODP
# key cannot. CPC families relevant to semiconductor defect / inspection.
BQ_CPC_LIKE = [
    "H01L21/67%",   # apparatus for manufacture/treatment of semiconductor devices
    "H01L22%",      # testing/measuring during semiconductor manufacture
    "H01L21/02%",   # semiconductor surface chemical/mechanical treatment
    "G01N21/95%",   # investigating surface defects (wafers)
    "G01N21/88%",   # investigating presence of flaws/contamination
]
_ATOM = "{http://www.w3.org/2005/Atom}"


# ---------------------------------------------------------------------------
# Source A: arXiv
# ---------------------------------------------------------------------------
def fetch_arxiv() -> list:
    records, seen = [], set()
    for query in ARXIV_QUERIES:
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": ARXIV_MAX_RESULTS,
        }
        try:
            resp = requests.get(ARXIV_ENDPOINT, params=params, timeout=30)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
        except Exception as exc:  # network / parse errors must not crash the build
            print(f"  arXiv query failed ('{query}'): {exc}")
            time.sleep(0.5)
            continue

        for entry in root.findall(f"{_ATOM}entry"):
            arxiv_url = (entry.findtext(f"{_ATOM}id") or "").strip()
            arxiv_id = arxiv_url.rsplit("/", 1)[-1]
            title = " ".join((entry.findtext(f"{_ATOM}title") or "").split())
            abstract = " ".join((entry.findtext(f"{_ATOM}summary") or "").split())
            if not arxiv_id or arxiv_id in seen:
                continue
            if len(abstract.split()) < 50:
                continue
            seen.add(arxiv_id)
            records.append({"id": f"arxiv_{arxiv_id}", "title": title,
                            "text": f"{title}. {abstract}"})
        time.sleep(0.5)
    return records


SERPAPI_ENDPOINT = "https://serpapi.com/search.json"
SERPAPI_PER_PAGE = 100   # max results per SerpApi search page
SERPAPI_MAX_PAGES = 3    # pages per query (keeps total searches small / credit-cheap)
# CPC domain -> Google Patents full-text query (the SerpApi engine has no direct
# CPC filter, so we approximate each domain with a topical query).
SERPAPI_CPC_QUERIES = {
    "H01L22": "semiconductor wafer defect inspection testing during manufacture",
    "H01L21/67": "apparatus manufacture treatment semiconductor wafer handling",
    "H01L21/02": "semiconductor wafer surface chemical mechanical planarization",
    "G01N21/95": "wafer surface defect optical inspection semiconductor",
    "H01L21/66": "in-line testing measuring defect semiconductor manufacture",
    "H01L21/3065": "plasma dry etching semiconductor wafer process",
}
_PATENT_BOILERPLATE = ("cross reference", "this application", "the present application",
                       "related application")


# ---------------------------------------------------------------------------
# Source B (preferred): SerpApi Google Patents — real titles + snippet text.
# ---------------------------------------------------------------------------
def fetch_patents_serpapi():
    """Return patent records via SerpApi, or None if no key. Uses search snippets
    (real patent text) and paginates, so a few searches yield many docs cheaply —
    no per-patent details calls that would burn the SerpApi quota."""
    if not settings.serpapi_api_key:
        return None

    records, seen = [], set()
    for cpc, query in SERPAPI_CPC_QUERIES.items():
        for page in range(SERPAPI_MAX_PAGES):
            params = {"engine": "google_patents", "q": query, "num": SERPAPI_PER_PAGE,
                      "page": page + 1, "api_key": settings.serpapi_api_key}
            try:
                resp = requests.get(SERPAPI_ENDPOINT, params=params, timeout=45)
                resp.raise_for_status()
                payload = resp.json()
            except Exception as exc:
                print(f"  SerpApi search failed (CPC {cpc} p{page + 1}): {exc}")
                time.sleep(1)
                break
            if payload.get("error"):
                print(f"  SerpApi error (CPC {cpc}): {payload['error']}")
                if page == 0 and not records:
                    return None  # bad key / quota on first call -> let caller fall back
                break
            results = payload.get("organic_results", [])
            if not results:
                break
            for item in results:
                pid = item.get("patent_id")
                num = str(item.get("publication_number") or "").strip() or \
                    (pid.split("/")[1] if pid else "")
                title = (item.get("title") or "").strip()
                snippet = (item.get("snippet") or "").strip()
                if not pid or pid in seen or not title:
                    continue
                low = snippet.lower()
                if len(snippet.split()) < 30 or any(low.startswith(b) for b in _PATENT_BOILERPLATE):
                    continue
                seen.add(pid)
                records.append({"id": f"patent_{num}", "title": title,
                                "text": f"{title}. {snippet}"})
                if len(records) >= MAX_PATENTS:
                    print(f"  SerpApi patents: {len(records)}")
                    return records
            time.sleep(1)
    print(f"  SerpApi patents: {len(records)}")
    return records


# ---------------------------------------------------------------------------
# Source B (alt): Google Patents Public Data on BigQuery — real abstracts.
# ---------------------------------------------------------------------------
def _localized_text(field) -> str:
    """Pick the best text from a BigQuery *_localized repeated field."""
    if not field:
        return ""
    for entry in field:
        # rows come back as dict-like; prefer English, else first non-empty.
        lang = (entry.get("language") or "").lower()
        text = (entry.get("text") or "").strip()
        if text and lang in ("en", "", "und"):
            return text
    for entry in field:
        text = (entry.get("text") or "").strip()
        if text:
            return text
    return ""


def fetch_patents_bigquery():
    """Return patent records with real abstracts, or None if BigQuery is unavailable.

    Requires a configured GCP project (settings.gcp_project or GOOGLE_CLOUD_PROJECT)
    and the google-cloud-bigquery SDK with Application Default Credentials.
    """
    project = settings.gcp_project or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        return None
    try:
        from google.cloud import bigquery
    except ImportError:
        print("  google-cloud-bigquery not installed — falling back to ODP patents. "
              "(pip install google-cloud-bigquery db-dtypes)")
        return None

    cpc_clause = " OR ".join(f"cpc_u.code LIKE '{p}'" for p in BQ_CPC_LIKE)
    query = f"""
        SELECT pub.publication_number AS publication_number,
               pub.title_localized AS title_localized,
               pub.abstract_localized AS abstract_localized
        FROM `patents-public-data.patents.publications` AS pub,
             UNNEST(pub.cpc) AS cpc_u
        WHERE pub.country_code = 'US'
          AND pub.publication_date >= 20100101
          AND ({cpc_clause})
        ORDER BY pub.filing_date DESC
        LIMIT 400
    """
    try:
        client = bigquery.Client(project=project)
        rows = client.query(query).result()
    except Exception as exc:
        print(f"  BigQuery query failed: {exc} — falling back to ODP patents.")
        return None

    records, seen = [], set()
    for row in rows:
        num = str(row["publication_number"])
        if num in seen:
            continue
        title = _localized_text(row["title_localized"])
        abstract = _localized_text(row["abstract_localized"])
        if not abstract or len(abstract.split()) < 30:
            continue
        seen.add(num)
        records.append({"id": f"patent_{num}", "title": title,
                        "text": f"{title}. {abstract}" if title else abstract})
        if len(records) >= MAX_PATENTS:
            break
    print(f"  BigQuery patents with abstracts: {len(records)}")
    return records


# ---------------------------------------------------------------------------
# Source B (fallback): USPTO ODP patent metadata (no abstracts available).
# ---------------------------------------------------------------------------
def fetch_patents() -> list:
    if not settings.uspto_api_key:
        print("USPTO_API_KEY not set — skipping patent corpus. "
              "Set it in .env to include patents.")
        return []

    headers = {"X-API-KEY": settings.uspto_api_key, "accept": "application/json"}
    records, seen = [], set()
    for cpc, query in USPTO_CPC_QUERIES.items():
        params = {"q": query, "pagination.offset": 0, "pagination.limit": 50,
                  "sort": "applicationMetaData.filingDate desc"}
        try:
            resp = requests.get(USPTO_ENDPOINT, headers=headers, params=params, timeout=45)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            print(f"  USPTO query failed (CPC {cpc}): {exc}")
            time.sleep(1)
            continue

        for item in payload.get("patentFileWrapperDataBag", []):
            meta = item.get("applicationMetaData", {}) or {}
            num = str(item.get("applicationNumberText") or "").strip()
            title = (meta.get("inventionTitle") or "").strip()
            if not num or not title or num in seen:
                continue
            # The ODP search exposes no abstract; compose a record from verified
            # metadata so the entry is on-topic and citable.
            uspc = meta.get("uspcSymbolText")
            inventor = meta.get("firstInventorName")
            extra = []
            if uspc:
                extra.append(f"US classification {uspc}")
            if inventor:
                extra.append(f"first inventor {inventor}")
            extra.append(f"CPC domain {cpc}")
            text = f"{title}. US patent application {num}; " + "; ".join(extra) + "."
            seen.add(num)
            records.append({"id": f"patent_{num}", "title": title, "text": text})
            if len(records) >= MAX_PATENTS:
                return records
        time.sleep(1)
    return records


# ---------------------------------------------------------------------------
# Source C: hardcoded seed abstracts (3 per single defect type)
# ---------------------------------------------------------------------------
SEED_DOCS = [
    # ---- Center ----
    {"id": "seed_center_001", "title": "Center-clustered wafer defects from spin-coat chuck vacuum contamination",
     "text": "Center-pattern wafer maps exhibit a dense circular cluster of failing dies concentrated at the wafer center. A common root cause is chuck vacuum contamination during photoresist spin coating: particulate or residue accumulated in the vacuum chuck creates a localized backside protrusion that lifts the wafer center, producing a focus and resist-thickness anomaly that prints as a centered defect cluster. The signature is rotationally symmetric and tightly bounded to the inner radius. Engineers should inspect chuck cleanliness, backside particle counts, and vacuum line integrity, and correlate the defect onset with spin-coater preventive-maintenance cycles."},
    {"id": "seed_center_002", "title": "Center non-uniformity from CMP over-polish at the wafer hub",
     "text": "A center defect morphology can originate in chemical mechanical planarization when polish rate is non-uniform and over-polishing concentrates at the wafer center. Carrier head pressure profiles and pad-wafer contact mechanics often produce a center-fast removal signature, thinning films at the hub and creating dishing or residue clearing that fails inner dies. The averaged map shows a smooth circular intensity peak at center. Recommended checks include radial removal-rate mapping, carrier head zone-pressure calibration, retaining-ring wear, and slurry flow distribution toward the wafer center."},
    {"id": "seed_center_003", "title": "Center thermal-gradient defects in rapid thermal annealing",
     "text": "Center-pattern defects also arise from thermal-gradient non-uniformity during rapid thermal annealing. When lamp zoning or wafer-rotation control allows the wafer center to reach a different peak temperature than the edge, dopant activation, silicide formation, or film stress vary radially and fail a centered die population. The pattern is concentric and reproducible across lots sharing the same RTA chamber. Investigation should cover lamp-bank calibration, pyrometer emissivity correction, susceptor condition, and center-to-edge temperature uniformity measured with instrumented monitor wafers."},
    # ---- Donut ----
    {"id": "seed_donut_001", "title": "Donut ring defects from edge-bead removal chemistry irregularity",
     "text": "Donut-pattern wafer maps show an annular ring of failing dies surrounding a clean center. One root cause is edge-bead removal chemistry concentration irregularity: when EBR solvent dispense rate or nozzle position drifts, an intermediate-radius band of resist is partially stripped or redeposited, creating a ring of process failures. The defect forms a closed annulus at a fixed radius. Engineers should audit EBR nozzle alignment, solvent dispense volume and timing, and back-rinse pressure, and compare the failing radius against the programmed EBR width."},
    {"id": "seed_donut_002", "title": "Annular resist banding from spin-coat centrifugal instability",
     "text": "Donut morphology can result from photoresist spin-coat centrifugal instability that produces annular thick/thin bands. Resonance in the spin acceleration profile or solvent evaporation front instabilities generate concentric thickness variation, and one band falls outside the lithographic process window, printing as a ring of defects. The pattern is concentric and radius-stable within a coater. Recommended actions: review spin ramp and final-spin speed, dispense temperature, exhaust airflow over the bowl, and resist viscosity, and map resist thickness radially with a reflectometer."},
    {"id": "seed_donut_003", "title": "Ring-shaped under-development from developer drain patterns",
     "text": "A donut defect can be caused by developer drain patterns that create a ring-shaped under-development zone. During puddle development, non-uniform developer drainage or surface-tension pinning leaves an annular region with insufficient development time, so resist is incompletely cleared in a ring and subsequent etch or implant fails there. The defect is a concentric annulus correlated with develop-module dispense geometry. Investigate developer dispense arm sweep, puddle dwell time, nozzle drip, and wafer rotation during develop, and check for ring-shaped resist residue under the microscope."},
    # ---- Edge-Loc ----
    {"id": "seed_edgeloc_001", "title": "Edge-localized defects from lithography focus offset due to wafer bow",
     "text": "Edge-Loc wafer maps show failing dies clustered along one localized portion of the wafer edge rather than the full ring. A frequent root cause is photolithography edge-die focus offset arising from wafer bow or tilt: warped wafers or chuck flatness errors push edge dies out of the depth of focus on one side, degrading pattern fidelity locally. The defect is an arc-shaped cluster at the periphery. Engineers should measure wafer flatness and bow, verify exposure-chuck pin cleanliness, and review edge-die focus-offset and leveling compensation in the scanner job."},
    {"id": "seed_edgeloc_002", "title": "Edge contamination from cassette slot mechanical contact",
     "text": "Edge-Loc defects can stem from edge-ring contamination caused by mechanical contact between the wafer edge and a cassette or FOUP slot. Repeated rubbing at a fixed handling orientation deposits particles or causes micro-damage on one edge segment, failing a localized arc of edge dies. The pattern recurs at the same angular position relative to the wafer notch. Recommended checks: inspect cassette and end-effector contact points for wear and residue, audit robot teach positions, and correlate the failing arc angle with wafer-handling orientation."},
    {"id": "seed_edgeloc_003", "title": "Localized edge etch-rate drop from gas-flow shadowing",
     "text": "An Edge-Loc pattern may originate from etch gas-flow shadowing near the wafer edge that causes a localized etch-rate drop. Asymmetry in showerhead flow, focus-ring erosion, or chamber exhaust conductance starves one edge region of reactant, leaving under-etched or residue-bearing dies in a peripheral arc. The defect is angularly localized and chamber-specific. Investigation should include focus-ring and edge-ring wear, gas-flow and pressure symmetry, RF coupling at the edge, and chamber-matching data across tools."},
    # ---- Edge-Ring ----
    {"id": "seed_edgering_001", "title": "Edge-Ring defects from CMP polishing pressure non-uniformity",
     "text": "Edge-Ring wafer maps show failing dies forming a complete ring around the outer edge. A primary root cause is CMP polishing pad pressure non-uniformity at the retaining-ring interface: edge-fast or edge-slow removal produced by retaining-ring pressure and wear creates a full annular band of film thickness error that fails all edge dies symmetrically. The pattern is a closed ring at maximum radius. Engineers should calibrate retaining-ring pressure, inspect ring wear and seating, map edge removal rate, and review carrier-head edge-zone control."},
    {"id": "seed_edgering_002", "title": "Edge-bead exposure failure producing a full defect ring",
     "text": "A full Edge-Ring can be caused by photoresist edge-bead buildup leading to ring-shaped exposure failure. Excess resist at the wafer perimeter, if not fully removed, defocuses or blocks exposure around the entire edge, failing the outermost die ring uniformly. The signature is a complete, narrow annulus at the wafer boundary. Recommended actions: verify edge-bead removal width and completeness, check resist thickness at the extreme edge, and confirm exposure dose and focus for the peripheral field shots."},
    {"id": "seed_edgering_003", "title": "Contamination ring from wafer-carrier seal-ring deposition",
     "text": "Edge-Ring defects may arise from a wafer carrier or chamber seal ring depositing a contamination ring at the wafer periphery. Outgassing, polymer flaking, or condensate from a seal ring contacts the edge symmetrically, leaving a ring of particle or film-composition failures. The defect is a uniform closed ring tied to a specific chamber's seal hardware. Investigation should cover seal-ring condition and material, edge particle adders per chamber, clamp or e-chuck edge contact, and preventive-maintenance history of the affected tool."},
    # ---- Local ----
    {"id": "seed_local_001", "title": "Local defect clusters from airborne particle deposition",
     "text": "Local wafer maps show a small, compact cluster of failing dies confined to one region without edge or center symmetry. A common cause is airborne particle deposition during open process steps: a settling particle or droplet contaminates a small die group, blocking exposure, etch, or deposition locally. The cluster position is random from wafer to wafer. Engineers should review cleanroom particle counts and airflow near the implicated tool, check open-cassette dwell time, and perform particle-per-wafer-pass monitoring to localize the contributing step."},
    {"id": "seed_local_002", "title": "Local etch non-uniformity from gas nozzle blockage",
     "text": "A Local defect can result from localized etch non-uniformity caused by a partially blocked gas nozzle or showerhead hole. The blockage creates a small region of reactant starvation or excess, producing a compact patch of under- or over-etched dies. The defect maps to a fixed spatial location corresponding to the obstructed nozzle. Recommended checks: inspect and clean showerhead holes, measure spatial etch-rate uniformity with a patterned monitor, and correlate the defect location with the chamber gas-injection geometry."},
    {"id": "seed_local_003", "title": "Point-source contamination from equipment in a specific die region",
     "text": "Local patterns also arise from point-source equipment contamination affecting a specific die region, such as a dripping fitting, a worn lift-pin, or a flaking chamber surface contacting one area. The result is a repeatable small cluster of failures at a fixed wafer position across many wafers in a tool. The defect is compact and tool-specific. Investigation should map the cluster to chamber hardware coordinates, inspect lift pins and nearby surfaces for residue or damage, and trace the affected lots back to a common tool and slot."},
    # ---- Near-Full ----
    {"id": "seed_nearfull_001", "title": "Near-Full wafer failure from wet-etch bath contamination or depletion",
     "text": "Near-Full wafer maps show defects covering nearly the entire wafer surface. A leading cause is bulk wet-etch chemical bath contamination or depletion: when an immersion bath drifts out of concentration or is contaminated, the whole wafer is uniformly over- or under-etched, failing almost all dies simultaneously. The pattern is global with little spatial structure. Engineers should check bath chemistry titration, replacement and bleed-and-feed schedules, metallic contamination monitors, and whether the failure spans whole lots processed in the same bath."},
    {"id": "seed_nearfull_002", "title": "Full-wafer deposition failure from chamber atmospheric contamination",
     "text": "A Near-Full pattern can result from deposition-chamber atmospheric contamination affecting the full wafer, such as a vacuum leak, moisture ingress, or precursor delivery fault. The contaminant alters film composition or adhesion across the entire wafer, producing near-complete die failure. The defect is global and often appears abruptly after a maintenance or gas-panel event. Investigation should include base-pressure and leak-rate checks, residual-gas analysis, precursor and carrier-gas purity, and correlation of onset with chamber PM or gas-cylinder changes."},
    {"id": "seed_nearfull_003", "title": "Catastrophic in-line equipment failure causing near-full loss",
     "text": "Near-Full defects may indicate a catastrophic equipment failure during an in-line process step, for example a sudden loss of temperature control, RF trip, or wafer mishandling that exposes the whole wafer to an out-of-spec condition. The entire wafer fails or nearly so, frequently for a single wafer or a short burst. The pattern is global and time-correlated with a tool fault. Engineers should review tool alarm and event logs, interlock trips, and sensor traces around the affected timestamp to identify the failure event."},
    # ---- Random ----
    {"id": "seed_random_001", "title": "Random scattered defects from cleanroom airborne particle events",
     "text": "Random wafer maps show failing dies scattered without spatial structure across the whole wafer. A typical cause is random cleanroom airborne particle events: sporadic particles from personnel, tool generation, or filtration excursions land at uncorrelated positions, each failing isolated dies. The defect density tracks ambient particle levels rather than any radial pattern. Engineers should monitor airborne particle counters, HEPA/ULPA filter integrity, gowning discipline, and tool-generated particle adders, and use spatial randomness tests to distinguish this from structured signatures."},
    {"id": "seed_random_002", "title": "Random die failures from ESD events during robotic transfer",
     "text": "Random patterns can be produced by electrostatic discharge events during robotic wafer transfer. Triboelectric charging and uncontrolled discharge zap dies at unpredictable locations, leaving scattered, uncorrelated failures. The defect shows no radial or angular structure and may correlate with low-humidity conditions or ungrounded handling hardware. Recommended checks: verify ionizer function and placement, measure wafer and chuck charging, audit grounding of end-effectors and load ports, and monitor cleanroom humidity against ESD limits."},
    {"id": "seed_random_003", "title": "Random defect nucleation from CMP micro-scratch events",
     "text": "Random defects may originate from micro-scratch nucleation during chemical mechanical planarization, where stray agglomerated slurry particles or pad debris create tiny, randomly placed scratches that fail scattered dies. Unlike a single long scratch, these are dispersed point events without linear structure. The density correlates with slurry quality and pad condition. Investigation should cover slurry filtration and agglomeration, pad conditioning and debris, point-of-use filter integrity, and post-CMP brush-clean effectiveness."},
    # ---- Scratch ----
    {"id": "seed_scratch_001", "title": "Linear scratch defects from robotic end-effector contact",
     "text": "Scratch wafer maps show a thin linear or curved track of failing dies crossing the wafer. A frequent cause is mechanical contact from a robotic end-effector during wafer transfer: a misaligned or contaminated blade drags across the surface, creating a line of damaged dies. The defect is narrow, elongated, and oriented consistently with the handling motion. Engineers should inspect end-effector pads and alignment, review robot teach positions and wafer slippage, and correlate scratch orientation with the transfer path."},
    {"id": "seed_scratch_002", "title": "Linear scratches from CMP pad-conditioning disk drag",
     "text": "A Scratch pattern can be caused by a CMP pad-conditioning disk dragging across the wafer, where dislodged diamond grit or an improperly retracted conditioner gouges a linear track. The result is a long, narrow scratch failing dies along a line. The defect is linear and often radial or arc-shaped relative to the polish kinematics. Recommended actions: inspect conditioner disk grit retention and sweep program, check for embedded particles in the pad, and review conditioning pressure and end-of-life status."},
    {"id": "seed_scratch_003", "title": "Edge-contact scratches from cassette slot during load and unload",
     "text": "Scratch defects may arise from wafer cassette slot edge contact during load and unload, where the wafer edge or surface rubs a slot rail and a linear damage track propagates inward. The scratch typically starts at the edge and extends along the insertion direction, failing a line of dies. The orientation is consistent with the cassette geometry. Investigation should cover slot rail wear and debris, mapper and elevator alignment, wafer seating, and whether the scratch entry point matches the handling reference."},
]


# ---------------------------------------------------------------------------
def main():
    os.makedirs(os.path.dirname(settings.corpus_jsonl_path) or ".", exist_ok=True)

    print("Fetching arXiv abstracts...")
    arxiv_docs = fetch_arxiv()
    print(f"  arXiv: {len(arxiv_docs)}")

    print("Fetching patents...")
    # Preference order: SerpApi (real abstracts) -> BigQuery -> USPTO ODP metadata.
    patent_docs = fetch_patents_serpapi()
    if patent_docs is None:
        patent_docs = fetch_patents_bigquery()
    if patent_docs is None:
        patent_docs = fetch_patents()
    print(f"  patents: {len(patent_docs)}")

    seed_docs = SEED_DOCS
    print(f"  seed: {len(seed_docs)}")

    # Combine and de-duplicate by id; seeds + patents first so they survive the cap.
    combined, seen = [], set()
    for doc in seed_docs + patent_docs + arxiv_docs:
        if doc["id"] in seen:
            continue
        seen.add(doc["id"])
        combined.append(doc)
        if len(combined) >= settings.corpus_max_docs:
            break

    with open(settings.corpus_jsonl_path, "w", encoding="utf-8") as fh:
        for doc in combined:
            fh.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"\nCounts -> arXiv {len(arxiv_docs)} | patents {len(patent_docs)} | "
          f"seed {len(seed_docs)} | total written {len(combined)} "
          f"(-> {settings.corpus_jsonl_path})")


if __name__ == "__main__":
    main()
