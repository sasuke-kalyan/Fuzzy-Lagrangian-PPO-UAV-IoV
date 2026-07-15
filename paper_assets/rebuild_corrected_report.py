from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from lxml import etree

SOURCE = Path("thisistoedit.docx")
OUTPUT = Path("thisistoedit_corrected_ready.docx")
FIGDIR = Path("paper_assets/revised_report_figures")

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
PR = "http://schemas.openxmlformats.org/package/2006/relationships"
NS = {"w": W, "a": A, "r": R, "wp": WP, "pr": PR}


def ptext(p):
    return "".join(t.text or "" for t in p.xpath(".//w:t", namespaces=NS)).strip()


def set_ptext(p, value):
    nodes = p.xpath(".//w:t", namespaces=NS)
    if not nodes:
        run = etree.SubElement(p, f"{{{W}}}r")
        node = etree.SubElement(run, f"{{{W}}}t")
        node.text = value
        return
    nodes[0].text = value
    for node in nodes[1:]:
        node.text = ""


with ZipFile(SOURCE) as zin:
    files = {name: zin.read(name) for name in zin.namelist()}

files["word/media/image1.png"] = (FIGDIR / "figure1_corrected_architecture.png").read_bytes()
files["word/media/image5.png"] = (FIGDIR / "figure4_corrected_screening.png").read_bytes()
files["word/media/image6.png"] = (FIGDIR / "figure5_fuzzy_risk_priority.png").read_bytes()

doc = etree.fromstring(files["word/document.xml"])
rels = etree.fromstring(files["word/_rels/document.xml.rels"])
targets = {rel.get("Id"): rel.get("Target") for rel in rels}
body = doc.find(f"{{{W}}}body")

remove_media = {
    "media/44e5046e44d7a6be0f22b784510fa6a2e11fa119.jpg",
    "media/image2.png",
    "media/image4.png",
}

for p in list(body):
    if p.tag != f"{{{W}}}p":
        continue
    rel_ids = [b.get(f"{{{R}}}embed") for b in p.xpath(".//a:blip", namespaces=NS)]
    if any(targets.get(rid) in remove_media for rid in rel_ids):
        body.remove(p)

remove_captions = {
    "Fig. 1 System model of the proposed Fuzzy LP-PPO framework for reliable UAV selection in dynamic IoV scenarios.",
    "Figure 3. Fuzzy decision engine and PPO-based UAV ranking process.",
    "Figure 2. Dynamic UAV-IoV operating scenarios used for evaluation.",
}
for p in list(body):
    if p.tag == f"{{{W}}}p" and ptext(p) in remove_captions:
        body.remove(p)

exact_replacements = {
    "Figure 1. Overall Fuzzy LP-PPO framework for reliable UAV selection in dynamic IoV scenarios.":
        "Figure 1. Implementation-faithful Fuzzy LP-PPO architecture for reliable UAV selection in dynamic IoV scenarios.",
    "Figure 4. Proposed UAV communication and reliable link selection architecture.":
        "Figure 2. Illustrative operating scenarios used to evaluate Fuzzy LP-PPO: urban canyon, suburban crossroads, and emergency response.",
    "Fig. 2 UAV-IoV topology and communication-selection process of the proposed Fuzzy LP-PPO framework.":
        "Figure 3. UAV-IoV topology and communication-selection process of the proposed Fuzzy LP-PPO framework.",
    "Fig. 5 Candidate UAV evaluation process.":
        "Figure 4. Hard feasibility screening, fuzzy scoring, candidate ranking, and PPO ranked-action mapping.",
    "Fig. 6 Fuzzy logic inference process for UAV prioritization.":
        "Figure 5. Implementation-faithful fuzzy risk-priority calculation used for candidate ranking and reward shaping.",
}

paragraph_one = (
    "Figure 1 presents the complete implementation-faithful decision loop. For every candidate link, the environment computes "
    "distance, signal quality, delay, PDR, and residual energy. Hard screening removes UAVs that violate the direct-selection "
    "conditions for distance, delay, signal, or energy; if none passes, the full fleet is retained. Normalized delay, PDR, signal, "
    "and energy risks, together with the emergency indicator, form a weighted fuzzy-priority value that contributes to the composite "
    "candidate score and reward. PPO selects a position in the ranked feasible list through its actor, while the critic supports value "
    "estimation during training. After communication, observed QoS violations update the non-negative Lagrangian multipliers and the "
    "resulting adaptive penalty contributes to subsequent policy learning."
)

for p in body.xpath("./w:p", namespaces=NS):
    text = ptext(p)
    if text in exact_replacements:
        set_ptext(p, exact_replacements[text])
    elif text.startswith("Figure 1 illustrates the UAV selection problem considered in this study."):
        set_ptext(p, paragraph_one)
    elif text.startswith("At every communication decision instant") and "Figure 2 illustrates the communication topology" in text:
        set_ptext(p, text.replace("Figure 2 illustrates", "Figure 3 illustrates"))

# Keep journal captions below their corresponding artwork.
for target, caption_start in (("media/image5.png", "Figure 4."), ("media/image6.png", "Figure 5.")):
    image_p = None
    caption_p = None
    for p in body.xpath("./w:p", namespaces=NS):
        blips = p.xpath(".//a:blip", namespaces=NS)
        if blips and targets.get(blips[0].get(f"{{{R}}}embed")) == target:
            image_p = p
        if ptext(p).startswith(caption_start):
            caption_p = p
    if image_p is not None and caption_p is not None:
        caption_p.getparent().remove(caption_p)
        image_p.addnext(caption_p)

# Resize replacement figures to retain their natural aspect ratios at journal-page width.
figure_targets = {
    "media/image1.png": (1800, 1000, 5943600),
    "media/image5.png": (1450, 980, 5486400),
    "media/image6.png": (1600, 980, 5943600),
}
for p in body.xpath("./w:p", namespaces=NS):
    blips = p.xpath(".//a:blip", namespaces=NS)
    if not blips:
        continue
    target = targets.get(blips[0].get(f"{{{R}}}embed"))
    if target not in figure_targets:
        continue
    pxw, pxh, cx = figure_targets[target]
    cy = round(cx * pxh / pxw)
    for extent in p.xpath(".//wp:extent", namespaces=NS):
        extent.set("cx", str(cx)); extent.set("cy", str(cy))
    for extent in p.xpath(".//a:xfrm/a:ext", namespaces=NS):
        extent.set("cx", str(cx)); extent.set("cy", str(cy))

files["word/document.xml"] = etree.tostring(doc, xml_declaration=True, encoding="UTF-8", standalone="yes")

with ZipFile(OUTPUT, "w", ZIP_DEFLATED) as zout:
    for name, data in files.items():
        zout.writestr(name, data)

print(OUTPUT)
