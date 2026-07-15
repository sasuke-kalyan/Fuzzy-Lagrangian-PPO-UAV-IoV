from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from lxml import etree
from PIL import Image

SOURCE = Path("thisistoedit_corrected_ready_regenerated_figures.docx")
OUTPUT = Path("thisistoedit_corrected_regenerated_fully_correct.docx")

REPLACEMENTS = {
    "media/image1.png": Path("paper_assets/revised_report_figures/figure1_architecture_text_corrected.png"),
    "media/image5.png": Path("paper_assets/revised_report_figures/figure4_screening_text_corrected.png"),
    "media/image6.png": Path("/home/shubh-om/Downloads/ChatGPT Image Jul 15, 2026, 12_00_47 AM.png"),
}

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
PR = "http://schemas.openxmlformats.org/package/2006/relationships"
NS = {"w": W, "a": A, "r": R, "wp": WP, "pr": PR}

with ZipFile(SOURCE) as zin:
    files = {name: zin.read(name) for name in zin.namelist()}

for target, source in REPLACEMENTS.items():
    files[f"word/{target}"] = source.read_bytes()

doc = etree.fromstring(files["word/document.xml"])
rels = etree.fromstring(files["word/_rels/document.xml.rels"])
targets = {rel.get("Id"): rel.get("Target") for rel in rels}

# Use near-full journal-page width and retain each regenerated image's natural ratio.
display_widths = {
    "media/image1.png": 5943600,
    "media/image5.png": 5486400,
    "media/image6.png": 5943600,
}

for p in doc.xpath(".//w:body/w:p", namespaces=NS):
    blips = p.xpath(".//a:blip", namespaces=NS)
    if not blips:
        continue
    target = targets.get(blips[0].get(f"{{{R}}}embed"))
    if target not in REPLACEMENTS:
        continue
    with Image.open(REPLACEMENTS[target]) as im:
        pxw, pxh = im.size
    cx = display_widths[target]
    cy = round(cx * pxh / pxw)
    for extent in p.xpath(".//wp:extent", namespaces=NS):
        extent.set("cx", str(cx))
        extent.set("cy", str(cy))
    for extent in p.xpath(".//a:xfrm/a:ext", namespaces=NS):
        extent.set("cx", str(cx))
        extent.set("cy", str(cy))

files["word/document.xml"] = etree.tostring(
    doc, xml_declaration=True, encoding="UTF-8", standalone="yes"
)

with ZipFile(OUTPUT, "w", ZIP_DEFLATED) as zout:
    for name, data in files.items():
        zout.writestr(name, data)

print(OUTPUT)
