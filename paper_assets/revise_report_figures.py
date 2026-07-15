from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT = Path("paper_assets/revised_report_figures")
OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#17365D"
BLUE = "#2F6FBA"
CYAN = "#DFF3FA"
PALE = "#F5F9FD"
TEAL = "#159A9C"
GREEN = "#E4F4EC"
ORANGE = "#F3A34A"
RED = "#C84747"
PURPLE = "#7557A8"
GRAY = "#66788A"
WHITE = "#FFFFFF"


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def center(draw, box, text, fnt, fill=NAVY):
    x1, y1, x2, y2 = box
    bb = draw.multiline_textbbox((0, 0), text, font=fnt, align="center", spacing=5)
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    draw.multiline_text(((x1 + x2 - w) / 2, (y1 + y2 - h) / 2), text,
                        font=fnt, fill=fill, align="center", spacing=5)


def box(draw, xy, text, fill=PALE, outline=BLUE, width=3, radius=18, fs=27, bold=False):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)
    center(draw, xy, text, font(fs, bold))


def arrow(draw, start, end, fill=NAVY, width=4, head=13):
    draw.line([start, end], fill=fill, width=width)
    x2, y2 = end
    x1, y1 = start
    import math
    ang = math.atan2(y2-y1, x2-x1)
    a1, a2 = ang + 2.55, ang - 2.55
    pts = [(x2, y2), (x2 + head*math.cos(a1), y2 + head*math.sin(a1)),
           (x2 + head*math.cos(a2), y2 + head*math.sin(a2))]
    draw.polygon(pts, fill=fill)


def drone(draw, cx, cy, scale=1.0, color=BLUE):
    r = int(13*scale)
    draw.rounded_rectangle((cx-26*scale, cy-12*scale, cx+26*scale, cy+12*scale),
                           radius=int(8*scale), fill=WHITE, outline=color, width=max(2, int(3*scale)))
    for dx, dy in [(-34,-23),(34,-23),(-34,23),(34,23)]:
        draw.line((cx, cy, cx+dx*scale, cy+dy*scale), fill=color, width=max(2,int(3*scale)))
        draw.ellipse((cx+(dx-13)*scale, cy+(dy-5)*scale, cx+(dx+13)*scale, cy+(dy+5)*scale),
                     outline=color, width=max(2,int(3*scale)))
    draw.ellipse((cx-r, cy-r, cx+r, cy+r), fill=CYAN, outline=color, width=max(2,int(2*scale)))


def architecture():
    im = Image.new("RGB", (1800, 1000), WHITE)
    d = ImageDraw.Draw(im)
    center(d, (20, 20, 1780, 85), "Fuzzy LP-PPO Architecture for Reliable UAV Selection", font(39, True))

    # Scenario/environment column
    d.rounded_rectangle((35, 110, 350, 900), radius=24, fill="#F7FAFD", outline=BLUE, width=3)
    center(d, (45, 120, 340, 175), "Dynamic IoV\nEnvironment", font(30, True))
    scenarios = [("Urban Canyon", 225), ("Suburban Crossroads", 365), ("Emergency Response", 505)]
    for label, y in scenarios:
        box(d, (70, y, 315, y+90), label, fill=CYAN, fs=19, bold=True)
    box(d, (70, 660, 315, 740), "Moving Vehicle\n+ Eight UAVs", fill=GREEN, outline=TEAL, fs=23)
    box(d, (70, 785, 315, 855), "Next State  s(t+1)", fill="#ECEAF7", outline=PURPLE, fs=23)

    # Measurement and ranking
    d.rounded_rectangle((395, 110, 905, 900), radius=24, fill="#FBFCFE", outline=BLUE, width=3)
    center(d, (410, 120, 890, 165), "Candidate Evaluation and Ranking", font(25, True))
    labels = [
        "Link Measurements\nDistance  •  Signal  •  Delay  •  PDR  •  Energy",
        "Hard Feasibility Screening\nDistance, delay, signal and energy",
        "Risk Components\nDelay • PDR • Signal • Energy • Emergency boost",
        "Weighted Fuzzy Priority  F(v,u)",
        "Composite Candidate Score",
        "Ranked Feasible UAV List",
    ]
    ys = [190, 300, 420, 555, 665, 775]
    heights = [80, 90, 100, 75, 75, 75]
    fills = [CYAN, "#FFF3DF", "#F1ECFA", "#F1ECFA", GREEN, GREEN]
    outlines = [BLUE, ORANGE, PURPLE, PURPLE, TEAL, TEAL]
    for i, (lab, y, h) in enumerate(zip(labels, ys, heights)):
        box(d, (430, y, 870, y+h), lab, fill=fills[i], outline=outlines[i], fs=18 if i == 2 else 21, bold=i in (1,3,5))
        if i < len(labels)-1:
            arrow(d, (650, y+h), (650, ys[i+1]-8))
    arrow(d, (350, 505), (420, 505))

    # PPO and feedback
    d.rounded_rectangle((950, 110, 1760, 900), radius=24, fill="#FBFCFE", outline=BLUE, width=3)
    center(d, (965, 120, 1745, 165), "PPO Decision and Constraint-Aware Learning", font(29, True))
    box(d, (995, 200, 1305, 320), "PPO Actor\nPolicy  πθ(a|s)", fill=CYAN, fs=25, bold=True)
    box(d, (1400, 200, 1710, 320), "PPO Critic\nValue  Vφ(s)", fill=CYAN, fs=25, bold=True)
    box(d, (1065, 390, 1640, 480), "Selected UAV and\nCommunication Outcome", fill=GREEN, outline=TEAL, fs=22, bold=True)
    box(d, (1065, 540, 1350, 640), "Reward\nQoS + task utility", fill="#FFF3DF", outline=ORANGE, fs=23)
    box(d, (1450, 540, 1710, 640), "Observed QoS\nviolations", fill="#FBEAEA", outline=RED, fs=23)
    box(d, (1065, 710, 1350, 815), "PPO Update\nActor + Critic", fill=CYAN, fs=23, bold=True)
    box(d, (1450, 710, 1710, 815), "Lagrangian Update\nλ ← clip(λ + ηv)", fill="#FBEAEA", outline=RED, fs=22, bold=True)
    arrow(d, (870, 812), (985, 260))
    arrow(d, (1305, 260), (1390, 260), fill=GRAY)
    arrow(d, (1510, 320), (1510, 380))
    arrow(d, (1300, 320), (1300, 380))
    arrow(d, (1350, 435), (1630, 435))
    arrow(d, (1350, 435), (1210, 530))
    arrow(d, (1500, 480), (1580, 530))
    arrow(d, (1210, 640), (1210, 700))
    arrow(d, (1580, 640), (1580, 700))
    arrow(d, (1450, 760), (1360, 760), fill=RED)
    d.text((1365, 720), "adaptive penalty", font=font(18), fill=RED)
    # environment return loop
    d.line((1210,815,1210,920,190,920,190,865), fill=NAVY, width=4)
    arrow(d, (190,920), (190,865))
    d.text((690, 928), "policy/environment feedback loop", font=font(21, True), fill=NAVY)
    im.save(OUT / "figure1_corrected_architecture.png", dpi=(300,300))


def screening():
    im = Image.new("RGB", (1450, 980), WHITE)
    d = ImageDraw.Draw(im)
    center(d, (30, 20, 1420, 80), "Candidate UAV Screening and Ranked-Action Mapping", font(38, True))
    steps = [
        ("1", "Communication request", "One representative active vehicle"),
        ("2", "Evaluate all eight UAVs", "Compute distance, signal, delay, PDR and residual energy"),
        ("3", "Hard feasibility screening", "Distance ≤ 500 m  •  Energy ≥ 20%  •  Signal ≥ 0.10  •  Delay ≤ 95 ms"),
        ("4", "Fallback rule", "If no UAV passes, retain all eight candidates"),
        ("5", "Fuzzy risk and composite score", "Use delay, PDR, signal, energy and emergency boost; then add QoS and distance terms"),
        ("6", "Rank feasible candidates", "Sort by composite score, energy, signal and lower delay"),
        ("7", "Map PPO action to ranked position", "a ∈ {0,…,7} selects an available ranked candidate"),
    ]
    y = 105
    for i,(n,title,sub) in enumerate(steps):
        fill = CYAN if i<2 else ("#FFF3DF" if i in (2,3) else ("#F1ECFA" if i==4 else GREEN))
        outline = BLUE if i<2 else (ORANGE if i in (2,3) else (PURPLE if i==4 else TEAL))
        d.ellipse((75,y+22,130,y+77), fill=outline)
        center(d,(75,y+22,130,y+77),n,font(25,True),WHITE)
        d.rounded_rectangle((155,y,1375,y+100),radius=18,fill=fill,outline=outline,width=3)
        d.text((190,y+16),title,font=font(26,True),fill=NAVY)
        d.text((190,y+56),sub,font=font(20),fill=GRAY)
        if i<len(steps)-1: arrow(d,(765,y+100),(765,y+120),fill=NAVY,width=3,head=10)
        y += 120
    d.rounded_rectangle((220,925,1230,970),radius=15,fill=PALE,outline=BLUE,width=2)
    center(d,(220,925,1230,970),"PDR is evaluated in fuzzy scoring and Lagrangian penalties; it is not part of the hard mask.",font(19,True))
    im.save(OUT / "figure4_corrected_screening.png", dpi=(300,300))


def fuzzy_risk():
    im = Image.new("RGB", (1600, 980), WHITE)
    d = ImageDraw.Draw(im)
    center(d,(20,20,1580,80),"Implementation-Faithful Fuzzy Risk Priority",font(39,True))
    # Inputs and formula cards
    inputs = [
        ("Delay urgency", "μD = clip((D − 55)/45)", ORANGE),
        ("PDR risk", "μP = clip((55 − PDR)/7)", BLUE),
        ("Signal risk", "μS = clip((0.18 − S)/0.13)", TEAL),
        ("Energy risk", "μE = clip((35 − E%)/35)", PURPLE),
        ("Emergency boost", "Iemg ∈ {0,1}", RED),
    ]
    y=130
    for title,eq,c in inputs:
        d.rounded_rectangle((55,y,475,y+110),radius=18,fill=PALE,outline=c,width=3)
        d.text((85,y+16),title,font=font(24,True),fill=NAVY)
        d.text((85,y+60),eq,font=font(22),fill=c)
        arrow(d,(475,y+55),(610,490),fill=c,width=3,head=11)
        y+=145
    # aggregation
    d.rounded_rectangle((620,330,1120,650),radius=25,fill="#F1ECFA",outline=PURPLE,width=4)
    center(d,(640,350,1100,405),"Weighted clipped aggregation",font(30,True))
    formula = "F(v,u) = clip[\n0.35 μD + 0.25 μP + 0.25 μS\n+ 0.15 μE + 0.10 Iemg\n]"
    center(d,(660,420,1080,570),formula,font(26),PURPLE)
    center(d,(650,585,1090,635),"Higher F means greater risk/urgency",font(21,True),RED)
    # outputs
    arrow(d,(1120,490),(1220,490),fill=NAVY,width=4,head=14)
    box(d,(1230,385,1540,595),"Risk-aware\nPriority Score\n0 ≤ F(v,u) ≤ 1",fill=GREEN,outline=TEAL,fs=29,bold=True)
    # downstream roles
    box(d,(730,740,1070,860),"Candidate Ranking\n+35 × F(v,u)",fill=CYAN,outline=BLUE,fs=25,bold=True)
    box(d,(1190,740,1530,860),"Reward Shaping\n+20 × F(v,u)",fill="#FFF3DF",outline=ORANGE,fs=25,bold=True)
    arrow(d,(1385,595),(900,730),fill=TEAL,width=3,head=12)
    arrow(d,(1385,595),(1360,730),fill=TEAL,width=3,head=12)
    box(d,(55,845,560,930),"Distance is handled by hard screening\nand the composite candidate score.",fill=PALE,outline=GRAY,fs=21)
    center(d,(585,885,1570,955),"No Mamdani rule base or centroid defuzzification\nis used in the current implementation.",font(18,True),GRAY)
    im.save(OUT / "figure5_fuzzy_risk_priority.png", dpi=(300,300))


if __name__ == "__main__":
    architecture()
    screening()
    fuzzy_risk()
    print(OUT)
