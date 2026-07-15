#!/usr/bin/env python3
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).parent / "proposed_methodology_package" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, PALE, GREEN, ORANGE, DARK = "#24557A", "#EAF3F8", "#DDF0E4", "#FCE8CE", "#183B56"

def box(ax, x, y, w, h, text, color=PALE, fs=9):
    p = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.018,rounding_size=0.02",
                       facecolor=color, edgecolor=BLUE, linewidth=1.6)
    ax.add_patch(p); ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=DARK, wrap=True)
    return p

def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, color=BLUE, linewidth=1.5))

def finish(fig, ax, name):
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off"); fig.tight_layout(); fig.savefig(OUT/name, dpi=300, bbox_inches="tight"); plt.close(fig)

# Figure 2: architecture. Parallel modules are separated to prevent overlap.
fig, ax = plt.subplots(figsize=(8.2, 10.0))
box(ax,.50,.95,.82,.072,"Dynamic UAV-Assisted IoV Environment\nVehicle/UAV positions, energy and scenario",GREEN,9.4)
box(ax,.50,.82,.82,.072,"Communication Measurement Module\nDistance, signal quality, delay, PDR and energy",PALE,9.2)

box(ax,.285,.68,.37,.082,"Hard Feasibility Mask\nRange, signal, delay and energy checks",PALE,8.8)
box(ax,.715,.68,.37,.082,"Fuzzy Priority Estimator\nDelay, PDR, signal, energy and emergency",ORANGE,8.8)

box(ax,.50,.52,.82,.072,"Candidate Scoring and Ranking\nOrdered feasible UAV list",PALE,9.2)
box(ax,.50,.37,.82,.072,"PPO Actor–Critic\nActor: ranked action | Critic: state value",PALE,9.2)
box(ax,.50,.22,.82,.072,"Reward and Lagrangian Module\nQoS + fuzzy bonus − adaptive penalties",PALE,9.2)
box(ax,.50,.07,.82,.072,"Selected UAV and Feedback\nUpdated energy, next state and decision log",GREEN,9.2)

arrow(ax,(.50,.912),(.50,.858))
arrow(ax,(.47,.782),(.285,.723))
arrow(ax,(.53,.782),(.715,.723))
arrow(ax,(.285,.637),(.47,.558))
arrow(ax,(.715,.637),(.53,.558))
arrow(ax,(.50,.482),(.50,.408))
arrow(ax,(.50,.332),(.50,.258))
arrow(ax,(.50,.182),(.50,.108))
finish(fig,ax,"figure_2_system_architecture.png")

# Figure 3: trace. Compact boxes and explicit gaps keep all steps distinct.
fig, ax = plt.subplots(figsize=(7.5, 10.0))
labs=["Acquire vehicle state at time t",
      "Read UAV u position and residual energy",
      "Calculate 3-D distance d(v,u)",
      "Estimate signal S(v,u), delay D(v,u),\nand PDR P(v,u)",
      "Normalize residual energy E%(u)",
      "Apply feasibility checks and\ncalculate fuzzy risks",
      "Calculate fuzzy priority F(v,u)",
      "Calculate candidate score C(v,u)",
      "Repeat for all eight UAVs\nand sort candidates"]
ys=[.95,.845,.74,.635,.53,.425,.32,.215,.095]
for y,l in zip(ys,labs):
    box(ax,.5,y,.72,.046,l, ORANGE if "fuzzy" in l.lower() else PALE,8.8)
for i in range(len(ys)-1):
    arrow(ax,(.5,ys[i]-.042),(.5,ys[i+1]+.042))
finish(fig,ax,"figure_3_candidate_trace.png")

# Figure 4: training flowchart. Larger gaps keep arrows visible and prevent
# arrowheads from sitting inside the process boxes.
fig, ax = plt.subplots(figsize=(9.4, 14.0))
labs=["Select operating scenario","Initialize PPO actor and critic","Fuzzy-guided warm start",
      "Reset environment and initialize eight UAVs","Construct observation s(t)","Evaluate, mask and rank candidate UAVs",
      "PPO selects ranked action","Execute communication and calculate reward","Update energy, vehicle state and multipliers",
      "Compute GAE and returns","Update PPO actor and critic","Save trained model, logs and metrics"]
ys=[.960,.880,.800,.720,.640,.560,.480,.400,.320,.225,.145,.055]
box_h=.038
for y,l in zip(ys,labs):
    box(ax,.46,y,.58,box_h,l, GREEN if y in (.960,.055) else PALE,8.6)
for i in range(len(ys)-1):
    arrow(ax,(.46,ys[i]-box_h/2-.014),(.46,ys[i+1]+box_h/2+.014))

# Episode loop: from the update step back to candidate evaluation.
ax.text(.91,.440,"Repeat until\nepisode ends",ha="center",va="center",fontsize=8.6,color=DARK)
arrow(ax,(.82,.320),(.82,.560))
arrow(ax,(.82,.560),(.765,.560))
finish(fig,ax,"figure_4_training_flowchart.png")

# Figure 5: sequence diagram
fig, ax = plt.subplots(figsize=(10, 6.5))
actors=["Vehicle","Environment","Link Module","Fuzzy Module","PPO Agent","Lagrangian"]
xs=[.07,.24,.41,.58,.75,.92]
for x,a in zip(xs,actors): box(ax,x,.93,.13,.065,a,GREEN,8.5); ax.plot([x,x],[.88,.08],"--",color="#8AA6B8",lw=1)
events=[(.07,.24,.80,"Current position"),(.24,.41,.70,"UAV states"),(.41,.58,.60,"Link metrics"),
        (.58,.24,.50,"Fuzzy priorities"),(.24,.75,.40,"Ranked state/candidates"),(.75,.24,.30,"Selected action"),
        (.24,.92,.20,"Reward and violations"),(.24,.07,.10,"Communication outcome")]
for x1,x2,y,t in events:
    arrow(ax,(x1,y),(x2,y)); ax.text((x1+x2)/2,y+.018,t,ha="center",va="bottom",fontsize=7.8,color=DARK)
finish(fig,ax,"figure_5_sequence_diagram.png")
print(OUT)
