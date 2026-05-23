import io

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from IPython.display import display
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle, Wedge

from gym import RotateSlideEnv


_TAB = sns.color_palette("Set2", 12)
# index v -> color for ball value v; ball 0 = blank
PALETTE = ["#e8e8e8"] + [_TAB[i] for i in range(1, 7)]
BG_COLOR = "#fafafa"
EDGE = "#222"
HIGHLIGHT = "#d92d20"
TEXT_COLOR = "#222"


def _color(v):
    return PALETTE[int(v)] if 0 <= int(v) < len(PALETTE) else "#888"


def _draw_front(ax, main, top_pos, top_ball, action=None):
    h, n = main.shape
    ax.set_xlim(-0.6, n + 0.2)
    ax.set_ylim(-0.4, h + 1.5)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_facecolor(BG_COLOR)

    for r in range(h):
        y = h - 1 - r  # draw level r at vertical position (level 0 at bottom)
        for c in range(n):
            v = int(main[r, c])
            ax.add_patch(Rectangle((c, y), 1, 1,
                                   facecolor=_color(v),
                                   edgecolor=EDGE, linewidth=1.2, zorder=2))
            if v != 0:
                ax.text(c + 0.5, y + 0.5, str(v),
                        ha='center', va='center',
                        fontsize=12, fontweight='bold',
                        color=TEXT_COLOR, zorder=3)
        ax.text(-0.25, y + 0.5, f"L{r}", ha='right', va='center',
                fontsize=9, color="#555")

    # Top cell sits above level H-1, drawn at the very top of the canvas
    top_y = h + 0.25
    ax.add_patch(Rectangle((top_pos, top_y), 1, 1,
                           facecolor=_color(top_ball),
                           edgecolor=EDGE, linewidth=1.5, zorder=2))
    if top_ball != 0:
        ax.text(top_pos + 0.5, top_y + 0.5, str(top_ball),
                ha='center', va='center', fontsize=12, fontweight='bold',
                color=TEXT_COLOR, zorder=3)
    ax.text(-0.25, top_y + 0.5, "top", ha='right', va='center',
            fontsize=9, color="#555")

    for c in range(n):
        ax.text(c + 0.5, -0.2, str(c), ha='center', va='top',
                fontsize=9, color="#555")

    if action is not None:
        _highlight_action(ax, action, h, n, top_pos, top_y)


def _highlight_action(ax, action, h, n, top_pos, top_y):
    if action in ("TL", "TR"):
        dx = -0.6 if action == "TL" else 0.6
        cx = top_pos + 0.5
        cy = top_y + 0.5
        ax.add_patch(Rectangle((top_pos, top_y), 1, 1, fill=False,
                               edgecolor=HIGHLIGHT, linewidth=2.5, zorder=10))
        ax.annotate("", xy=(cx + dx, cy), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle="-|>", lw=2,
                                    color=HIGHLIGHT, mutation_scale=20),
                    zorder=11)
    elif action.startswith("L_") or action.startswith("R_"):
        l = int(action.split("_")[1])
        y = h - 1 - l
        ax.add_patch(Rectangle((0, y), n, 1, fill=False,
                               edgecolor=HIGHLIGHT, linewidth=2.5, zorder=10))
        dx = -0.6 if action.startswith("L_") else 0.6
        ax.annotate("", xy=(n / 2 + dx, y + 0.5), xytext=(n / 2, y + 0.5),
                    arrowprops=dict(arrowstyle="-|>", lw=2,
                                    color=HIGHLIGHT, mutation_scale=20),
                    zorder=11)
    elif action.startswith("U_"):
        r = int(action.split("_")[1])
        y_src = h - 1 - r + 0.5
        y_dst = y_src + 1.0 if r < h - 1 else top_y + 0.5
        ax.annotate("", xy=(top_pos + 0.5, y_dst),
                    xytext=(top_pos + 0.5, y_src),
                    arrowprops=dict(arrowstyle="-|>", lw=2.5,
                                    color=HIGHLIGHT, mutation_scale=22),
                    zorder=11)
    elif action.startswith("D_"):
        r = int(action.split("_")[1])
        if r == h:
            y_src = top_y + 0.5
            y_dst = h - 1 - (h - 1) + 0.5  # level h-1
        else:
            y_src = h - 1 - r + 0.5
            y_dst = y_src - 1.0
        ax.annotate("", xy=(top_pos + 0.5, y_dst),
                    xytext=(top_pos + 0.5, y_src),
                    arrowprops=dict(arrowstyle="-|>", lw=2.5,
                                    color=HIGHLIGHT, mutation_scale=22),
                    zorder=11)


def _draw_ring(ax, row, top_pos, label=None):
    n = len(row)
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_facecolor(BG_COLOR)

    seg = 360.0 / n
    for c in range(n):
        v = int(row[c])
        # column 0 at 12 o'clock, going clockwise
        ang1 = 90 - c * seg
        ang0 = 90 - (c + 1) * seg
        ax.add_patch(Wedge((0, 0), 1.0, ang0, ang1,
                           width=0.55,
                           facecolor=_color(v),
                           edgecolor=EDGE, linewidth=0.8))
        if v != 0:
            ang_mid = np.deg2rad(90 - (c + 0.5) * seg)
            ax.text(0.72 * np.cos(ang_mid), 0.72 * np.sin(ang_mid),
                    str(v), ha='center', va='center',
                    fontsize=7.5, fontweight='bold', color=TEXT_COLOR)

    # mark current top_pos with an inward arrow outside the ring
    ang_mid = np.deg2rad(90 - (top_pos + 0.5) * seg)
    ax.annotate("",
                xy=(1.02 * np.cos(ang_mid), 1.02 * np.sin(ang_mid)),
                xytext=(1.32 * np.cos(ang_mid), 1.32 * np.sin(ang_mid)),
                arrowprops=dict(arrowstyle="-|>", lw=1.4,
                                color=HIGHLIGHT, mutation_scale=12))

    if label:
        ax.text(-1.35, 1.2, label, ha='left', va='top',
                fontsize=8, color="#555")


def show_state(state, fig, action=None):
    main = np.asarray(state["main"])
    top_pos = int(state["top_pos"])
    top_ball = int(state["top_ball"])
    h, n = main.shape

    gs = GridSpec(h, 2, figure=fig,
                  width_ratios=[2.2, 1], wspace=0.05, hspace=0.15,
                  left=0.06, right=0.97, top=0.92, bottom=0.05)

    ax_front = fig.add_subplot(gs[:, 0])
    _draw_front(ax_front, main, top_pos, top_ball, action)

    # rings: L0 on top, L{h-1} on bottom
    for r in range(h):
        ax_ring = fig.add_subplot(gs[r, 1])
        _draw_ring(ax_ring, main[r], top_pos, label=f"L{r}")


def show_solution(state, actions):
    env = RotateSlideEnv()
    env.set_state(state)

    trace = [(env.get_state(), None, False)]
    for a in actions:
        s, _, solved, _ = env.step(a)
        trace.append((s, a, solved))
        if solved:
            print("Solved!")
            break

    def render_png(i):
        s, action, solved = trace[i]
        fig = plt.figure(figsize=(8.5, 7), facecolor=BG_COLOR)
        show_state(s, fig, action=action)
        title = (f"step {i}"
                 + (f" — {action}" if action else "")
                 + (" Solved!" if solved else ""))
        fig.suptitle(title, fontsize=12)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', facecolor=fig.get_facecolor())
        plt.close(fig)
        return buf.getvalue()

    frames = [render_png(i) for i in range(len(trace))]

    img = widgets.Image(value=frames[0], format='png')
    slider = widgets.IntSlider(0, 0, len(frames) - 1, 1,
                               continuous_update=True,
                               layout=widgets.Layout(width='600px'))
    slider.observe(lambda c: setattr(img, 'value', frames[c['new']]), names='value')

    prev_btn = widgets.Button(description='◀', layout=widgets.Layout(width='40px'))
    next_btn = widgets.Button(description='▶', layout=widgets.Layout(width='40px'))
    prev_btn.on_click(lambda _: setattr(slider, 'value', max(0, slider.value - 1)))
    next_btn.on_click(lambda _: setattr(slider, 'value', min(len(frames) - 1, slider.value + 1)))

    play = widgets.Play(
        value=0, min=0, max=len(frames) - 1, step=1,
        interval=250,
        description='play',
    )
    widgets.jslink((play, 'value'), (slider, 'value'))

    controls = widgets.HBox([play, prev_btn, slider, next_btn])
    ui = widgets.VBox([controls, img], layout=widgets.Layout(align_items='center'))
    display(ui)


