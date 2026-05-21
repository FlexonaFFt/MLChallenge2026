import io

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle

from gym import Fifteen2DEnv


TILE_FACE = "#e8dcc4"
TILE_BEVEL_LIGHT = "#f5ecd8"
TILE_BEVEL_DARK = "#a89878"
TILE_EDGE = "#3a2a1a"
NUMBER_COLOR = "#1a1208"
FRAME_COLOR = "#5c1f1f"
FRAME_INNER = "#2a0e0e"
BG_COLOR = "#d8c8a8"


def _draw_tile(ax, r, c, value):
    bevel = 0.08
    x0, y0 = c, r
    x1, y1 = c + 1, r + 1

    ax.add_patch(Polygon(
        [(x0, y0), (x1, y0), (x1 - bevel, y0 + bevel), (x0 + bevel, y0 + bevel)],
        facecolor=TILE_BEVEL_LIGHT, edgecolor="none", zorder=3,
    ))
    ax.add_patch(Polygon(
        [(x0, y0), (x0 + bevel, y0 + bevel), (x0 + bevel, y1 - bevel), (x0, y1)],
        facecolor=TILE_BEVEL_LIGHT, edgecolor="none", zorder=3,
    ))
    ax.add_patch(Polygon(
        [(x1, y0), (x1, y1), (x1 - bevel, y1 - bevel), (x1 - bevel, y0 + bevel)],
        facecolor=TILE_BEVEL_DARK, edgecolor="none", zorder=3,
    ))
    ax.add_patch(Polygon(
        [(x0, y1), (x1, y1), (x1 - bevel, y1 - bevel), (x0 + bevel, y1 - bevel)],
        facecolor=TILE_BEVEL_DARK, edgecolor="none", zorder=3,
    ))
    ax.add_patch(Rectangle(
        (x0 + bevel, y0 + bevel), 1 - 2 * bevel, 1 - 2 * bevel,
        facecolor=TILE_FACE, edgecolor="none", zorder=4,
    ))
    ax.add_patch(Rectangle(
        (x0, y0), 1, 1,
        facecolor="none", edgecolor=TILE_EDGE, linewidth=1.5, zorder=5,
    ))

    ax.text(c + 0.5, r + 0.5, str(value),
            ha="center", va="center",
            fontsize=28, fontweight="bold",
            color=NUMBER_COLOR,
            family="sans-serif",
            zorder=6)


def _draw_blank(ax, r, c):
    ax.add_patch(Rectangle(
        (c, r), 1, 1,
        facecolor=FRAME_INNER, edgecolor=TILE_EDGE, linewidth=1.5, zorder=3,
    ))


def show_state(state, ax, action=None):
    state = np.asarray(state)
    h, w = state.shape

    pad = 0.15
    ax.set_xlim(-pad, w + pad)
    ax.set_ylim(h + pad, -pad)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor(BG_COLOR)

    ax.add_patch(FancyBboxPatch(
        (-pad, -pad), w + 2 * pad, h + 2 * pad,
        boxstyle="round,pad=0,rounding_size=0.08",
        facecolor=FRAME_COLOR, edgecolor="#1a1208", linewidth=2, zorder=1,
    ))
    ax.add_patch(Rectangle(
        (0, 0), w, h,
        facecolor=FRAME_INNER, edgecolor="none", zorder=2,
    ))

    blank_pos = None
    for r in range(h):
        for c in range(w):
            v = int(state[r, c])
            if v == 0:
                blank_pos = (r, c)
                _draw_blank(ax, r, c)
            else:
                _draw_tile(ax, r, c, v)

    if action is not None and blank_pos is not None:
        br, bc = blank_pos
        if action == "U":
            tr, tc = br + 1, bc
        elif action == "D":
            tr, tc = br - 1, bc
        elif action == "L":
            tr, tc = br, bc + 1
        elif action == "R":
            tr, tc = br, bc - 1
        else:
            tr, tc = None, None

        if tr is not None and 0 <= tr < h and 0 <= tc < w:
            ax.add_patch(Rectangle(
                (tc, tr), 1, 1,
                fill=False, edgecolor="#d92d20", linewidth=3, zorder=7,
            ))
            sx, sy = bc + 0.5, br + 0.5
            ex, ey = tc + 0.5, tr + 0.5
            ax.annotate(
                "", xy=(ex, ey), xytext=(sx, sy),
                arrowprops=dict(arrowstyle="-|>", lw=3,
                                facecolor="#d92d20", edgecolor="#d92d20",
                                mutation_scale=22),
                zorder=8,
            )


def show_solution(state, actions):
    env = Fifteen2DEnv()
    env.set_state(state)
    h, w = env.h, env.w
    solved_state = env.solved_state()

    trace = [(state, None, False)]
    for a in actions:
        state, _, solved, _ = env.step(a)
        trace.append((state, a, solved))
        if solved:
            print("Solved!")
            break

    def render_png(i):
        s, action, solved = trace[i]
        fig, axes = plt.subplots(1, 2, figsize=(10, 5),
                                 facecolor=BG_COLOR)
        show_state(s, axes[0], action=action)
        axes[0].set_title(
            f"step {i}"
            + (f" — {action}" if action else "")
            + (" Solved!" if solved else ""),
            color=NUMBER_COLOR,
        )
        show_state(solved_state, axes[1])
        axes[1].set_title("Goal", color=NUMBER_COLOR)
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
        interval=200,
        description='play',
    )
    widgets.jslink((play, 'value'), (slider, 'value'))

    controls = widgets.HBox([play, prev_btn, slider, next_btn])
    ui = widgets.VBox([controls, img], layout=widgets.Layout(align_items='center'))
    display(ui)


