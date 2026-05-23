import io

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from matplotlib.patches import Circle

from gym import LightsOutEnv


def show_state(state, ax, action=None):
    state = np.asarray(state)
    h, w = state.shape

    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_aspect('equal')
    ax.set_facecolor("#1a1a1a")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(h + 1):
        ax.axhline(i - 0.5, color='#333', linewidth=1, zorder=1)
    for j in range(w + 1):
        ax.axvline(j - 0.5, color='#333', linewidth=1, zorder=1)

    for r in range(h):
        for c in range(w):
            if state[r, c]:
                for radius, alpha in [(0.6, 0.10), (0.5, 0.18), (0.4, 0.28)]:
                    ax.add_patch(Circle((c, r), radius,
                                        color="#ffeb3b", alpha=alpha, zorder=2))
                ax.add_patch(Circle((c, r), 0.32,
                                    color="#fff59d", zorder=3))
            else:
                ax.add_patch(Circle((c, r), 0.32,
                                    color="#2a2a2a", zorder=3))

    if action is not None:
        r, c = map(int, action.split("_"))
        ax.add_patch(Circle((c, r), 0.45, fill=False,
                            edgecolor="#ff5252", linewidth=3, zorder=4))
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                ax.add_patch(Circle((nc, nr), 0.45, fill=False,
                                    edgecolor="#90caf9", linewidth=2,
                                    linestyle="--", zorder=4))


def show_solution(state, actions):
    env = LightsOutEnv()
    env.set_state(state)
    h, w = env.h, env.w
    solved_state = np.zeros((h, w), dtype=np.int8)

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
                                 facecolor="#1a1a1a")
        show_state(s, axes[0], action=action)
        axes[0].set_title(
            f"step {i}"
            + (f" — {action}" if action else "")
            + (" Solved!" if solved else ""),
            color="white",
        )
        show_state(solved_state, axes[1])
        axes[1].set_title("Goal", color="white")
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
        interval=400,
        description='play',
    )
    widgets.jslink((play, 'value'), (slider, 'value'))

    controls = widgets.HBox([play, prev_btn, slider, next_btn])
    ui = widgets.VBox([controls, img], layout=widgets.Layout(align_items='center'))
    display(ui)



