"""Tk + matplotlib GUI: draw a highlight mask with Bezier shapes on the scalp grid, refine it (FM / active contour
on |flux| / flow alignment) and render both masks in Blender side by side.

    bash run.sh [--sample base_36_idx_42564] [--n 32]

Mouse (editor):
    new band / new closed   left click adds an anchor, right click or Enter finishes, Backspace removes the last
                            anchor, Esc cancels
    edit                    drag an anchor; right click an anchor deletes it; click inside a shape selects it;
                            double click on the selected shape's curve inserts an anchor; mouse wheel changes the
                            selected band's width; Delete removes the selected shape
    Ctrl+Z undo
"""
from __future__ import annotations

import argparse
import os
import queue
import threading
import time
import tkinter as tk
import traceback
from tkinter import colorchooser, filedialog, messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt  # noqa: E402

for _k in ("keymap.back", "keymap.forward", "keymap.home", "keymap.save", "keymap.quit", "keymap.fullscreen",
           "keymap.grid", "keymap.xscale", "keymap.yscale", "keymap.copy"):
    plt.rcParams[_k] = []                     # the editor's own keys (e/b/c, Backspace, Ctrl+...) win
import numpy as np  # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.path import Path  # noqa: E402

from highlighting.generate_from_3D_models.bezier_mask_gui import session as ses  # noqa: E402
from highlighting.generate_from_3D_models.bezier_mask_gui import shapes as shp  # noqa: E402
from highlighting.generate_from_3D_models.strands_editing import bezier_mask as bm  # noqa: E402

OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
METHODS = ("FM", "Active contour (|flux|)", "Flow 對齊 (band)")
PICK_PX = 9
COL_MASK, COL_REF = "#f28e2b", "#2ca02c"


def _hex(rgb):
    return "#%02x%02x%02x" % tuple(int(round(255 * min(max(c, 0), 1))) for c in rgb)


class App:
    def __init__(self, root, sample=None, n=32):
        self.root = root
        self.S = ses.Session()
        self.shapes: list[shp.Shape] = []
        self.drawing: shp.Shape | None = None
        self.sel = None
        self.drag = None                                   # (shape index, anchor index)
        self.undo_stack = []
        self.region = self.labels = None
        self.refined = None                                # dict(labels, region, info, shapes?, contours?)
        self.scores = {}
        self.renders = {}
        self.busy = False
        self.q = queue.Queue()
        self.base_rgb = (0.05, 0.05, 0.05)
        self.hl_rgb = (0.95, 0.75, 0.35)
        self._build_ui(sample, n)
        self.root.after(100, self._poll)

    # ================================================================ UI
    def _build_ui(self, sample, n):
        self.root.title("Bezier highlight mask editor")
        self.root.geometry("1680x960")
        pw = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        pw.pack(fill=tk.BOTH, expand=True)
        ctrl = ttk.Frame(pw, padding=4)
        mid = ttk.Frame(pw)
        right = ttk.Frame(pw)
        pw.add(ctrl, weight=0)
        pw.add(mid, weight=3)
        pw.add(right, weight=2)

        # ---- sample
        f = ttk.LabelFrame(ctrl, text="髮型", padding=4)
        f.pack(fill=tk.X, pady=2)
        self.v_sample = tk.StringVar(value=sample or "")
        cb = ttk.Combobox(f, textvariable=self.v_sample, values=ses.list_samples(), width=26)
        cb.grid(row=0, column=0, columnspan=3, sticky="ew")
        ttk.Label(f, text="grid n").grid(row=1, column=0, sticky="w")
        self.v_n = tk.IntVar(value=n)
        ttk.Spinbox(f, from_=16, to=128, increment=16, textvariable=self.v_n, width=5).grid(row=1, column=1, sticky="w")
        ttk.Button(f, text="載入", command=self.on_load).grid(row=1, column=2, sticky="e")

        # ---- display
        f = ttk.LabelFrame(ctrl, text="顯示", padding=4)
        f.pack(fill=tk.X, pady=2)
        self.v_flow = tk.BooleanVar(value=True)
        self.v_flux = tk.BooleanVar(value=False)
        self.v_ctrl = tk.BooleanVar(value=True)
        self.v_ref = tk.BooleanVar(value=True)
        for i, (t, v) in enumerate((("flow 箭頭", self.v_flow), ("|flux|", self.v_flux),
                                    ("Bezier 控制點", self.v_ctrl), ("調整結果", self.v_ref))):
            ttk.Checkbutton(f, text=t, variable=v, command=self.redraw).grid(row=i // 2, column=i % 2, sticky="w")

        # ---- drawing
        f = ttk.LabelFrame(ctrl, text="繪製", padding=4)
        f.pack(fill=tk.X, pady=2)
        self.v_mode = tk.StringVar(value="edit")
        for i, (t, m) in enumerate((("編輯 (E)", "edit"), ("新增 band (B)", "band"), ("新增 closed (C)", "closed"))):
            ttk.Radiobutton(f, text=t, value=m, variable=self.v_mode, command=self._mode_changed).grid(
                row=0 if i < 2 else 1, column=i % 2, sticky="w")
        self.v_sub = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="減去 (挖空)", variable=self.v_sub, command=self._sub_changed).grid(row=1, column=1, sticky="w")
        ttk.Label(f, text="band 寬 (格)").grid(row=2, column=0, sticky="w")
        self.v_width = tk.DoubleVar(value=2.0)
        ttk.Scale(f, from_=0.3, to=10, variable=self.v_width, command=self._width_changed).grid(row=2, column=1, sticky="ew")
        self.l_width = ttk.Label(f, text="")
        self.l_width.grid(row=3, column=0, columnspan=2, sticky="w")
        bf = ttk.Frame(f)
        bf.grid(row=4, column=0, columnspan=2, sticky="ew")
        for i, (t, c) in enumerate((("完成形狀", self.finish_shape), ("刪除選取", self.delete_selected),
                                    ("復原", self.undo), ("清除全部", self.clear_all))):
            ttk.Button(bf, text=t, command=c, width=9).grid(row=i // 2, column=i % 2, sticky="ew")
        af = ttk.Frame(f)
        af.grid(row=5, column=0, columnspan=2, sticky="ew")
        ttk.Button(af, text="自動 chunky", command=self.on_auto).pack(side=tk.LEFT)
        ttk.Label(af, text=" seed").pack(side=tk.LEFT)
        self.v_seed = tk.IntVar(value=0)
        ttk.Spinbox(af, from_=0, to=9999, textvariable=self.v_seed, width=5).pack(side=tk.LEFT)

        # ---- refinement
        f = ttk.LabelFrame(ctrl, text="調整 mask", padding=4)
        f.pack(fill=tk.X, pady=2)
        self.v_method = tk.StringVar(value=METHODS[0])
        ttk.Combobox(f, textvariable=self.v_method, values=METHODS, state="readonly", width=24).grid(
            row=0, column=0, columnspan=4, sticky="ew")
        self.v_overlap = tk.StringVar(value="projection")
        self.v_lam = tk.StringVar(value="1000")
        self.v_passes = tk.StringVar(value="3")
        self.v_bfm = tk.BooleanVar(value=False)
        self.v_snap = tk.StringVar(value="8")
        self.v_rounds = tk.StringVar(value="1")
        self.v_totip = tk.BooleanVar(value=False)
        self.v_blend = tk.StringVar(value="1.0")
        rows = [("FM overlap", ttk.Combobox(f, textvariable=self.v_overlap, state="readonly", width=14,
                                            values=("projection", "cell", "projection_depth"))),
                ("FM λ / passes", None), ("active snap / rounds", None), ("flow blend", None)]
        ttk.Label(f, text=rows[0][0]).grid(row=1, column=0, columnspan=2, sticky="w")
        rows[0][1].grid(row=1, column=2, columnspan=2, sticky="ew")
        ttk.Label(f, text="FM λ / passes").grid(row=2, column=0, columnspan=2, sticky="w")
        ttk.Entry(f, textvariable=self.v_lam, width=7).grid(row=2, column=2)
        ttk.Entry(f, textvariable=self.v_passes, width=4).grid(row=2, column=3)
        ttk.Checkbutton(f, text="Boundary FM", variable=self.v_bfm).grid(row=3, column=0, columnspan=4, sticky="w")
        ttk.Label(f, text="snap 半徑 / rounds").grid(row=4, column=0, columnspan=2, sticky="w")
        ttk.Entry(f, textvariable=self.v_snap, width=7).grid(row=4, column=2)
        ttk.Entry(f, textvariable=self.v_rounds, width=4).grid(row=4, column=3)
        ttk.Label(f, text="flow blend (0-1)").grid(row=5, column=0, columnspan=2, sticky="w")
        ttk.Entry(f, textvariable=self.v_blend, width=7).grid(row=5, column=2)
        ttk.Checkbutton(f, text="延伸到髮際", variable=self.v_totip).grid(row=6, column=0, columnspan=4, sticky="w")
        bf = ttk.Frame(f)
        bf.grid(row=7, column=0, columnspan=4, sticky="ew")
        ttk.Button(bf, text="執行調整", command=self.on_refine).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(bf, text="採用結果", command=self.on_adopt).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(bf, text="清除結果", command=self.on_clear_refined).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # ---- render
        f = ttk.LabelFrame(ctrl, text="渲染 (Blender)", padding=4)
        f.pack(fill=tk.X, pady=2)
        self.b_base = tk.Button(f, text="底色", width=8, command=lambda: self._pick_color("base"))
        self.b_hl = tk.Button(f, text="挑染色", width=8, command=lambda: self._pick_color("hl"))
        self.b_base.grid(row=0, column=0, sticky="ew")
        self.b_hl.grid(row=0, column=1, sticky="ew")
        self._color_buttons()
        self.v_samples = tk.StringVar(value="64")
        self.v_res = tk.StringVar(value="512")
        self.v_sub_strands = tk.StringVar(value="0.3")
        for i, (t, v) in enumerate((("samples", self.v_samples), ("解析度", self.v_res), ("髮絲比例", self.v_sub_strands))):
            ttk.Label(f, text=t).grid(row=1 + i, column=0, sticky="w")
            ttk.Entry(f, textvariable=v, width=8).grid(row=1 + i, column=1, sticky="w")
        self.v_view = tk.StringVar(value="multiview")
        vf = ttk.Frame(f)
        vf.grid(row=4, column=0, columnspan=2, sticky="w")
        ttk.Radiobutton(vf, text="正面", value="front", variable=self.v_view, command=self.show_renders).pack(side=tk.LEFT)
        ttk.Radiobutton(vf, text="六視角", value="multiview", variable=self.v_view, command=self.show_renders).pack(side=tk.LEFT)
        bf = ttk.Frame(f)
        bf.grid(row=5, column=0, columnspan=2, sticky="ew")
        ttk.Button(bf, text="渲染 mask", command=lambda: self.on_render(("mask",))).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(bf, text="渲染結果", command=lambda: self.on_render(("refined",))).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(bf, text="兩者", command=lambda: self.on_render(("mask", "refined"))).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # ---- io + metrics
        f = ttk.LabelFrame(ctrl, text="存檔 / 指標", padding=4)
        f.pack(fill=tk.X, pady=2)
        bf = ttk.Frame(f)
        bf.pack(fill=tk.X)
        ttk.Button(bf, text="存檔", command=self.on_save).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(bf, text="載入 shapes", command=self.on_load_shapes).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(bf, text="grid score", command=self.on_score).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.l_metrics = ttk.Label(f, text="", justify=tk.LEFT, font=("TkFixedFont", 9))
        self.l_metrics.pack(fill=tk.X)

        self.l_status = ttk.Label(ctrl, text="請選髮型後按「載入」", foreground="#1f4e9c", wraplength=300)
        self.l_status.pack(fill=tk.X, pady=2)
        self.log = tk.Text(ctrl, height=8, width=42, font=("TkFixedFont", 8), state=tk.DISABLED)
        self.log.pack(fill=tk.BOTH, expand=True)

        # ---- editor figure
        self.fig = Figure(figsize=(7, 7), dpi=100)
        self.ax = self.fig.add_axes([0.02, 0.02, 0.96, 0.93])
        self.canvas = FigureCanvasTkAgg(self.fig, master=mid)
        self.toolbar = NavigationToolbar2Tk(self.canvas, mid)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        for ev, h in (("button_press_event", self.on_press), ("motion_notify_event", self.on_motion),
                      ("button_release_event", self.on_release), ("key_press_event", self.on_key),
                      ("scroll_event", self.on_scroll)):
            self.canvas.mpl_connect(ev, h)
        self.ax.set_axis_off()
        self.ax.set_title("load a hairstyle", fontsize=10)

        # ---- render figure
        self.rfig = Figure(figsize=(5, 7), dpi=100)
        self.rcanvas = FigureCanvasTkAgg(self.rfig, master=right)
        self.rcanvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.show_renders()
        self._width_label()

    def _color_buttons(self):
        for b, rgb in ((self.b_base, self.base_rgb), (self.b_hl, self.hl_rgb)):
            b.configure(bg=_hex(rgb), fg="white" if sum(rgb) < 1.5 else "black", activebackground=_hex(rgb))

    def _pick_color(self, which):
        cur = self.base_rgb if which == "base" else self.hl_rgb
        c = colorchooser.askcolor(color=_hex(cur), parent=self.root)
        if c and c[0]:
            rgb = tuple(x / 255 for x in c[0])
            if which == "base":
                self.base_rgb = rgb
            else:
                self.hl_rgb = rgb
            self._color_buttons()

    # ================================================================ background jobs
    def status(self, msg):
        self.q.put(("log", msg))

    def run_bg(self, title, fn, done=None):
        if self.busy:
            messagebox.showinfo("忙碌中", "上一個工作還沒結束")
            return
        self.busy = True
        self.l_status.configure(text=f"⏳ {title} ...")
        self.root.configure(cursor="watch")

        def work():
            t = time.time()
            try:
                res = fn()
                self.q.put(("done", (title, done, res, time.time() - t)))
            except Exception as e:  # noqa: BLE001
                self.q.put(("error", (title, e, traceback.format_exc())))
        threading.Thread(target=work, daemon=True).start()

    def _poll(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    self._append_log(payload)
                elif kind == "done":
                    title, done, res, dt = payload
                    self.busy = False
                    self.root.configure(cursor="")
                    self.l_status.configure(text=f"✓ {title} ({dt:.0f} s)")
                    if done:
                        done(res)
                elif kind == "error":
                    title, e, tb = payload
                    self.busy = False
                    self.root.configure(cursor="")
                    self.l_status.configure(text=f"✗ {title}: {e}")
                    self._append_log(tb)
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _append_log(self, msg):
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, time.strftime("%H:%M:%S ") + str(msg).rstrip() + "\n")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    # ================================================================ loading
    @property
    def loaded(self):
        return self.S.sample is not None

    def on_load(self):
        name = self.v_sample.get().strip()
        if not name:
            return
        n = int(self.v_n.get())

        def done(_):
            self.shapes, self.drawing, self.sel, self.undo_stack = [], None, None, []
            self.refined, self.scores, self.renders = None, {}, {}
            self.out_dir = os.path.join(OUT_ROOT, name)
            self._bg_cache = None
            self.mask_changed()
            self.show_renders()
            self._width_label()
        self.run_bg(f"載入 {name}", lambda: self.S.load(name, n, log=self.status), done)

    # ================================================================ shape state
    def push_undo(self):
        self.undo_stack.append([s.copy() for s in self.shapes])
        del self.undo_stack[:-50]

    def undo(self):
        if self.drawing is not None:
            self.drawing = None
            self.redraw()
            return
        if self.undo_stack:
            self.shapes = self.undo_stack.pop()
            self.sel = None
            self.mask_changed()

    def clear_all(self):
        if self.shapes:
            self.push_undo()
        self.shapes, self.drawing, self.sel = [], None, None
        self.mask_changed()

    def delete_selected(self):
        if self.sel is not None:
            self.push_undo()
            del self.shapes[self.sel]
            self.sel = None
            self.mask_changed()

    def finish_shape(self):
        d = self.drawing
        if d is None:
            return
        if d.complete:
            self.push_undo()
            self.shapes.append(d)
            self.sel = len(self.shapes) - 1
        self.drawing = None
        self.mask_changed()

    def on_auto(self):
        if not self.loaded:
            return

        def done(new):
            self.push_undo()
            self.shapes += new
            self.mask_changed()
        seed = int(self.v_seed.get())
        self.run_bg("自動 chunky (strands_editing.bezier_mask)", lambda: self.S.auto_chunky(seed=seed), done)

    def _mode_changed(self):
        if self.drawing is not None:
            self.finish_shape()
        self.sel = None if self.v_mode.get() != "edit" else self.sel
        self.redraw()

    def _sub_changed(self):
        if self.sel is not None and self.v_mode.get() == "edit":
            self.push_undo()
            self.shapes[self.sel].subtract = bool(self.v_sub.get())
            self.mask_changed()

    def _width_changed(self, _=None):
        w = float(self.v_width.get())
        self._width_label()
        if self.sel is not None and self.shapes[self.sel].kind == "band":
            self.shapes[self.sel].width = w
            self.redraw_shapes()
            if not getattr(self, "_width_pending", False):
                self._width_pending = True
                self.root.after(250, self._width_commit)

    def _width_commit(self):
        self._width_pending = False
        self.mask_changed()

    def _width_label(self):
        w = float(self.v_width.get())
        mm = f" ≈ {w * self.S.cell_spacing_mm:.0f} mm" if self.loaded else ""
        self.l_width.configure(text=f"{w:.1f} 格{mm}")

    def mask_changed(self):
        """shapes changed: rebuild the region and the per-strand labels, redraw"""
        if self.loaded:
            done = [s for s in self.shapes if s.complete]
            self.region = self.S.region(done)
            self.labels = self.S.labels(self.region)
            self.scores.pop("mask", None)
        self.redraw()

    # ================================================================ refinement
    def on_refine(self):
        if not self.loaded or self.labels is None:
            return
        m = self.v_method.get()
        labels, region, shapes = self.labels.copy(), self.region, [s.copy() for s in self.shapes]
        try:
            lam, passes = int(float(self.v_lam.get())), int(self.v_passes.get())
            snap, rounds = float(self.v_snap.get()), int(self.v_rounds.get())
            blend = float(self.v_blend.get())
        except ValueError:
            messagebox.showerror("參數", "參數格式錯誤")
            return
        overlap, bfm, totip = self.v_overlap.get(), bool(self.v_bfm.get()), bool(self.v_totip.get())

        if m == METHODS[0]:
            if not labels.any():
                messagebox.showinfo("FM", "mask 是空的")
                return

            def fn():
                st, reg, info = self.S.refine_fm(labels, region, overlap=overlap, lam=lam, max_passes=passes,
                                                 boundary=bfm, log=self.status)
                return dict(labels=st, region=reg, info=info, score=self.S.grid_score(st))
        elif m == METHODS[1]:
            def fn():
                st, reg, info, xy = self.S.refine_active(region, snap_radius=snap, n_rounds=rounds)
                return dict(labels=st, region=reg, info=info, contours=xy, score=self.S.grid_score(st))
        else:
            if not any(s.kind == "band" for s in shapes):
                messagebox.showinfo("Flow 對齊", "flow 對齊只作用在 band 形狀上")
                return

            def fn():
                new, info = self.S.flow_align(shapes, to_tip=totip, blend=blend)
                reg = self.S.region([s for s in new if s.complete])
                st = self.S.labels(reg)
                info["flow |sin|"] = f"{self.S.flow_alignment(shapes):.3f} → {self.S.flow_alignment(new):.3f}"
                return dict(labels=st, region=reg, info=info, shapes=new, score=self.S.grid_score(st))

        def done(res):
            self.refined = res
            self.scores["refined"] = res.pop("score")
            if "mask" not in self.scores:
                self.scores["mask"] = None
            self.renders.pop("refined", None)
            self._append_log(" | ".join(f"{k}: {v}" for k, v in res["info"].items()))
            self.redraw()
            self.show_renders()
            self.on_score(only_missing=True)
        self.run_bg(m, fn, done)

    def on_adopt(self):
        """the refinement result becomes the editable mask (flow align: its shapes; FM / active contour: the
        region's outlines as closed shapes)"""
        r = self.refined
        if r is None:
            return
        self.push_undo()
        if r.get("shapes") is not None:
            self.shapes = [s.copy() for s in r["shapes"]]
        else:
            self.shapes = shp.shapes_from_region(r["region"])
        self.sel = None
        self.refined = None
        self.renders["mask"] = self.renders.pop("refined", None)
        if self.renders["mask"] is None:
            self.renders.pop("mask")
        self.mask_changed()
        self.show_renders()
        self._append_log(f"採用結果 → {len(self.shapes)} 個形狀 (FM / active contour 的結果會轉成 closed 輪廓)")

    def on_clear_refined(self):
        self.refined = None
        self.scores.pop("refined", None)
        self.renders.pop("refined", None)
        self.redraw()
        self.show_renders()

    def on_score(self, only_missing=False):
        if not self.loaded or self.labels is None:
            return
        todo = {"mask": self.labels}
        if self.refined is not None:
            todo["refined"] = self.refined["labels"]
        if only_missing:
            todo = {k: v for k, v in todo.items() if self.scores.get(k) is None}
        if not todo:
            return

        def done(res):
            self.scores.update(res)
            self.update_metrics()
        self.run_bg("grid score", lambda: {k: self.S.grid_score(v) for k, v in todo.items()}, done)

    # ================================================================ render
    def on_render(self, which):
        if not self.loaded:
            return
        jobs = {}
        if "mask" in which and self.labels is not None:
            jobs["mask"] = self.labels.copy()
        if "refined" in which and self.refined is not None:
            jobs["refined"] = self.refined["labels"].copy()
        if not jobs:
            messagebox.showinfo("渲染", "沒有可渲染的 mask（調整結果需要先執行調整）")
            return
        try:
            kw = dict(base_rgb=self.base_rgb, hl_rgb=self.hl_rgb, samples=int(self.v_samples.get()),
                      resolution=int(self.v_res.get()), subsample=float(self.v_sub_strands.get()))
        except ValueError:
            messagebox.showerror("渲染", "參數格式錯誤")
            return
        stamp = time.strftime("%Y%m%d_%H%M%S")

        def fn():
            out = {}
            for k, lab in jobs.items():
                self.status(f"Blender 渲染 {k} ({lab.mean():.1%} 髮絲挑染) ...")
                out[k] = self.S.render(lab, os.path.join(self.out_dir, "renders", f"{stamp}_{k}"), log=self.status, **kw)
                self.status(f"{k}: {out[k]['front']} ({out[k]['seconds']} s)")
            return out

        def done(res):
            self.renders.update(res)
            self.show_renders()
        self.run_bg("Blender 渲染", fn, done)

    def show_renders(self):
        self.rfig.clear()
        view = self.v_view.get()
        for i, (k, title) in enumerate((("mask", "drawn mask"), ("refined", "refined"))):
            ax = self.rfig.add_subplot(2, 1, i + 1)
            ax.set_axis_off()
            r = self.renders.get(k)
            path = (r or {}).get(view) or (r or {}).get("front")
            if path and os.path.exists(path):
                ax.imshow(plt.imread(path))
                s = self.scores.get(k)
                ax.set_title(title + (f"   grid score {s:.4f}" if s is not None else ""), fontsize=9,
                             color=COL_MASK if k == "mask" else COL_REF)
            else:
                ax.text(0.5, 0.5, f"{title}: not rendered", ha="center", va="center", color="#888",
                        transform=ax.transAxes)
        self.rfig.tight_layout()
        self.rcanvas.draw_idle()

    # ================================================================ io
    def on_save(self):
        if not self.loaded:
            return
        d = os.path.join(self.out_dir, "saves", time.strftime("%Y%m%d_%H%M%S"))
        extra = dict(scores={k: v for k, v in self.scores.items() if v is not None},
                     refined_info=self.refined["info"] if self.refined else None)
        self.S.save(d, self.shapes, self.labels, self.refined["labels"] if self.refined else None, extra)
        if self.refined is not None and self.refined.get("shapes") is not None:
            import json
            json.dump([s.to_dict() for s in self.refined["shapes"]], open(os.path.join(d, "refined_shapes.json"), "w"))
        self.fig.savefig(os.path.join(d, "editor.png"), dpi=120)
        self._append_log(f"已存到 {d}")
        self.l_status.configure(text=f"已存到 {d}")

    def on_load_shapes(self):
        p = filedialog.askopenfilename(initialdir=OUT_ROOT, filetypes=[("shapes", "shapes.json"), ("json", "*.json")])
        if not p:
            return
        meta, shapes = ses.load_shapes(p)
        if self.loaded and (meta.get("sample") != self.S.sample or meta.get("n") != self.S.n):
            if not messagebox.askyesno("載入", f"檔案是 {meta.get('sample')} / n={meta.get('n')}，仍要套用到目前髮型?"):
                return
        if not self.loaded:
            messagebox.showinfo("載入", f"請先載入髮型 {meta.get('sample')} (n={meta.get('n')})")
            self.v_sample.set(meta.get("sample", ""))
            self.v_n.set(meta.get("n", 32))
            return
        self.push_undo()
        self.shapes = shapes
        self.mask_changed()

    # ================================================================ drawing
    def _background(self):
        S, n = self.S, self.S.n
        ext = (-0.5, n - 0.5, n - 0.5, -0.5)
        ax = self.ax
        ax.imshow(np.where(S.hair, 0.2, np.nan), cmap="Greys", vmin=0, vmax=1, extent=ext, interpolation="nearest")
        if self.v_flux.get():
            fl = np.where(S.hair, np.abs(S.flux), np.nan)
            ax.imshow(fl, cmap="Blues", vmin=0, vmax=max(np.nanpercentile(fl, 98), 1e-6), alpha=0.6, extent=ext,
                      interpolation="nearest")
        if self.v_flow.get():
            step = max(1, n // 32)
            yy, xx = np.nonzero(S.hair & (np.add.outer(np.arange(n) % step, np.arange(n) % step) == 0))
            ax.quiver(xx, yy, S.V[yy, xx, 0], S.V[yy, xx, 1], angles="xy", scale=45,
                      width=0.0018, color="#555", zorder=2)

    def redraw(self):
        ax = self.ax
        ax.clear()
        ax.set_axis_off()
        if not self.loaded:
            self.canvas.draw_idle()
            return
        S, n = self.S, self.S.n
        ext = (-0.5, n - 0.5, n - 0.5, -0.5)
        self._background()
        if self.region is not None and self.region.inside.any():
            ax.imshow(np.where(self.region.inside, 1.0, np.nan), cmap=ListedColormap([COL_MASK]), alpha=0.45,
                      extent=ext, interpolation="nearest", zorder=3)
        if self.refined is not None and self.v_ref.get():
            X, Y = bm.Region._pixel_centers(n, self.refined["region"].f)
            reg = self.refined["region"]
            if reg.inside.any() and not reg.inside.all():
                ax.contour(X, Y, reg.phi, [0], colors=COL_REF, linewidths=1.6, linestyles="--", zorder=4)
            for C in self.refined.get("contours") or []:
                C = np.asarray(C)
                if len(C):
                    ax.plot(C[:, 0], C[:, 1], ":", color="#155724", lw=0.8, zorder=4)
            for s in self.refined.get("shapes") or []:
                c = s.curve()
                ax.plot(c[:, 0], c[:, 1], "-", color=COL_REF, lw=1.0, zorder=4)
        self.shape_artists = []
        self.redraw_shapes(full=True)
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(n - 0.5, -0.5)
        mode = {"edit": "edit", "band": "new band", "closed": "new closed"}[self.v_mode.get()]
        ax.set_title(f"{S.sample}  n={n}  [{mode}]   orange = drawn mask, green dashed = refined", fontsize=9)
        self.update_metrics()
        self.canvas.draw_idle()

    def redraw_shapes(self, full=False):
        """(re)draw the shape curves, outlines and anchors only (cheap: used while dragging)"""
        if not self.loaded:
            return
        for a in getattr(self, "shape_artists", []):
            a.remove()
        self.shape_artists = []
        ax = self.ax
        items = [(i, s) for i, s in enumerate(self.shapes)]
        if self.drawing is not None:
            items.append((-1, self.drawing))
        for i, s in items:
            sel = i == self.sel or i == -1
            col = "#d62728" if sel else ("#7f3fbf" if s.subtract else "#1f4e9c")
            poly = s.polygon()
            if poly is not None:
                P = np.vstack([poly, poly[:1]])
                self.shape_artists += ax.plot(P[:, 0], P[:, 1], "-", color=col, lw=1.4 if sel else 1.0, zorder=5)
            if s.kind == "band" and len(s.anchors) >= 2:
                c = s.curve()
                self.shape_artists += ax.plot(c[:, 0], c[:, 1], "--", color=col, lw=0.8, zorder=5)
            if self.v_ctrl.get() and sel and len(s.anchors) >= 2:
                for seg in s.bezier_segments():
                    self.shape_artists += ax.plot(seg[[0, 1], 0], seg[[0, 1], 1], "-", color="#999", lw=0.6, zorder=5)
                    self.shape_artists += ax.plot(seg[[2, 3], 0], seg[[2, 3], 1], "-", color="#999", lw=0.6, zorder=5)
                    self.shape_artists += ax.plot(seg[1:3, 0], seg[1:3, 1], "s", color="#999", ms=3, zorder=6)
            A = s.anchors
            if len(A):
                self.shape_artists += ax.plot(A[:, 0], A[:, 1], "o", ms=6 if sel else 4, mfc="white" if sel else col,
                                              mec=col, zorder=7)
                if s.kind == "band":
                    self.shape_artists += ax.plot(A[:1, 0], A[:1, 1], "^", ms=8, mfc=col, mec="white", zorder=7)
        if not full:
            self.canvas.draw_idle()

    def update_metrics(self):
        if not self.loaded or self.labels is None:
            self.l_metrics.configure(text="")
            return
        S = self.S
        nb = sum(s.kind == "band" for s in self.shapes)
        lines = [f"形狀 {len(self.shapes)} (band {nb})  flow |sin| {S.flow_alignment(self.shapes):.3f}",
                 f"mask    挑染 {self.labels.mean():6.1%}  grid score {self._fmt(self.scores.get('mask'))}"]
        if self.refined is not None:
            r = self.refined
            lines.append(f"結果    挑染 {r['labels'].mean():6.1%}  grid score {self._fmt(self.scores.get('refined'))}")
            lines.append(f"變動髮絲 {(r['labels'] != self.labels).sum()}  ({r['info'].get('method', '')})")
            if "objective" in r["info"]:
                lines.append(r["info"]["objective"])
        self.l_metrics.configure(text="\n".join(lines))

    @staticmethod
    def _fmt(v):
        return "—" if v is None else f"{v:.4f}"

    # ================================================================ mouse / keys
    def _anchor_at(self, event, only=None):
        """(shape index, anchor index) of the anchor within PICK_PX pixels of the event, or None"""
        best, bd = None, PICK_PX
        cands = [(i, s) for i, s in enumerate(self.shapes) if only is None or i == only]
        for i, s in cands:
            if len(s.anchors) == 0:
                continue
            px = self.ax.transData.transform(s.anchors)
            d = np.hypot(px[:, 0] - event.x, px[:, 1] - event.y)
            j = int(np.argmin(d))
            if d[j] < bd:
                best, bd = (i, j), d[j]
        return best

    def on_press(self, ev):
        if not self.loaded or ev.inaxes is not self.ax or self.toolbar.mode:
            return
        self.canvas.get_tk_widget().focus_set()
        x, y = ev.xdata, ev.ydata
        mode = self.v_mode.get()
        if mode in ("band", "closed"):
            if ev.button == 1:
                if self.drawing is None:
                    self.drawing = shp.Shape(mode, np.zeros((0, 2)), width=float(self.v_width.get()),
                                             subtract=bool(self.v_sub.get()))
                self.drawing.anchors = np.vstack([self.drawing.anchors, [x, y]])
                self.redraw_shapes()
            elif ev.button == 3:
                self.finish_shape()
            return
        # edit mode
        hit = self._anchor_at(ev, only=self.sel) or self._anchor_at(ev)
        if ev.button == 1 and ev.dblclick and self.sel is not None:
            self._insert_anchor(self.sel, x, y)
            return
        if hit is not None:
            i, j = hit
            if ev.button == 3:
                s = self.shapes[i]
                if len(s.anchors) > s.min_anchors:
                    self.push_undo()
                    s.anchors = np.delete(s.anchors, j, axis=0)
                    self.sel = i
                    self.mask_changed()
                return
            if ev.button == 1:
                self.push_undo()
                self.drag = hit
                self._select(i)
            return
        if ev.button == 1:
            sel = None
            for i in range(len(self.shapes) - 1, -1, -1):
                poly = self.shapes[i].polygon()
                if poly is not None and Path(poly).contains_point((x, y)):
                    sel = i
                    break
            self._select(sel)

    def _select(self, i):
        self.sel = i
        if i is not None:
            s = self.shapes[i]
            self.v_sub.set(s.subtract)
            if s.kind == "band":
                self.v_width.set(s.width)
                self._width_label()
        self.redraw_shapes()

    def _insert_anchor(self, i, x, y):
        s = self.shapes[i]
        per = 24
        C = s.curve(per_seg=per)
        k = int(np.argmin(np.hypot(C[:, 0] - x, C[:, 1] - y)))
        seg = min(k // per, len(s.anchors) - (1 if s.kind == "band" else 0) - 1)
        self.push_undo()
        s.anchors = np.insert(s.anchors, seg + 1, [x, y], axis=0)
        self.mask_changed()

    def on_motion(self, ev):
        if self.drag is None or ev.inaxes is not self.ax or ev.xdata is None:
            return
        i, j = self.drag
        self.shapes[i].anchors[j] = (ev.xdata, ev.ydata)
        self.redraw_shapes()

    def on_release(self, ev):
        if self.drag is not None:
            self.drag = None
            self.mask_changed()

    def on_scroll(self, ev):
        if self.sel is None or self.shapes[self.sel].kind != "band":
            return
        w = float(np.clip(self.shapes[self.sel].width * (1.1 if ev.button == "up" else 1 / 1.1), 0.3, 10))
        self.v_width.set(w)
        self._width_changed()

    def on_key(self, ev):
        k = (ev.key or "").lower()
        if k == "enter":
            self.finish_shape()
        elif k == "escape":
            self.drawing = None
            self.redraw_shapes()
        elif k == "backspace" and self.drawing is not None:
            self.drawing.anchors = self.drawing.anchors[:-1]
            self.redraw_shapes()
        elif k in ("delete", "backspace"):
            self.delete_selected()
        elif k == "ctrl+z":
            self.undo()
        elif k in ("e", "b", "c"):
            self.v_mode.set({"e": "edit", "b": "band", "c": "closed"}[k])
            self._mode_changed()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", default=None)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--no_autoload", action="store_true")
    a = ap.parse_args()
    root = tk.Tk()
    app = App(root, a.sample, a.n)
    if a.sample and not a.no_autoload:
        root.after(200, app.on_load)
    root.mainloop()


if __name__ == "__main__":
    main()
