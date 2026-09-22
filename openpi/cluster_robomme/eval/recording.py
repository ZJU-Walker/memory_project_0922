"""Streaming H.264 recorder with a caption area outside the two camera images."""
from pathlib import Path
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from common import array


class RolloutRecorder:
    def __init__(self, directory, task, episode, fps=30, checkpoint="", mode="policy"):
        self.directory, self.task, self.episode = Path(directory), task, episode
        self.fps, self.checkpoint, self.mode = fps, checkpoint, mode
        self.count = 0
        self.writer = None
        self.raw_writer = None
        font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        self.font = ImageFont.truetype(font, 23) if Path(font).exists() else ImageFont.load_default()
        self.small = ImageFont.truetype(font, 17) if Path(font).exists() else ImageFont.load_default()

    def add(self, front, wrist, prediction, step, phase, status, goal):
        images = [Image.fromarray(array(im).astype(np.uint8)).convert("RGB") for im in (front, wrist)]
        images = [ImageOps.pad(im, (480, 360), color="#080c12") for im in images]
        canvas = Image.new("RGB", (960, 600), "#0c111b")
        draw = ImageDraw.Draw(canvas)
        draw.text((18, 12), "{}  |  episode {}  |  {}  |  {}".format(self.task, self.episode, phase, status), font=self.small, fill="#eef3fc")
        for x, im, title in zip((0, 480), images, ("FRONT", "WRIST")):
            canvas.paste(im, (x, 64))
            draw.text((x + 18, 42), title, font=self.small, fill="#9caec6")
        if phase == "conditioning":
            draw.rectangle((2, 64, 957, 423), outline="#f3c67a", width=4)
        title = {"policy": "PREDICTED SUBTASK", "oracle-subtask": "FORCED OFFICIAL SUBTASK — DIAGNOSTIC",
                 "action-chunk-diagnostic": "PREDICTED SUBTASK — EXECUTION HORIZON COMPARISON",
                 "base-policy": "PLAIN PI0.5 — ACTION POLICY",
                 "demonstration-replay": "RELEASED DEMONSTRATION — ACTION REPLAY"}.get(self.mode, "RECORDING CHECK — HOLD POSITION")
        draw.text((18, 436), title, font=self.small, fill="#67d9b5")
        sentence = prediction.get("subtask", "") if prediction else ""
        sentence = sentence or ("No subtask predictor in the plain baseline" if self.mode == "base-policy" else "No prediction yet" if self.mode == "policy" else "No learned policy connected")
        own = (prediction or {}).get("own_subtask")
        lines = textwrap.wrap(" ".join(sentence.split()), width=70) or [""]
        if own is not None:
            # forced mode with the model's own note shown (09-19): the forced sentence keeps one line, the own note the next
            lines = [textwrap.shorten(" ".join(sentence.split()), width=70, placeholder="...")]
        elif len(lines) > 2:
            lines = [lines[0], lines[1][:-3] + "..."]
        for i, line in enumerate(lines):
            draw.text((18, 463 + 28 * i), line, font=self.font, fill="#eef3fc")
        if own is not None:
            own_conf = (prediction or {}).get("own_confidence")
            same = " ".join(own.split()) == " ".join(sentence.split())
            own_text = "model's own note: " + textwrap.shorten(" ".join(own.split()) or "(empty)", width=52, placeholder="...")
            if own_conf is not None:
                own_text += "  ({:.0%})".format(own_conf)
            draw.text((18, 494), own_text, font=self.font, fill="#8fd9a8" if same else "#f3c67a")
        confidence = (prediction or {}).get("subtask_confidence")
        conf = "—" if confidence is None else "{:.1%}".format(confidence)
        detail = "Step {}   |   Confidence {}   |   Memory writes {}   |   {}".format(
            step, conf, (prediction or {}).get("writes", 0), self.checkpoint)
        draw.text((18, 531), detail, font=self.small, fill="#9caec6")
        draw.text((18, 560), "Goal: " + textwrap.shorten(goal, width=98, placeholder="..."), font=self.small, fill="#9caec6")
        if self.writer is None:
            import imageio.v2 as imageio
            options = dict(fps=self.fps, codec="libx264", pixelformat="yuv420p", macro_block_size=2,
                           output_params=["-movflags", "+faststart"], quality=8)
            self.writer = imageio.get_writer(str(self.directory / "rollout.mp4"), **options)
            self.raw_writer = imageio.get_writer(str(self.directory / "cameras.mp4"), **options)
            canvas.save(self.directory / "poster.jpg")
        self.writer.append_data(np.asarray(canvas))
        self.raw_writer.append_data(np.hstack([np.asarray(im) for im in images]))
        self.count += 1

    def close(self):
        for writer in (self.writer, self.raw_writer):
            if writer is not None:
                writer.close()
