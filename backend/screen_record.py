import subprocess, sys, os, time
from pathlib import Path
import imageio_ffmpeg as ffmpeg

class ScreenRecorder:
    def __init__(self,
                 out_path="screen_capture.mp4",
                 fps=30,
                 display="auto",      # "auto" or int index (mac only)
                 region=None,         # (x, y, w, h) or None
                 audio=None,          # mac: None or int audio idx; win/linux: device name (optional)
                 overwrite=True,
                 platform="auto"):
        self.out_path = str(out_path)
        self.fps = int(fps)
        self.display = display
        self.region = region
        self.audio = audio
        self.overwrite = overwrite
        self.platform = self._detect_platform(platform)
        self.proc = None
        self.ffmpeg_path = ffmpeg.get_ffmpeg_exe()

    def _detect_platform(self, platform):
        if platform != "auto":
            return platform
        if sys.platform.startswith("darwin"):
            return "mac"
        if os.name == "nt":
            return "win"
        return "linux"

    # ---------------------
    # macOS: device helpers
    # ---------------------
    def _list_avfoundation_devices(self):
        """
        Returns (video_devices, audio_devices) where each is a list of (idx:int, name:str).
        Uses the bundled ffmpeg, parses stderr (where ffmpeg prints device list).
        """
        proc = subprocess.run(
            [self.ffmpeg_path, "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        txt = (proc.stdout or "") + (proc.stderr or "")
        vids, auds, mode = [], [], None
        for line in txt.splitlines():
            if "AVFoundation video devices" in line:
                mode = "video"; continue
            if "AVFoundation audio devices" in line:
                mode = "audio"; continue
            if mode and "[" in line and "]" in line:
                try:
                    idx = int(line.split("[",1)[1].split("]",1)[0].strip())
                    name = line.split("]",1)[1].strip()
                    if mode == "video": vids.append((idx, name))
                    else: auds.append((idx, name))
                except Exception:
                    pass
        return vids, auds

    def _pick_mac_screen_index(self):
        """
        If display is 'auto', pick the first AVFoundation video device
        whose name contains 'Capture screen'. Otherwise, trust the int.
        """
        if isinstance(self.display, int):
            return self.display
        vids, _ = self._list_avfoundation_devices()
        # Prefer 'Capture screen 0' then any 'Capture screen'
        screen0 = [i for (i, n) in vids if "Capture screen 0" in n]
        if screen0:
            return screen0[0]
        screens = [i for (i, n) in vids if "Capture screen" in n]
        if screens:
            return screens[0]
        # Fallback: 1 is often the screen on many Macs
        return 1

    # ---------------------
    # ffmpeg command build
    # ---------------------
    def _ffmpeg_cmd(self):
        common = [
            self.ffmpeg_path,
            "-hide_banner",
            "-loglevel", "warning",
            "-y" if self.overwrite else "-n",
            "-framerate", str(self.fps),
        ]

        vf = []
        if self.region:
            x, y, w, h = self.region
            vf.append(f"crop={w}:{h}:{x}:{y}")
        # Keep even dims for H.264 (applied last after optional crop)
        vf.append("scale=trunc(iw/2)*2:trunc(ih/2)*2")

        if self.platform == "mac":
            vid_idx = self._pick_mac_screen_index()
            av_in = f"{vid_idx}"
            if self.audio is not None:
                av_in = f"{vid_idx}:{self.audio}"
            inp = [
                "-f", "avfoundation",
                "-capture_cursor", "1",
                "-capture_mouse_clicks", "1",
                "-i", av_in
            ]

        elif self.platform == "win":
            # Full desktop; use region to restrict capture
            if self.region:
                x, y, w, h = self.region
                inp = [
                    "-f", "gdigrab",
                    "-offset_x", str(x), "-offset_y", str(y),
                    "-video_size", f"{w}x{h}",
                    "-i", "desktop"
                ]
            else:
                inp = ["-f", "gdigrab", "-i", "desktop"]
            if self.audio:
                inp += ["-f", "dshow", "-i", f"audio={self.audio}"]

        else:  # linux
            display_env = os.environ.get("DISPLAY", ":0.0")
            if self.region:
                x, y, w, h = self.region
                inp = [
                    "-f", "x11grab",
                    "-video_size", f"{w}x{h}",
                    "-i", f"{display_env}+{x},{y}"
                ]
            else:
                inp = ["-f", "x11grab", "-i", display_env]
            if self.audio:
                inp += ["-f", "pulse", "-i", self.audio]

        out = [
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "veryfast",
            "-movflags", "+faststart",
            "-vf", ",".join(vf),
            self.out_path,
        ]
        return common + inp + out

    # ---------------------
    # lifecycle
    # ---------------------
    def start(self):
        if self.proc is not None:
            raise RuntimeError("Recorder already running.")
        Path(self.out_path).parent.mkdir(parents=True, exist_ok=True)
        cmd = self._ffmpeg_cmd()
        # Capture stderr for debugging, discard stdout
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        time.sleep(0.4)  # tiny warmup to avoid first-frame hiccup

    def stop(self, timeout=8):
        if self.proc is None:
            return
        try:
            # Graceful stop so MP4 moov atom is written
            if self.proc.stdin:
                try:
                    self.proc.stdin.write(b"q")
                    self.proc.stdin.flush()
                except Exception:
                    pass
                try:
                    self.proc.stdin.close()
                except Exception:
                    pass
        except Exception:
            pass
        # Wait for graceful exit
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                # Try SIGINT for graceful finalize
                self.proc.send_signal(subprocess.signal.SIGINT)
                self.proc.wait(timeout=3)
            except Exception:
                try:
                    self.proc.terminate()
                    self.proc.wait(timeout=2)
                except Exception:
                    try:
                        self.proc.kill()
                    except Exception:
                        pass
        # Optionally surface ffmpeg stderr on error
        try:
            if self.proc.returncode not in (0, None):
                err = self.proc.stderr.read().decode("utf-8", errors="ignore") if self.proc.stderr else ""
                if err:
                    print("[ScreenRecorder] ffmpeg stderr:\n" + err)
        except Exception:
            pass
        self.proc = None
