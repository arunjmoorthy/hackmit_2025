from screen_record import ScreenRecorder
import time

rec = ScreenRecorder(out_path="runs/screen_test.mp4", fps=30, display="auto", audio=None)
rec.start()
time.sleep(3)        # wiggle the mouse on screen to see cursor captured
rec.stop()