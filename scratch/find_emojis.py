import re
import unicodedata
from pathlib import Path

dashboard_path = Path(r"C:\Users\NEAL\AXIS_OJT\src\axis_talent_intelligence\frontend\dashboard.html")
content = dashboard_path.read_text(encoding="utf-8")
lines = content.splitlines()

for i, line in enumerate(lines, 1):
    non_ascii = [c for c in line if ord(c) > 127]
    if non_ascii:
        has_emoji = False
        for c in non_ascii:
            cat = unicodedata.category(c)
            if cat in ('So', 'Sk'):
                has_emoji = True
                break
        if has_emoji:
            # Safe print using backslashreplace
            print(f"Line {i}: {line.strip().encode('ascii', 'backslashreplace').decode('ascii')}")
