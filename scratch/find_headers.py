from pathlib import Path

dashboard_path = Path(r"C:\Users\NEAL\AXIS_OJT\src\axis_talent_intelligence\frontend\dashboard.html")
content = dashboard_path.read_text(encoding="utf-8")
lines = content.splitlines()

targets = [
    "telemetry",
    "console",
    "inside-out",
    "optimization"
]

for i, line in enumerate(lines, 1):
    for target in targets:
        if target in line.lower():
            print(f"Line {i}: {line.strip()}")
