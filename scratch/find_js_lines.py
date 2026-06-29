from pathlib import Path

dashboard_path = Path(r"C:\Users\NEAL\AXIS_OJT\src\axis_talent_intelligence\frontend\dashboard.html")
content = dashboard_path.read_text(encoding="utf-8")
lines = content.splitlines()

for i, line in enumerate(lines, 1):
    if "executeBatchMatchmaking" in line:
        print(f"Line {i}: {line.strip()}")
