from pathlib import Path

dashboard_path = Path(r"C:\Users\NEAL\AXIS_OJT\src\axis_talent_intelligence\frontend\dashboard.html")
content = dashboard_path.read_text(encoding="utf-8")
lines = content.splitlines()

chunks_targets = [
    ("Chunk 1 (Activate Simulator)", "activateDemandSimulator(event)", "Activate Simulator"),
    ("Chunk 2 (Talents List)", "Hand off these talents into the matchmaking engine", "Trigger All Matches"),
    ("Chunk 3 (Pipeline Funnel)", "Pipeline funnel", "Movement across AXIS stages"),
    ("Chunk 4 (Compliance Drift)", "Real-Time Corporate Compliance Drift Monitoring", "Drift Logs"),
    ("Chunk 5 (Gemini agent modal)", "🤖 AXIS Intelligence Loop Output", "Continuous inside-out skill"),
    ("Chunk 6 (Agent Lifecycle)", "🤖 Core AI Agent Lifecycle Workspace", "Core AI Agent Lifecycle Workspace"),
    ("Chunk 7 (Run Diagnostics)", "AI Live Telemetry Stream", "⚡ Run Diagnostics"),
    ("Chunk 8 (Talent Detail Modal Admin)", "id=\"detail-admin-panel\"", "Bypass/Edit Path"),
    ("Chunk 9 (Modify Pathway Modal Notes)", "id=\"override-action\"", "Bypass Module (Skip & Mark Mastered)"),
    ("Chunk 10 (Reset Database State)", "Reset Database State", "Security Protocol")
]

for name, start_target, end_target in chunks_targets:
    start_line = None
    end_line = None
    for idx, line in enumerate(lines, 1):
        if start_target in line:
            start_line = idx
        if end_target in line and start_line is not None:
            end_line = idx
            break
    print(f"{name}: Start: {start_line}, End: {end_line}")
