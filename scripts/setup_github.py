"""Set up GitHub for the FishCast team: labels, milestones, issues and a project board.

Run once from the repository root after pushing it:

    gh auth login
    gh auth refresh -s project          # lets gh create the project board
    python scripts/setup_github.py      # add --dry-run to preview

Safe to re-run: existing labels, milestones, issues (matched by title) and the
project (matched by title) are reused, not duplicated.

Options:
    --repo OWNER/NAME      default gt-big-data/fish-cast
    --start-date YYYY-MM-DD  Monday of week 1; sets milestone due dates
    --protect-main         require CI and one review before merging to main (needs admin)
"""
import argparse
import json
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

LABELS = {
    "analysis": ("0f7a75", "Analysis subteam"),
    "platform": ("4353b3", "Platform subteam"),
    "viz": ("a8650a", "Data Visualization subteam"),
    "good first issue": ("7057ff", "Small, well-scoped first task"),
    "optional": ("cfd3d7", "Nice to have; do if time allows"),
    "experiment": ("d4552b", "A modelling experiment to run and benchmark"),
    "bug": ("d73a4a", "Something is wrong"),
    "blocked": ("b60205", "Waiting on something else"),
}
MILESTONES = [
    ("Phase 1: Establish skill (weeks 1-4)", 4, "Ends with Demo 1: benchmark rebuilt with one command."),
    ("Phase 2: Test new inputs (weeks 5-9)", 9, "Ends with Demo 2: new inputs tested against the baselines."),
    ("Phase 3: Communicate (weeks 10-14)", 14, "Ends with the final showcase."),
]
SUBTEAMS = {"analysis": "Analysis", "platform": "Platform", "viz": "Data Viz"}
PROJECT_TITLE = "FishCast Fall 2026"

DRY = False


def gh(*args, parse=False, check=True):
    cmd = ["gh", *args]
    if DRY and _is_write(args):
        print("DRY:", " ".join(a if " " not in a else repr(a) for a in cmd)[:200])
        return {} if parse else ""
    out = subprocess.run(cmd, capture_output=True, text=True)
    if check and out.returncode != 0:
        sys.exit(f"gh {' '.join(args[:3])} failed:\n{out.stderr.strip()}")
    return json.loads(out.stdout or "null") if parse else out.stdout.strip()


def _is_write(args):
    if args[0] == "api":
        return any(a in {"-X", "--method", "-f", "-F"} for a in args)
    if args[0] in {"label", "issue", "project"}:
        return args[1] in {"create", "edit", "item-add", "item-edit", "field-create", "link"}
    return False


def main():
    global DRY
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="gt-big-data/fish-cast")
    p.add_argument("--tasks", default=str(Path(__file__).with_name("tasks.json")))
    p.add_argument("--start-date")
    p.add_argument("--protect-main", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    DRY = a.dry_run
    repo, owner = a.repo, a.repo.split("/")[0]
    tasks = json.load(open(a.tasks))

    print("Labels")
    for name, (color, desc) in LABELS.items():
        gh("label", "create", name, "--repo", repo, "--color", color, "--description", desc, "--force")

    print("Milestones")
    existing = {m["title"]: m["number"] for m in (gh("api", f"repos/{repo}/milestones?state=all&per_page=100", parse=True) or [])}
    start = date.fromisoformat(a.start_date) if a.start_date else None
    for title, week, desc in MILESTONES:
        if title in existing:
            continue
        args = ["api", f"repos/{repo}/milestones", "-f", f"title={title}", "-f", f"description={desc}"]
        if start:
            due = start + timedelta(weeks=week, days=-2)  # Saturday of that week
            args += ["-f", f"due_on={due.isoformat()}T23:59:00Z"]
        gh(*args)

    print("Issues")
    have = {i["title"]: i["url"] for i in (gh("issue", "list", "--repo", repo, "--state", "all", "--limit", "500",
                                               "--json", "title,url", parse=True) or [])}
    urls = {}
    for t in tasks:
        if t["title"] in have:
            urls[t["title"]] = have[t["title"]]
            continue
        args = ["issue", "create", "--repo", repo, "--title", t["title"], "--body", t["body"], "--milestone", t["milestone"]]
        for lab in t["labels"]:
            args += ["--label", lab]
        urls[t["title"]] = gh(*args) or f"(dry-run) {t['title']}"
        print("  created:", t["title"])

    print("Project board")
    projects = (gh("project", "list", "--owner", owner, "--format", "json", parse=True) or {}).get("projects", [])
    proj = next((x for x in projects if x["title"] == PROJECT_TITLE), None)
    if not proj:
        proj = gh("project", "create", "--owner", owner, "--title", PROJECT_TITLE, "--format", "json", parse=True) or {"number": 0, "id": ""}
    num, pid = str(proj["number"]), proj["id"]
    gh("project", "link", num, "--owner", owner, "--repo", repo, check=False)

    fields = (gh("project", "field-list", num, "--owner", owner, "--format", "json", parse=True) or {}).get("fields", [])
    names = {f["name"] for f in fields}
    if "Subteam" not in names:
        gh("project", "field-create", num, "--owner", owner, "--name", "Subteam", "--data-type", "SINGLE_SELECT",
           "--single-select-options", ",".join(SUBTEAMS.values()))
    if "Week" not in names:
        gh("project", "field-create", num, "--owner", owner, "--name", "Week", "--data-type", "NUMBER")
    fields = (gh("project", "field-list", num, "--owner", owner, "--format", "json", parse=True) or {}).get("fields", [])
    byname = {f["name"]: f for f in fields}

    if not DRY:
        for t in tasks:
            item = gh("project", "item-add", num, "--owner", owner, "--url", urls[t["title"]], "--format", "json", parse=True)
            sub = byname["Subteam"]
            opt = next(o["id"] for o in sub["options"] if o["name"] == SUBTEAMS[t["team"]])
            gh("project", "item-edit", "--id", item["id"], "--project-id", pid, "--field-id", sub["id"],
               "--single-select-option-id", opt)
            gh("project", "item-edit", "--id", item["id"], "--project-id", pid, "--field-id", byname["Week"]["id"],
               "--number", str(t["week"]))
    print(f"  {len(tasks)} issues on the board")

    if a.protect_main:
        print("Branch protection on main")
        body = json.dumps({
            "required_status_checks": {"strict": True, "contexts": ["test"]},
            "enforce_admins": False,
            "required_pull_request_reviews": {"required_approving_review_count": 1},
            "restrictions": None,
        })
        if DRY:
            print("DRY: PUT branch protection", body)
        else:
            subprocess.run(["gh", "api", "-X", "PUT", f"repos/{repo}/branches/main/protection", "--input", "-"],
                           input=body, text=True, check=True)

    print("\nDone. Next: open the project, add a Board view grouped by Status, and assign issues to people.")
    print(f"https://github.com/orgs/{owner}/projects/{num}")


if __name__ == "__main__":
    main()
