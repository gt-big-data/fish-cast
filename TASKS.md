# How tasks work

All work is tracked as GitHub Issues on the **FishCast Fall 2026** project board (Projects tab of the `gt-big-data` organisation).

## Board columns

| Status | Meaning |
| --- | --- |
| Todo | Planned, nobody working on it yet |
| In Progress | Assigned and being worked on; a branch or draft PR exists |
| Done | Merged or otherwise finished |

Each card also has **Subteam** (Analysis, Platform, Data Viz), **Week** (planned week of the semester) and a **milestone** (the phase it belongs to).

## For leads: sending out a task

1. Open the issue and set **Assignees** to the person. GitHub emails them automatically.
2. Move the card to **In Progress** once they confirm.
3. If the task needs more detail, edit the issue body. Keep a clear "Done when" line.
4. New work: open an issue from a template (Task, Experiment or Bug). It lands on the board if you add it to the project in the right-hand panel.

Useful board views (create once with the + next to the view tabs):

- **Board** grouped by Status, filtered `milestone:"Phase 1: Establish skill (weeks 1-4)"` for the current phase
- **By person**: Table grouped by Assignees
- **By subteam**: Board grouped by Subteam

## For members: picking up a task

1. Check **Assigned to me** on the board (or https://github.com/issues/assigned).
2. Create a branch: `git checkout -b <subteam>/<short-description>`.
3. Open a pull request with `Closes #<issue number>` in the description. Merging it closes the issue and moves the card to Done.
4. Stuck for more than a day? Add the `blocked` label and say why in a comment.
