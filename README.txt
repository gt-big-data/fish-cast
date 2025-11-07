Project Repository Structure
=============================

This repository is organized for collaboration between three teams: Processing, Analysis, and Visualization.
Each team works on separate branches and contributes to one shared development flow.
The goal is to keep collaboration simple, prevent merge conflicts, and maintain a consistent structure.

------------------------------------------------------------

Branch Overview
================

main
│
└── dev
     ├── processing/
     │    ├── nash
     │    ├── vedant
     │    ├── rohan
     │    ├── amulya
     │    └── moiz
     │
     ├── analysis/
     │    ├── baseline-models
     │    ├── metrics
     │    └── forecasting
     │
     └── viz/
          ├── dashboard
          ├── api
          └── ui-improvements

------------------------------------------------------------

Branch Descriptions
===================

main
    The stable release branch used for production.
    Only merged into after full testing and approval.

dev
    The shared integration branch where all teams combine work before release.

processing, analysis, viz
    Parent branches for each team’s work area.

processing/nash, processing/vedant, processing/rohan, processing/amulya, processing/moiz
    Personal branches for each member of the Processing team.
    Each person commits and tests their code here before merging into the shared processing branch.

------------------------------------------------------------

Repository Folder Layout
========================

/processing
│
├── scripts/
│   ├── data_cleaning.py
│   ├── ingestion.py
│   ├── vegetation_filter.py
│   ├── geo_parser.py
│   └── db_pipeline.py
│
├── tests/
│   ├── test_data_cleaning.py
│   ├── test_ingestion.py
│   └── test_geo_parser.py
│
└── README.txt

scripts/
    Contains Python scripts for data ingestion, cleaning, and transformation.

tests/
    Contains unit tests for the main scripts.

README.txt
    Explains the structure and workflow of the Processing module.

------------------------------------------------------------

Visual Workflow Diagram
=======================

             +-----------------------+
             |      main (stable)    |
             +-----------^-----------+
                         |
               +---------+----------+
               |       dev (shared) |
               +---------^----------+
                         |
         +---------------+---------------+
         |               |               |
     +---+---+       +---+---+       +---+---+
     |processing|    |analysis|      |  viz  |
     +---^---+       +---^---+       +---^---+
         |               |               |
 +-------+---------+     |               |
 | nash  vedant    |     |               |
 | rohan amulya    |     |               |
 | moiz            |     |               |
 +-----------------+     |               |
         |               |               |
         +--------------> dev ----------> main

Each Processing member commits to their own branch.
Those branches merge into processing/, which later merges into dev.
After integration testing, dev is merged into main for release.

------------------------------------------------------------

Workflow Summary
================

1. Pull the latest changes from dev:
   git checkout dev
   git pull origin dev

2. Create or switch to your personal branch:
   git checkout -b processing/nash

3. Make changes in /processing/scripts
   Test your updates locally.

4. Stage and commit your work:
   git add .
   git commit -m "Add new ingestion module"

5. Push changes to your remote branch:
   git push origin processing/nash

6. Open a pull request into the processing branch for review.

7. After approval, the Processing lead merges changes into processing/,
   then processing/ merges into dev, and finally dev merges into main.

------------------------------------------------------------

Permissions
===========

Branch              Who Can Merge           Review Required
------------------------------------------------------------
main                Project leads           Yes
dev                 Team leads              Yes
processing/         Processing lead         Yes
processing/<name>   Individual contributor  Optional

------------------------------------------------------------

Guidelines
===========

- Keep your branch updated with processing/ before starting new work.
- Write concise and descriptive commit messages.
- Run tests before submitting a pull request.
- Document all major functions and scripts.
- Use consistent naming for files and variables.
- Only merge into higher branches after review and testing.

------------------------------------------------------------

Summary
=======

This structure allows each team and individual to work independently while maintaining a clear and reliable integration process.
Each Processing member owns a single branch, which merges into the shared processing branch, then into dev, and finally into main once tested and approved.

------------------------------------------------------------

End of README.txt
