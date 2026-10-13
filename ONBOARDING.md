# Onboarding checklist

Work through this in your first week. The same steps are on the Week 1 cards on the Trello board (setup by Sunday, Oct 18; subteam reading by Wednesday, Oct 21). Ask in the team channel if anything is blocked for more than a day.

## Accounts and access (day 1)

- [ ] Send your GitHub username to Nash to be added to the `gt-big-data` organisation
- [ ] Accept the invite and confirm you can see https://github.com/gt-big-data/fish-cast
- [ ] Join the team channel and the Trello board
- [ ] Open the FishCast Data folder: https://drive.google.com/drive/folders/1bZsC04b6aWKsyWe3-XNH_PEvlAaAhTcC
- [ ] Request PACE cluster access under the `paceship-fishcast` account (ask Nash to sponsor you)
- [ ] Log in to PACE once and check you can run `squeue -u $USER`
- [ ] Analysis and Platform only: create a free Copernicus Climate Data Store account and set up your API key (https://cds.climate.copernicus.eu/how-to-api)

## Set up your environment (day 1-2)

- [ ] Install Python 3.10 or newer and Git
- [ ] Clone the repo and follow the Quick start in `README.md`
- [ ] Run `pytest` and confirm the tests pass
- [ ] Download `krillcast_merged.csv` from the FishCast Data folder into `data/` (see `data/README.md`)
- [ ] Train one small model: `python scripts/train_fishcast.py --model-type climate_conditioned_gp --epochs 5 --output-dir artifacts/test`
- [ ] Open `artifacts/test/metrics.json` and find `test_metrics` and `test_baseline`

## Understand the project (days 2-4)

- [ ] Read the semester plan, especially sections 1, 2 and your own subteam's section
- [ ] Read `docs/CHANGES_preprocessing_v2.md`
- [ ] Work through your subteam's Week 1 card on Trello (reading, checklist and "Check yourself" questions)
- [ ] Look through `fishcast/data.py` and find where hotspots are defined

## First contribution (week 2)

- [ ] Pick a yellow card for your subteam on the Trello board and join it
- [ ] Create a branch, make the change, open a pull request following `CONTRIBUTING.md`
- [ ] Move the card to "In review" and paste the pull request link in a comment
