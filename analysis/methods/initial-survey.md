# Initial methods survey: alternatives to the Gaussian process

- **Date:** 2026-10-27
- **Purpose:** Our current models are Gaussian processes (GPs). Before we spend the rest of the semester on them, we should investigate other models that could predict krill density, as well as their potential benefits and drawbacks.
- **Our data:** Our data is composed of 12,544 net hauls from KRILLBASE. While this seems like a large data pool, about 36% hauls caught no krill at all. This makes sampling patchy: some places and years have lots of hauls, others have almost none.
- **Sources:** Elith, Leathwick & Hastie (2008), *A working guide to boosted regression trees*; the sdmTMB introduction and related guides.
- **Audience:** Analysis subteam and Sahil. 

## Key problems

Three things make modeling krill desnsity difficult:

1. **Lots of zeros.** Over a third of hauls catch nothing. Many common models assume the answer is a smooth number that can be a bit above or below the true value. Exact zeros are not outliers in that setup; they are a pile of mass the model is not built to produce.
2. **Nearby hauls are not independent.** Two nets a few kilometers apart tend to catch similar amounts because they share similar habitats. 
3. **We care about the future.** We care about extrapolation, or predicting what will happen years, or even decades from now. Trees and spatial memory of old hotspots are bad at this.

**What we have currently.** The GP looks at coordinates and time and assumes similar krill data for similar points. Climate (temperature, ice, and so on) is added as a linear adjustment, and density is treated as a Gaussian error after a `log1p` transform. A separate hotspot score is built from the same leftover spatial pattern. While this is a reasonable engineering setup, it is a poor match for abundance of zeros, and the climate part is a linear regression, which may be oversimplifying things.

Below are some alternative approaches that we can use, as well as evaluations on their benefits/disadvantages at tackling the aforementioned three main difficulties with modleing krill.

## Approach A: boosted regression trees (BRT)

A decision tree splits the data with yes/no questions (is SST above some cutoff? is ice cover low?) and ends each path with a simple guess. One tree is easy to read but usually not very accurate. Boosting grows many small trees in sequence, each one focused on the previous trees' mistakes, then adds them together with a small weight on each. Elith et al. (2008) walk through this for ecology; their eel example used about 13,000 sites, similar in size to KRILLBASE. Trees pick useful variables on their own, fit nonlinear relationships, and can pick up combinations like warm water and low ice without us specifying that by hand.

1. **Zeros.** We can train one tree for presence (was any krill caught?) and a second for how many, given that some were caught. That two-step setup is called a hurdle. Fitting a single tree as if density were Gaussian would repeat the GP's mistake.
2. **Nearby hauls.** BRT treats each haul as independent. Including lat/lon as inputs helps it copy the sampling map, but it has no leftover spatial process after climate is accounted for. In heavily sampled areas (e.g. the Peninsula) it will act more confident than it should. Lat/lon also lets it memorize places that were sampled in both train and test years, so a year split is not a clean test of climate skill.
3. **The future.** This is the main drawback. Tree splits are cutoffs, so if 2050 SST is higher than anything in the training data, the prediction stays at whatever it learned for the warmest water it has already seen. The same issue hits new combinations (warm and ice-free together) even when each variable on its own is in range. Elith's maps fill in unsampled sites inside climates that were already observed, which is not the same as a new climate.

BRT is still worth running. It is fast, and partial-dependence plots show the typical shape of density vs ice (or SST) after averaging over the other variables, which would tell us what the GP's linear climate term is missing. Score it on the climate track (no lagged krill), not as a 2050 model.

## Approach B: spatial GLMMs with sdmTMB

A GLM is a regression that can use a noise model matched to the data (yes/no, counts, positive amounts) instead of always assuming Gaussian errors. A GLMM adds unobserved terms; here that would be a smooth map of patches that are consistently higher or lower than climate alone would suggest.

sdmTMB is an R package for this kind of model on survey tows. The intro tutorial uses Pacific cod biomass from trawls with many empty tows, which is close to our hauls. You start with a plain regression, add a spatial map, optionally let that map change by year, and pick a noise model that allows zeros.

Two noise models matter for us:

- **Tweedie:** one model that allows exact zeros and a spread of positive densities.
- **Delta / hurdle (e.g. delta-gamma):** probability of catching anything, then how much you catch when you catch something. This is the cleaner version of our density + hotspot heads. It is a better fit if presence and amount have different causes (ice for recruitment vs temperature/food for swarm density).

1. **Zeros.** Tweedie or a hurdle, as above. This is the main upgrade over our Gaussian-on-log density GP. Tweedie is the easier first fit (one model, fewer things to go wrong). A hurdle is worth it if we think presence and amount have different drivers.
2. **Nearby hauls.** Nearby points share a leftover spatial map, so a cluster of hauls is not treated as many independent discoveries. Patchy sampling is handled by predicting onto a grid rather than pretending empty ocean was sampled. A single global stretch of the correlation ellipse will not follow the ACC around Antarctica; that is weaker than our GP's location-dependent kernel. Coordinates need a south-polar projection in km, not raw lat/lon.
3. **The future.** Temperature and ice enter as a formula (linear or a smooth curve), so they can keep changing the prediction past historical values. That is only as good as the formula. A persistent spatial map still applies in 2050 at the same places, so historical hotspots can get copied forward even if the climate has changed. Year-to-year maps that remember last year tend to drift back toward the historical average. For CMIP6, report the climate-driven part separately, and put static habitat (shelf, depth) in the formula if we have it, so the leftover map is doing less of the work.

Compared with our GP, we give up some flexibility in how "nearby" is defined. In return we get a noise model that matches empty nets and uncertainty that respects clustering. 12,544 points is fine for a spatial-only model. A separate spatial map for each of ~90 years over the whole Southern Ocean is not a week-9 starting point.

## Our GP on the same three points

1. **Zeros.** Weakest part. Gaussian error around `log1p` density does not naturally produce 36% exact zeros.
2. **Nearby hauls.** Strongest part. This is what the GP is built for.
3. **The future.** Mixed. The climate line can go past historical temperature, but spatial memory sits on historical haul locations and fades with time, so it will not invent a new climate response. If climate-only skill stays near zero after the longitude-join fix, the linear climate term is the first thing to question, not the spatial kernel.

## What to prototype first

1. **sdmTMB (Tweedie first, then a hurdle).** This should be the week 9 prototype. It is R, so we export predictions and score them with `scripts/benchmark_skill.py` on the climate track (no lagged krill), using the same train / 1998–2004 val / 2004–2016 test years. Compare on `log1p` density so the RMSE matches the GP table. Include temperature and ice, use a polar mesh, turn a single spatial map on (not year-by-year maps to start), and report climate-only predictions separately from climate + spatial map. Future warming maps need the first of those. If this cannot beat a historical average and a simple climate regression, climate skill is still missing; that could be the covariates, the year split, or a misspecified formula, not proof that the GP kernel is irrelevant.

2. **BRT hurdle as a helper, not the 2050 model.** One presence model and one "how many when present" model, using ice, temperature, and coordinates. Use it to inspect interactions and curve shapes. Do not use it for 2050 maps.

3. **Keep the GP.** If sdmTMB wins mainly because of how it handles zeros, a later experiment is to give the GP a hurdle-style noise model. That is more work and less standard, so it should not be the first prototype.

## Optional biology check

Winter sea ice is the nursery; many KRILLBASE hauls are in summer, when adults can be in open water. After we fit any of these models, check whether high density is being driven by winter ice (or lagged ice) vs just "this is a place we always sampled." Warm, ice-free summer cells are not automatically wrong. That check does not require building a full population model this semester.

## Recommendation

Prototype sdmTMB first (Tweedie, spatial-only, climate track). Use BRT to inspect relationships. Use winter ice vs sampling geography as a check on future maps. Keep the GP as the main approach only if it matches or beats sdmTMB on held-out years, and if its climate-only maps actually move with ice and temperature rather than with the old haul pattern.
