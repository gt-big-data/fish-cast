# Initial methods survey: alternatives to the Gaussian process

- **Date:** 2026-10-27
- **Purpose:** Our current models are Gaussian processes (GPs). Before we spend the rest of the semester on them, this note asks: what else could predict krill density, and why might it do better or worse on our data?
- **Our data:** 12,544 net hauls from KRILLBASE. About 36% caught no krill at all. Sampling is patchy: some places and years have lots of hauls, others almost none.
- **Sources:** Elith, Leathwick & Hastie (2008), *A working guide to boosted regression trees*; the sdmTMB introduction and related guides.
- **Audience:** Analysis subteam and Sahil. This is meant to feed the week 9 methods prototype.

## The problem in plain language

A model is a recipe that takes inputs (location, year, sea temperature, sea ice, …) and outputs a guess for krill density.

Three things make this hard:

1. **Lots of zeros.** A third of hauls catch nothing. Many common models assume the answer is a smooth number that can be a bit above or below the truth. They do not expect a big pile of exact zeros.
2. **Nearby hauls are not independent.** Two nets a few kilometers apart tend to catch similar amounts, because they share habitat we did not fully measure. If we treat every haul as a brand-new fact, we will be overconfident.
3. **We care about a warmer future.** Predicting “what was krill density in a sampled bay in 2005?” is interpolation (filling gaps inside what we have seen). Predicting “what happens if the water is warmer than anything in the training data?” is extrapolation (stepping outside what we have seen). Trees and spatial “memory” of old hotspots are bad at that second job. Climate variables (temperature, ice) have to do the work.

**What we do now.** The GP looks at coordinates and time and says “nearby, similar-time points should have similar leftover krill after climate is accounted for.” Climate (temperature, ice, and so on) is added as a straight-line adjustment. Density is treated as a bell-shaped (Gaussian) error around that guess, after a `log1p` transform (log of one plus density). A separate “hotspot” score is built from the same leftover spatial pattern. That is a reasonable engineering setup. It is a poor match for the pile of zeros, and the climate part is a simple straight line.

The question is not “should we throw the GP away?” It is: which other approach is most likely to beat it on years we held out, and to make maps under future climate (CMIP6) that we can trust.

## Approach A: boosted regression trees (BRT)

Think of a **decision tree** as a flowchart of yes/no questions: “Is summer temperature above 2°C? If yes, is ice cover low? …” Each path ends in a simple guess (average density, or probability krill were present). One tree is easy to read but usually not very accurate.

**Boosting** means: grow a small tree, look at the mistakes, grow another tree that focuses on those mistakes, repeat hundreds or thousands of times, and add the trees together (each one only allowed a small vote). Elith et al. (2008) walk through this for ecology. Their eel example used about 13,000 sites — similar in size to KRILLBASE. Trees automatically pick useful variables, bend the relationship instead of forcing a straight line, and can capture combinations like “warm *and* low ice” without us writing that by hand.

| Hard part | How BRT handles it |
| --- | --- |
| **Zeros** | We can train a tree to predict “was any krill caught?” (yes/no) and a second tree to predict “if yes, how many?” That two-step setup is called a **hurdle**. Training one tree as if density were a smooth bell curve repeats the GP’s mistake. |
| **Nearby hauls** | BRT treats each haul as independent. We can include latitude and longitude as inputs, which helps the model copy the sampling map, but it does not have a real model of “left-over similarity after climate.” Where many hauls sit on top of each other (for example the Peninsula), the model will think it has more information than it does. |
| **Warmer future** | This is the big weakness. A tree’s last question is a cutoff. If we have never seen SST as high as 2050, the tree does not keep the trend going; it stays at the guess for the warmest water it *has* seen. Elith’s maps fill in unsampled rivers *inside* the climates already observed. That is not the same as a new climate. |

**Still useful.** BRT is fast. Partial-dependence plots (Elith’s main practical trick) show the typical shape of “density vs ice” after averaging over other variables. That can tell us which bends and combinations our GP’s straight climate line is missing. It is a fair baseline for “can we predict held-out *historical* years?” It is a weak tool for 2050 maps.

## Approach B: spatial GLMMs with sdmTMB

A **GLM** is a regression model that can use the right kind of noise for the data (yes/no, counts, positive amounts, …) instead of always assuming a bell curve.

A **GLMM** adds extra terms that are not fully observed. Here those terms are a smooth map of “this patch of ocean is usually higher or lower than climate alone would say.”

**sdmTMB** is an R package that fits that kind of model for survey tows. The intro tutorial uses Pacific cod biomass from trawls with many empty tows — the same shape as our hauls. You start with a plain regression, then add a spatial map, then optionally a map that can change by year, then pick a noise model that allows zeros.

Two noise models matter for us:

- **Tweedie:** one formula that allows exact zeros *and* a spread of positive densities.
- **Delta / hurdle (for example delta-gamma):** first, probability of catching anything; second, how much you catch when you catch something. That is the statistically careful version of our density + hotspot heads. Use this if presence and amount have different causes (for example ice for “are juveniles around?” vs temperature/food for “how dense is the swarm?”).

| Hard part | How sdmTMB handles it |
| --- | --- |
| **Zeros** | Tweedie or a hurdle, as above. This is the main upgrade over our Gaussian-on-log density GP. |
| **Nearby hauls** | This is what the package is for. Nearby points share a smooth leftover map, so a cluster of hauls is not counted as 50 independent discoveries. We can also let similarity stretch more in one direction than another (useful if the Antarctic Circumpolar Current smears krill along a path). Patchy sampling is handled by predicting onto a regular grid, not by pretending empty ocean was sampled. |
| **Warmer future** | Temperature and ice can keep changing the prediction beyond historical values, because they enter as a formula (a line, or a smooth curve), not as a tree cutoff. That can be right or dangerously wrong if the line is a bad description of biology. The leftover spatial map does **not** know about 2050: far from past hauls it fades toward “no extra hotspot.” Year-to-year leftover maps that remember last year tend to drift back toward the historical average. So for CMIP6 we should read the **climate-driven** part of the prediction, and treat the leftover map as “habitat we did not measure,” not as a future bloom. |

**Trade-off vs our GP.** Our GP tries to learn a flexible, location-dependent idea of “what nearby means” (including stretching along currents). sdmTMB uses a more standard, smoother leftover map. We give up some of that flexibility. We gain a noise model that matches empty nets, uncertainty that respects clustering, and a normal scientific workflow for maps. 12,544 points is a comfortable size for this package.

## Our Gaussian process, in the same three columns

- **Zeros:** weakest part. A bell-shaped error around `log1p` density does not naturally produce 36% exact zeros.
- **Nearby hauls:** strongest part. That is what the GP is built to do.
- **Warmer future:** mixed. The climate line *can* go past historical temperature. The spatial “memory” sits on historical haul locations and fades with time, so it will not invent a new climate response. If climate-only skill stays near zero after the longitude-join data fix, suspect the straight climate line first, not the spatial part.

## What I would prototype first

**1. sdmTMB (try hurdle first, then Tweedie).** This is the week 9 prototype. Train on the same early years we already use. Check on 1998–2004, then score once on 2004–2016, with the same skill table as the GPs. Include temperature and ice. Turn the leftover spatial map on. Report two predictions: climate-only, and climate + leftover map. Future warming maps need the first of those. If this cannot beat “always guess the historical average” and a simple climate regression, the problem is the climate data or the year split — not the GP’s fancy spatial kernel.

**2. BRT hurdle as a helper, not the 2050 model.** One yes/no model and one “how many when present” model, using ice, temperature, and coordinates. Use it to see interactions and curve shapes, and as a ceiling for “how well can we do if we ignore leftover spatial clustering?” Do not use it for 2050 maps.

**3. Keep the GP.** If the sdmTMB model wins mainly because of zeros, a later experiment is to give the GP a hurdle-style noise model. That is more work and less standard; it should not be the first prototype.

## Optional biology check (not a third ML model)

Krill recruitment is tied to winter sea ice, and growth cares about temperature. After we fit any statistical model, look at cells that are warm and ice-poor. If the model still paints those as hotspots, it is copying *where we used to sample*, not responding to climate. That check does not require building a full population simulator this semester.

## Recommendation for discussion with Sahil

Build **sdmTMB** first. Use **BRT** to inspect relationships and as a historical baseline. Use a simple **ice / growth story** as a smell test on future maps. Keep the **GP** as the main codebase only if it matches or beats the sdmTMB model on the held-out years *and* its climate-only maps move with ice and temperature, not with the old haul pattern.
