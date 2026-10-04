Status: Draft

# FishCast - KrillCast

## Current Model Family

The repository now implements four named forecasting variants derived from this methodology draft:

* **Climate-Conditioned GP (`climate_conditioned_gp`):** The original direct spatio-temporal Gaussian Process formulation driven by climate covariates and coordinates.
* **Autoregressive GP (`autoregressive_gp`):** The same GP backbone, but with lightweight ecological memory features added to the input space, such as lagged krill density and lagged hotspot state by sector. At forecast time this model can be rolled forward recursively so predicted krill state becomes part of the next-step input.
* **Latent-State Augmented GP (`latent_state_augmented_gp`):** The autoregressive GP extended with a learned latent ecological state derived from lagged krill density and hotspot history. This preserves the GP spatial field while making the temporal memory pathway more expressive than raw lag covariates alone.
* **Autoregressive Sequence Forecaster (`autoregressive_sequence`):** A true recursive forecasting pipeline that aggregates data into sector-year sequences and rolls predictions forward over a configurable forecast horizon using `seq_len` and `horizon_len`.

The first three models preserve the non-stationary GP framing described below. The fourth is an explicit autoregressive extension intended for experiments where prior krill state should directly influence future forecasts.

## Motivation

Antarctic krill (Euphausia superba) are the keystone of the Southern Ocean, acting as the primary food source for millions of predators (whales, penguins, seals) and a massive "biological pump" that traps atmospheric carbon on the ocean floor.

Climate change is rapidly warming the Antarctic, causing sea ice—the critical nursery for juvenile krill—to retreat. This is forcing krill populations to shift their habitats, potentially moving out of fixed Marine Protected Areas [(MPAs)](https://marineprotectedareas.noaa.gov/) and into zones where commercial fishing is unregulated.

The goal of this project is to develop a Spatio-Temporal Gaussian Process (STGP) model that fuses 90 years of historical data [(KRILLBASE)](https://data.bas.ac.uk/full-record.php?id=GB/NERC/BAS/PDC/00915) with future climate projections [(CMIP6)](https://cds.climate.copernicus.eu/datasets/projections-cmip6?tab=overview). By treating the ocean as a dynamic, non-stationary field, we aim to:

* Forecast krill density "hotspots" under future warming scenarios.
* Quantify the "Spillover Risk" where populations migrate beyond current legal boundaries.
* Provide a predictive tool for "Dynamic Management," ensuring international protections move in real-time with the species they are designed to save.

## Mathematical Framework

To predict Antarctic krill (*Euphausia superba*) biomass and the emergence of localized ecological "blooms," we deploy a joint Spatio-Temporal Gaussian Process (STGP) framework. This approach fuses 90 years of historical sampling data [(KRILLBASE)](https://data.bas.ac.uk/full-record.php?id=GB/NERC/BAS/PDC/00915) with forward-looking climate projections [(CMIP6)](https://cds.climate.copernicus.eu/datasets/projections-cmip6?tab=overview), allowing us to track population shifts relative to established Marine Protected Area (MPA) boundaries [(CCAMLRGIS)](https://gis.ccamlr.org/).

### 1. Joint Spatio-Temporal Formulation

We define our spatio-temporal domain as $X = \mathcal{T} \times \mathcal{S}$, where $\mathcal{T}$ represents time (years) and $\mathcal{S} \subset \mathbb{R}^2$ represents geographic coordinates in the Southern Ocean. For any coordinate $x_{it} = (t, s_i) \in X$, the model jointly evaluates two correlated target variables:

* **Continuous Biomass Model ($y_{it}$):** The observed krill density at location $i$ and time $t$, derived directly from KRILLBASE. We model this as:

$$y_{it} = \mu_{it} + f(x_{it}) + \epsilon_{it}$$



Here, $\epsilon_{it} \sim \mathcal{N}(0, \sigma_\epsilon^2)$ accounts for sampling noise. The baseline mean $\mu_{it}$ is a linear function of local environmental drivers—specifically, Sea Surface Temperature (SST) and Sea Ice Concentration extracted from CMIP6.
* **Hotspot Classification Model ($h_{it}$):** A binary variable where $h_{it} = 1$ defines an ecological "bloom" (e.g., density exceeding the historical 90th percentile). The probability of a hotspot is linked to the same latent ecological state via a Bernoulli likelihood with a sigmoid function $\sigma$:

$$p(h_{it} = 1 | f) = \sigma(f(x_{it}))$$



The shared latent variable $f(x)$ encapsulates the unobserved "environmental suitability" of the ocean. We place a zero-mean Gaussian Process prior on this function: $f \sim \mathcal{GP}(0, k(x, x'))$.

### 2. Deep Non-Stationary Kernel (Modeling Ocean Currents)

Standard machine learning models assume spatial stationarity—meaning they assume krill interact the exact same way across a flat grid. In reality, krill distribution is heavily warped by fluid dynamics like the Antarctic Circumpolar Current. To mathematically represent this, we construct a **non-stationary deep neural kernel**:

$$k(x, x') = \nu(t, t') \cdot \sum_{r=1}^R w^{(r)}_{s'} \upsilon^{(r)}(s, s')$$

* **Temporal Component ($\nu$):** A stationary Gaussian kernel that decays over time, capturing the multi-year latency of how sea ice retreat impacts krill life cycles.
* **Spatial Component ($\upsilon$):** This represents the physical ocean. Instead of relying on raw Euclidean distance between two points $s$ and $s'$, the correlation relies on location-specific covariance matrices $\Sigma_s$ and $\Sigma_{s'}$:

$$\upsilon(s, s') \propto |\Sigma_s + \Sigma_{s'}|^{-\frac{1}{2}} \exp \left( -\frac{1}{2} (s' - s)^T (\Sigma_s + \Sigma_{s'})^{-1} (s' - s) \right)$$



**The Neural Network Integration:** To determine how these covariance matrices ($\Sigma_s$) shift across the ocean, we use a deep neural network $\phi(s)$. The network takes a GPS coordinate as input and outputs the geometry of $\Sigma_s$. This allows the model to dynamically "learn" that krill density correlations stretch longitudinally along ocean currents, but compress tightly near ice shelves.

### 3. Scalable Variational Inference

A standard Gaussian Process requires inverting an $N \times N$ covariance matrix, which operates at $\mathcal{O}(N^3)$ complexity. Given the scale of KRILLBASE and the high-resolution grids of CMIP6, exact inference would be computationally intractable.

To bypass this bottleneck, we implement **Sparse Variational Inference**. We introduce a small set of $M$ inducing points $Z = \{z_1, ..., z_M\}$ in the spatio-temporal domain, acting as "summaries" of the broader ocean data, with corresponding inducing variables $u \sim \mathcal{GP}(0, K_{ZZ})$.

We optimize the Evidence Lower Bound (ELBO) using stochastic gradient descent. The learning objective maximizes the joint log-likelihood of both the continuous krill density ($y$) and the binary hotspot occurrences ($h$):

$$\mathcal{L}_{ELBO} = \sum_{n=1}^N \mathbb{E}_{q(f)} [\log p(y_n, h_n | f_n)] - \text{KL}[q(u) || p(u)]$$

This optimization allows the model to process massive climate datasets in mini-batches. Ultimately, it generates a predictive distribution with robust Uncertainty Quantification (confidence intervals), allowing us to cross-reference predicted future hotspots against CCAMLRGIS polygons to confidently assess the risk of krill shifting out of protected zones.


## Computational Pipeline

### 1. Input Data

* **Biological Ground Truth (KRILLBASE):** 90 years of "point data" (e.g., *“On Jan 12, 1995, at this exact Lat/Lon, we caught 500g of krill”*).
* **Environmental Covariates (CMIP6):** Global, high-resolution 3D grids of Sea Surface Temperature (SST), Sea Ice Concentration, and Chlorophyll-a.

### 2. Data Preprocessing & Initialization

* **Spatio-Temporal Join:** KRILLBASE provides discrete data points, while CMIP6 provides continuous global grids. For every single historical krill catch, our pipeline will perform a spatio-temporal lookup to extract the specific Sea Surface Temperature (SST), Sea Ice Concentration, and Chlorophyll-a levels at that exact $(t, latitude, longitude)$ coordinate.
* **Normalization:** The environmental covariates are scaled using standard z-score normalization. This ensures the linear mean module ($\mu_{it}$) does not heavily bias variables with larger raw magnitudes (e.g., percentage of ice cover vs. raw temperature in Celsius).
* **Inducing Point Initialization (The "Buoys"):** To bypass the $\mathcal{O}(N^3)$ computational bottleneck of the Gaussian Process, we establish $M=500$ inducing points ($Z$) to act as a low-rank spatio-temporal summary. To prevent placing these computational anchors in empty, unsampled ocean, we'll run a **K-Means Clustering** algorithm ($K=M$) directly on the KRILLBASE spatio-temporal coordinates.

The resulting 500 cluster centroids serve as the initial coordinates for our inducing points, guaranteeing that our GPU memory is concentrated strictly in historically sampled regions.

* **The Result:** The preprocessing pipeline outputs a clean, aligned master tensor where each row follows the structure: `[Year, Lat, Lon, SST, Ice, Chl] -> [Krill Density]`, alongside the initialized $500 \times 3$ tensor of inducing point coordinates.

### 3. Relevant Formulas

As the data enters the model, it is split into two mathematical paths that merge at the end.

#### Path A: The Deterministic Baseline ($\mu_{it}$)

The model passes the environmental data through a single linear layer:


$$\mu_{it} = w_1(SST) + w_2(Ice) + w_3(Chl) + b$$

* **What it does:** This finds the "average" krill density expected for those weather conditions. If it's too warm, this number drops.

#### Path B: The Latent Suitability ($f(x_{it})$)

The model passes the raw **Coordinates (Lat, Lon, Time)** into the Deep Kernel:


$$f(x_{it}) = K_{x Z} K_{ZZ}^{-1} u$$

* **What it does:** This uses the Deep Neural Network to "bend" the map according to ocean currents. It looks at the $M$ inducing points (the "summary buoys") to see if this location has historically been a hotspot, regardless of what the current temperature is.

### 4. Arriving at the Result (Joint Inference)

The model adds the two paths together to get the final density prediction, and applies a "squish" for the hotspot alarm:

1. **Biomass ($y_{it}$):** $\mu_{it} + f(x_{it}) + \epsilon_{it}$ (The sum of the weather guess + the spatial adjustment + random noise).
2. **Hotspot ($h_{it}$):** $\sigma(f(x_{it}))$ (The spatial adjustment passed through a sigmoid to get a 0–100% probability).

### 5. The Final Output: The "Risk Map"

The result is a **Predictive Distribution**.

* **The Mean:** Our "best guess" for krill density in the year 2050.
* **The Variance (Uncertainty):** A measure of how much we *don't* know. If CMIP6 shows a wild range of possible temperatures for 2050, our variance grows.
* **Final Decision:** We overlay our high-probability hotspots ($p > 0.85$) onto the **CCAMLRGIS** shapefiles.

**How we arrive at it:** We arrive at this by optimizing the **ELBO**. The model "guesses" a map, compares it to the KRILLBASE history, sees its mistakes, and uses backpropagation to tweak the weights of the Neural Network and the positions of the Inducing Points until the "Risk Map" matches the historical reality as closely as possible.

### 6. Optimization and Training

The model is trained end-to-end using **mini-batch optimization** to jointly evaluate the continuous biomass regression and the binary hotspot classification.

* **Objective Function:** We minimize a joint loss function comprising the negative Multi-Task Evidence Lower Bound (ELBO) for the continuous spatio-temporal GP and Binary Cross-Entropy (BCE) for the hotspot classification.
* **Trainable Parameters:** During optimization, we jointly update three core components:
1. **Deep Kernel Weights ($\phi$):** Adapting the MLP to map raw geographic coordinates to valid, non-stationary covariance ellipses representing ocean fluid dynamics.
2. **Baseline Coefficients ($\mathbf{w}$):** Fine-tuning the linear mapping of CMIP6 environmental covariates (SST, Ice Concentration, Chlorophyll-a) to the mean density ($\mu_{it}$).
3. **Inducing Points ($Z, \mathbf{u}$):** The continuous spatio-temporal coordinates of the $M$ inducing points are treated as learnable hyperparameters. The optimizer dynamically repositions them to regions of high ecological variance, maximizing the resolution of the sparse approximation.


* **Training Strategy:** We utilize the **Adam optimizer**, processing the historical data in small mini-batches. This batching strategy, enabled by the variational ELBO formulation, allows the model to scale across the 90-year KRILLBASE dataset without encountering the $\mathcal{O}(N^3)$ memory bottleneck of exact GP inference.

## Implementation

### 1. Data Processing Pipeline

Before the model can learn, the disparate data sources must be merged into a cohesive spatio-temporal tensor.

* **Ingestion:** We will use `xarray` and `dask` via the Pangeo ecosystem to stream CMIP6 climate data out-of-core, preventing local memory overflow. KRILLBASE data will be ingested via `pandas`.
* **Alignment:** For every historical KRILLBASE net haul at coordinate $(t, latitude, longitude)$, we will extract the corresponding CMIP6 covariates (Sea Surface Temperature, Sea Ice Concentration, Chlorophyll-a).
* **Target Generation:** The continuous target ($y$) is the raw krill density. The binary hotspot target ($h$) is generated by flagging any observation that exceeds the historical 90th percentile for its geographic sector.

### 2. Model Architecture Configuration

Our core architecture will be built using `PyTorch` for the neural network components and `GPyTorch` for the scalable Gaussian Process mechanics.

* **The Mean Module:** A standard `nn.Linear` layer that projects the 3-dimensional CMIP6 covariate vector into a baseline krill density guess.
* **The Deep Kernel:** The spatial component of the covariance is parameterized by a 3-layer Multi-Layer Perceptron (MLP). It takes 2D GPS coordinates as input and outputs the geometry required to build valid, dynamically stretching covariance matrices ($\Sigma_s$) for the ocean currents.
* **Inducing Points ($Z$):** To initialize the $M=500$ inducing points for Sparse Variational Inference, we will run a K-Means clustering algorithm on the historical KRILLBASE coordinates. This ensures the model places its "summary anchors" where krill actually exist, rather than wasting memory computing the empty open ocean.

### 3. Optimization and Training

* **Loss Function:** The model uses a custom Multi-Task Evidence Lower Bound (ELBO) that combines the continuous prediction error (Mean Squared Error) and the hotspot classification error (Binary Cross-Entropy).
* **Mini-Batching:** Because we use the ELBO, the optimizer (Adam) can process the decades of data in small mini-batches (e.g., 256 samples per batch), allowing the neural network weights and inducing points to update iteratively.

## Experiments

To rigorously validate the predictive capabilities, physical realism, and computational viability of the Spatio-Temporal GP, we will evaluate the architecture across four experimental axes. All continuous predictions will be evaluated using Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE), while hotspot classifications will be scored via F1-score and Area Under the ROC Curve (AUC).

### Baselines

We will benchmark our architecture against three distinct modeling paradigms to prove the necessity of the hybrid approach:

* **Ecological Standard (XGBoost / Random Forest):** Tree-based ensembles are the current standard in marine species distribution modeling. They are highly non-linear but lack inherent spatial awareness and treat time/space coordinates as independent tabular features.
* **Time-Series Deep Learning (LSTM):** Long Short-Term Memory networks represent the standard deep learning approach to temporal sequences. We will map the historical KRILLBASE data into a spatial grid and utilize an LSTM to forecast the next time step, demonstrating the limitations of standard sequential models when handling highly irregular, sparse oceanic point data.
* **Stationary Gaussian Process:** An exact GP using a standard, stationary Radial Basis Function (RBF) kernel. This isolates the impact of the neural network, proving that the ocean cannot be modeled using simple Euclidean distance.

### Ablations

To isolate and quantify the contribution of specific architectural components, we will perform the following ablation studies:

* **No Path A (Removing the Climate Baseline):** We will remove the `nn.Linear` mean module ($\mu_{it}$) and force the model to predict krill density relying purely on the Deep Kernel's spatial memory. This tests the hypothesis that physical geography and environmental weather must act in tandem.
* **No Deep Spatial Warping:** We will replace the MLP output of covariance matrices ($\Sigma_s$) with a static identity matrix for all coordinates. This degrades the model to assume uniform circular correlation, quantifying exactly how much predictive power is gained by learning the ocean currents.
* **Temporal Decay Removal:** We will set $\nu(t, t') = 1$, assuming infinite temporal memory, to prove that krill population dynamics are inherently transient and rely heavily on recent (multi-year) historical states rather than century-long averages.

### Diagnostics

Beyond raw accuracy, a probabilistic model must be evaluated on its mathematical behavior and physical interpretability:

* **Uncertainty Calibration:** We will calculate the Expected Calibration Error (ECE) of the predictive distribution. If the model outputs a 95% confidence interval for krill biomass in a specific region, we will verify that 95% of the held-out KRILLBASE ground truth points actually fall within those bounds.
* **Latent Space Visualization (Physical Realism):** We will extract the learned covariance matrices ($\Sigma_s$) from the trained deep kernel and plot their corresponding ellipses across a map of the Antarctic.

We will visually verify that the principal eigenvectors (the longest part of the stretched ellipses) align with known physical fluid dynamics, such as stretching longitudinally along the Antarctic Circumpolar Current and compressing strictly near the Weddell Sea ice shelves.

---

> **Project Data**
> Please keep all code, notebooks, and documentation strictly within this repository. 
> All heavy deliverables like cleaned data should be uploaded to our [Google Drive Folder](https://drive.google.com/drive/folders/1_3XDOrVUuGGrNq-zUBRkQv1RsVKW0yiL?usp=sharing).
