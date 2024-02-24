# Integrated Hydrological and Water Management Modeling for Drought Analysis

## Overview

This is an ongoing project that integrates data-driven reservoir operation model (as water management component) with a runoff routing model, to simulate the dynamics of streamflow drought and water supply deficit at regulated river basisn. 

### Hydrological component:

- Runoff simulation at $0.125^{\degree} \times 0.125^{\degree}$ spatial resolution, retrieved from [NLDAS VIC land surface product](https://disc.gsfc.nasa.gov/datasets/NLDAS_VIC0125_H_2.0/summary?keywords=NLDAS).
- River routing: linear river routing, by [Oki et al. (1999)](https://www.jstage.jst.go.jp/article/jmsj1965/77/1B/77_1B_235/_article).
- Flow direction upscaling: Dominant River Tracing (DRT) algorithm, by [Wu et al. (2011)](https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2009WR008871).

### Water management component

- Reservoir operation: Generic Data-driven Reservoir Operation Model (GDROM), by [Chen & Li et al. (2022)](https://www.sciencedirect.com/science/article/pii/S0309170822001397).

## More details & results to be continued...

