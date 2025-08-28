# EV Charging Station Resilience Simulation

A Python-based simulation framework for analyzing the resilience of electric vehicle charging infrastructure in the UK under various failure scenarios.

## Project Structure

```
├── src/                          # Core simulation source code
├── helpers/                      # Utility scripts and data processing tools
├── data/                         # Input data files
└── results/                      # Simulation output files
```

## Source Files (`src/`)

### Core Simulation Files
- **`main.py`** - Main entry point that orchestrates the charging station assignment process and creates complete visualizations
- **`simPySimulation.py`** - Primary discrete-event simulation engine using SimPy, handles EV spawning, charging queues, and journey completion tracking
- **`charging_utils.py`** - Charging logic utilities including station capacity calculations and queue tolerance management
- **`posterVisualisation.py`** - Generates high-quality poster maps (SVG/JPEG) for presentations and publications

### Supporting Modules
- **`station_assignment.py`** - Assigns charging stations to graph nodes based on geographic proximity
- **`evDriver.py`** - EV driver behavior model including battery management and connector compatibility
- **`pathfinding.py`** - Navigation algorithms for finding nearest charging stations and route planning
- **`queue_time_tracker.py`** - Tracks and analyzes charging queue times throughout simulation periods
- **`visualization.py`** - Creates network visualizations showing charging stations and road network

## Helper Scripts (`helpers/`)

### Data Processing
- **`filterMap.py`** - Filters UK geographic data to mainland only (excludes Northern Ireland and islands)
- **`simplifyGeojson.py`** - Simplifies GeoJSON geometries for improved performance
- **`WorkingFlag.py`** - Adds working status flags to charging station data

### Failure Simulation
- **`addResiliencyRandom.py`** - Simulates random 10% charging station failures
- **`addResiliencyTargeted.py`** - Simulates targeted failures based on station importance/usage

### Visualization
- **`CreateUkMap.py`** - Creates individual UK maps showing different failure scenarios
- **`plotGraphs.py`** - Generates comparison maps and statistical visualizations

## Data Files (`data/`)

- **`UK_Mainland_GB_simplified.geojson`** - Simplified UK mainland geographic boundaries
- **`cleanedM_charging_stations.json`** - Cleaned charging station database
- **`AllWorkingChargingStations.json`** - Baseline scenario with all stations operational
- **`RandomFailuresChargingStations.json`** - 10% random failure scenario
- **`TargetedWeightedFailures.json`** - 10% targeted failure scenario

## Results (`results/`)

Contains simulation output files with queue time data and performance metrics in JSON format.

## Quick Start

1. Run the main simulation:
   ```bash
   python src/main.py
   ```

2. Generate failure scenarios:
   ```bash
   python helpers/addResiliencyRandom.py
   python helpers/addResiliencyTargeted.py
   ```

3. Create visualizations:
   ```bash
   python helpers/CreateUkMap.py
   ```

4. Run resilience simulation:
   ```bash
   python src/simPySimulation.py
   ```

## Key Features

- **Multi-scenario analysis**: Compare baseline, random failure, and targeted failure scenarios
- **Queue simulation**: Realistic modeling of charging station queues and wait times
- **Geographic visualization**: High-quality maps showing station distribution and failures
- **Performance metrics**: Comprehensive statistics on journey completion rates and charging efficiency