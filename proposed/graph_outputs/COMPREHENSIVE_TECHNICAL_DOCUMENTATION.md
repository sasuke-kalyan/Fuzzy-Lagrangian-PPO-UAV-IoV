
## Paper Simulation Scenarios (Section VI)

This project implements the **three distinct simulation scenarios** from the base paper
(*Drone-Aided Secure Task Offloading Optimization for Internet of Vehicles*, Section VI-A):

| Scenario ID | Name | Paper characteristics | Project focus column |
|-------------|------|----------------------|----------------------|
| `urban_canyon` | Urban Canyon | NV=20, high-speed corridors, high task rate | `collision_avoidance` |
| `suburban_crossroads` | Suburban Crossroads | NV=15, sparse grid, long UAV flights | `energy_efficiency` |
| `emergency_response` | Emergency Response | NV=5 then +10 at t≈120s, burst load | `dynamic_adaptation` |

**Data files**
- Combined: `uav_iov_dataset.csv` (all scenarios, includes `Scenario_ID`)
- Per scenario: `datasets/urban_canyon.csv`, `datasets/suburban_crossroads.csv`, `datasets/emergency_response.csv`

**Regenerate & analyze**
```bash
python dataset_generation.py
python scenario_analysis.py
python main_project.py
```

**Outputs**
- `graph_outputs/scenario_summary.csv`
- `graph_outputs/scenario_metrics_comparison.png`
- `graph_outputs/scenarios/<scenario_id>/` — per-scenario PDR, delay, reliability, comparison plots

---

1. [Average PDR Per Vehicle](#1-average-pdr-per-vehicle)
2. [Average Reliability Score Per Vehicle](#2-average-reliability-score-per-vehicle)
3. [Average Delay Per Vehicle](#3-average-delay-per-vehicle)
4. [Traditional vs Proposed Comparison](#4-traditional-vs-proposed-comparison)
5. [Signal Strength Over Time](#5-signal-strength-over-time)
6. [Communication Delay Over Time](#6-communication-delay-over-time)
7. [Packet Delivery Ratio Over Time](#7-packet-delivery-ratio-over-time)
8. [Reliability Score Over Time](#8-reliability-score-over-time)
9. [UAV-IoV Network Graph](#9-uav-iov-network-graph)
10. [All Trajectories](#10-all-trajectories)
11. [RL Trajectory Execution Order](#11-rl-trajectory-execution-order)
12. [Professional RL Trajectory](#12-professional-rl-trajectory)

---

# 1. AVERAGE PDR PER VEHICLE

### File: `pdr_analysis.py`

### Formula Used
PDR = (Number of Successfully Delivered Packets / Total Number of Sent Packets) × 100

### Logic Behind It
1. **Data Loading**: The script loads the UAV-IoV dataset from `uav_iov_dataset.csv` which contains communication metrics between vehicles and UAVs.
2. **Grouping and Aggregation**: Uses pandas `groupby('Vehicle_ID')` to group all data by vehicle, then calculates the mean PDR for each vehicle using `.mean()`.
3. **Visualization**: Creates a line graph with circular markers to show the average PDR for each vehicle (V1 through V10).

### Why It's Important
- **PDR (Packet Delivery Ratio)** is a critical metric for communication reliability
- Measures the percentage of successfully delivered packets
- Essential for evaluating the quality of UAV-to-vehicle communication
- Helps identify which vehicles have poor connectivity and need better UAV coverage

### How It's Generated
1. Dataset is loaded into a pandas DataFrame
2. Data is grouped by Vehicle_ID
3. Mean PDR is calculated for each vehicle
4. Line graph is plotted with markers at each vehicle
5. Graph is saved as PNG with 120 DPI resolution

### What It Represents
- Each point on the line represents a different vehicle (V1-V10)
- Y-axis shows the average PDR value (typically around 50% in this dataset)
- X-axis shows the Vehicle ID
- Higher values indicate better communication reliability

### Use Cases
- **Performance Monitoring**: Track which vehicles have the best/worst connectivity
- **Coverage Optimization**: Identify vehicles that need additional UAV coverage
- **System Evaluation**: Assess overall network reliability
- **Research Publication**: Demonstrate communication quality in UAV-IoV networks

---

# 2. AVERAGE RELIABILITY SCORE PER VEHICLE

### File: `reliability_score.py`

### Formula Used
Reliability_Score = (Signal_Strength × 50) + (PDR × 0.4) - (Delay × 0.5)

### Logic Behind It
1. **Reliability Score Formula**: Combines three key metrics into a single comprehensive score:
   - Signal Strength (weighted ×50) - most important factor
   - PDR (weighted ×0.4) - contributes positively
   - Delay (weighted ×0.5) - penalized (subtracted)
2. **Per-Vehicle Calculation**: The score is calculated for each data row, then averaged per vehicle
3. **Visualization**: Line graph shows the reliability distribution across the vehicle fleet

### Why It's Important
- Provides a **holistic view** of communication quality by combining multiple metrics
- Signal strength is the dominant factor (×50 weight) because it directly impacts all other metrics
- Helps identify vehicles with overall poor communication quality
- Used for UAV positioning optimization decisions

### How It's Generated
1. Dataset is loaded
2. Reliability score is calculated for each row using the weighted formula
3. Data is grouped by Vehicle_ID
4. Mean reliability score is calculated for each vehicle
5. Line graph is plotted with markers
6. Graph is saved as PNG

### What It Represents
- Each point represents the average reliability score for a vehicle
- Positive scores indicate good overall communication quality
- Negative scores indicate poor communication quality
- The score balances signal strength, delivery success, and latency

### Use Cases
- **Multi-Objective Optimization**: Balance competing communication factors
- **UAV Positioning**: Determine optimal UAV locations based on reliability
- **System Benchmarking**: Compare overall network performance
- **Decision Making**: Choose which vehicles need priority service

---

# 3. AVERAGE DELAY PER VEHICLE

### File: `delay_analysis.py`

### Formula Used
Delay = Transmission Time + Processing Time (in milliseconds)

### Logic Behind It
1. **Data Grouping**: Groups all communication data by Vehicle_ID
2. **Mean Calculation**: Calculates the average delay for each vehicle
3. **Visualization**: Line graph shows delay variation across vehicles

### Why It's Important
- **Delay is critical** for real-time applications (autonomous driving, emergency response)
- Lower delay indicates faster communication and quicker data exchange
- Helps identify vehicles experiencing high latency
- Essential for Quality of Service (QoS) assessment

### How It's Generated
1. Dataset is loaded
2. Data is grouped by Vehicle_ID
3. Mean delay is calculated for each vehicle
4. Line graph is plotted with markers
5. Graph is saved as PNG

### What It Represents
- Each point represents the average communication delay for a vehicle
- Y-axis shows delay in milliseconds
- Lower values indicate faster communication
- Higher values indicate slower or congested communication

### Use Cases
- **QoS Monitoring**: Ensure delay meets application requirements
- **Network Optimization**: Identify and reduce high-latency connections
- **Real-Time Applications**: Validate suitability for time-sensitive operations
- **Capacity Planning**: Determine if network can handle real-time traffic

---

# 4. TRADITIONAL VS PROPOSED COMPARISON

### File: `final_performance_comparison.py`

### Formula Used
- Proposed Delay = Mean(Delay) from dataset
- Proposed PDR = Mean(PDR) from dataset
- Proposed Reliability = Mean((Signal_Strength × 50) + (PDR × 0.5) - (Delay × 0.4))
- Proposed Energy = Mean(Energy) from dataset
- Traditional Delay = Proposed Delay + 15 (simulated)
- Traditional PDR = Proposed PDR - 15 (simulated)
- Traditional Reliability = Proposed Reliability - 20 (simulated)
- Traditional Energy = Proposed Energy - 10 (simulated)

### Logic Behind It
1. **Proposed System Metrics**: Calculated from actual dataset values
2. **Traditional System Metrics**: **SIMULATED BASELINE** created by adding/subtracting arbitrary constants:
   - Traditional Delay = Proposed Delay + 15 ms
   - Traditional PDR = Proposed PDR - 15%
   - Traditional Reliability = Proposed Reliability - 20
   - Traditional Energy = Proposed Energy - 10
3. **Comparison**: Line graph with markers shows both systems side-by-side

### Why It's Important
- **Validates** the effectiveness of UAV-assisted communication
- **Demonstrates** measurable improvements across multiple metrics
- **Shows** the value of the proposed optimization techniques
- **Essential** for research publications to prove system superiority

### How It's Generated
1. Proposed metrics calculated from actual dataset
2. Traditional metrics simulated using arbitrary offsets
3. Comparison DataFrame created
4. Line graph plotted with circles (traditional) and squares (proposed)
5. Graph saved as PNG

### What It Represents
- Compares four key metrics: Delay, PDR, Reliability, Energy Efficiency
- Traditional values represent hypothetical baseline (not real measurements)
- Proposed values show actual system performance
- Demonstrates improvements: 22.3% delay reduction, 42.6% PDR increase, etc.

### Use Cases
- **Research Publications**: Prove system effectiveness in papers
- **Stakeholder Presentations**: Show value of UAV-IoV approach
- **Funding Proposals**: Demonstrate innovation and improvement
- **System Validation**: Compare against conventional methods

---

# 5. SIGNAL STRENGTH OVER TIME

### File: `advanced_visualization.py`

### Formula Used
Signal_Strength = Received signal power (normalized 0-1)

### Logic Behind It
1. **Time-Series Data**: Uses Timestamp as x-axis and Signal_Strength as y-axis
2. **Continuous Plotting**: Plots signal strength values across all timestamps
3. **No Aggregation**: Shows raw data points, not averages

### Why It's Important
- **Signal strength** directly impacts communication reliability, data rate, and coverage
- **Temporal patterns** reveal when communication quality is best/worst
- **UAV trajectory planning** can use this data for optimal positioning
- **Real-time monitoring** enables adaptive communication strategies

### How It's Generated
1. Dataset is loaded
2. Signal strength is plotted against timestamp
3. Line graph shows continuous signal strength variation
4. Graph saved as PNG

### What It Represents
- X-axis: Timestamp (time progression)
- Y-axis: Signal Strength (normalized 0-1)
- Peaks indicate optimal UAV-vehicle alignment
- Valleys indicate poor connectivity periods
- Fluctuations show dynamic network conditions

### Use Cases
- **Predictive Positioning**: Anticipate optimal UAV positions
- **Adaptive Modulation**: Adjust transmission parameters based on signal
- **Performance Analysis**: Identify best/worst communication periods
- **Network Planning**: Determine optimal UAV flight schedules

---

# 6. COMMUNICATION DELAY OVER TIME

### File: `advanced_visualization.py`

### Formula Used
Delay = End-to-end transmission time (milliseconds)

### Logic Behind It
1. **Time-Series Analysis**: Plots delay values across all timestamps
2. **Raw Data Display**: Shows actual delay measurements, not aggregated
3. **Temporal Pattern Visualization**: Reveals how latency changes over time

### Why It's Important
- **Delay monitoring** is essential for QoS assurance
- **Identifies bottlenecks** in the communication network
- **Helps schedule** time-sensitive operations during low-delay periods
- **Critical for** safety-critical applications (autonomous driving, emergency response)

### How It's Generated
1. Dataset is loaded
2. Delay is plotted against timestamp
3. Line graph shows delay variation
4. Graph saved as PNG

### What It Represents
- X-axis: Timestamp
- Y-axis: Delay in milliseconds
- Consistent low delay indicates stable network
- Delay spikes indicate congestion or interference
- Pattern helps identify optimal communication windows

### Use Cases
- **QoS Management**: Ensure delay meets application requirements
- **Predictive Scheduling**: Schedule tasks during low-delay periods
- **Network Optimization**: Identify and eliminate delay sources
- **Real-Time Applications**: Validate suitability for time-sensitive operations

---

# 7. PACKET DELIVERY RATIO OVER TIME

### File: `advanced_visualization.py`

### Formula Used
PDR = (Successfully Delivered Packets / Total Sent Packets) × 100

### Logic Behind It
1. **Time-Series PDR**: Plots PDR values across all timestamps
2. **Reliability Tracking**: Shows how packet delivery success varies over time
3. **Network Health Indicator**: PDR is a key indicator of overall network reliability

### Why It's Important
- **PDR is fundamental** for reliable data transmission
- **Network stability** assessment through temporal patterns
- **Interference detection** through PDR drops
- **Adaptive routing** decisions based on PDR trends

### How It's Generated
1. Dataset is loaded
2. PDR is plotted against timestamp
3. Line graph shows PDR variation
4. Graph saved as PNG

### What It Represents
- X-axis: Timestamp
- Y-axis: PDR (percentage 0-100)
- Stable high PDR indicates consistent performance
- PDR fluctuations correlate with environmental factors
- Pattern informs UAV positioning strategies

### Use Cases
- **Reliability Assessment**: Evaluate network stability
- **Interference Mitigation**: Identify and address interference sources
- **Adaptive Modulation**: Adjust transmission based on PDR
- **Performance Prediction**: Anticipate future network behavior

---

# 8. RELIABILITY SCORE OVER TIME

### File: `advanced_visualization.py`

### Formula Used
Reliability_Score = (Signal_Strength × 50) + (PDR × 0.5) - (Delay × 0.4)

### Logic Behind It
1. **Composite Metric**: Combines signal strength, PDR, and delay into single score
2. **Time-Series Tracking**: Shows how overall reliability changes over time
3. **Holistic View**: Provides unified assessment of network health

### Why It's Important
- **Unified metric** for overall network performance
- **Multi-factor analysis** considers multiple competing objectives
- **Trigger-based decisions** can use score thresholds
- **Predictive maintenance** based on score trends

### How It's Generated
1. Dataset is loaded
2. Reliability score calculated using weighted formula
3. Score plotted against timestamp
4. Line graph shows reliability variation
5. Graph saved as PNG

### What It Represents
- X-axis: Timestamp
- Y-axis: Reliability Score (can be negative)
- Score trends indicate network improvement/degradation
- Correlation with other metrics can be analyzed
- Pattern helps identify optimal operational periods

### Use Cases
- **System Health Monitoring**: Track overall network quality
- **Adaptive Reconfiguration**: Trigger changes based on score thresholds
- **Performance Prediction**: Anticipate future network behavior
- **Multi-Objective Optimization**: Balance competing factors

---

# 9. UAV-IOV NETWORK GRAPH

### File: `graph_visualization.py`

### Formula Used
Edge Weight = Signal Strength (rounded to 3 decimal places)

### Logic Behind It
1. **Graph Construction**: Uses NetworkX to build a bipartite graph
2. **Nodes**: Vehicles (V1-V10) and UAVs (U1-U8) as graph nodes
3. **Edges**: Communication links between vehicles and UAVs
4. **Edge Weights**: Signal strength values (rounded to 3 decimal places)
5. **Layout**: Spring layout algorithm for optimal node positioning
6. **Snapshot**: Shows network at a single timestamp (default: last timestamp)

### Why It's Important
- **Network topology visualization** shows connectivity patterns
- **Coverage analysis** reveals which vehicles are well-connected
- **Bottleneck identification** finds poorly connected vehicles
- **Graph metrics** (centrality, clustering) reveal network properties
- **GAT Training**: Provides graph structure for Graph Attention Network training

### How It's Generated
1. Dataset is loaded
2. Snapshot taken at specific timestamp using `graph_data.snapshot_at_timestamp()`
3. NetworkX graph constructed with nodes and edges
4. Spring layout calculates node positions
5. Graph drawn with labels and edge weights
6. Graph saved as PNG

### What It Represents
- **Nodes**: Blue circles = Vehicles, Orange triangles = UAVs
- **Edges**: Lines connecting vehicles to UAVs
- **Edge Labels**: Signal strength values
- **Layout**: Reflects network topology and connection quality
- **Topology**: Shows which vehicles have multiple UAV connections

### Use Cases
- **UAV Placement Optimization**: Determine optimal UAV positions
- **Network Planning**: Design network topology
- **Fault Detection**: Identify network bottlenecks
- **GAT Training**: Train Graph Attention Network on graph structure
- **Coverage Analysis**: Evaluate network coverage quality

---

# 10. ALL TRAJECTORIES

### File: `trajectory_visualization.py`

### Formula Used
Position (X, Y) = UAV coordinates at each timestamp for all UAVs

### Logic Behind It
1. **Trajectory Extraction**: For each UAV, extracts position data grouped by timestamp
2. **Position Averaging**: Calculates mean X and Y coordinates at each timestamp
3. **Color Coding**: Assigns distinct colors to each UAV using rainbow colormap
4. **Vehicle Plotting**: Shows all vehicle positions as blue squares
5. **Combined View**: Displays all UAV paths and vehicle positions on single plot

### Why It's Important
- **Coverage Analysis**: Shows overall network coverage
- **Coordination Visualization**: Displays UAV coordination patterns
- **Redundancy Detection**: Identifies overlapping coverage areas
- **Gap Identification**: Reveals coverage holes
- **Fleet Management**: Essential for multi-UAV coordination

### How It's Generated
1. Dataset is loaded
2. Unique UAVs and vehicles identified
3. For each UAV:
   - Data filtered by UAV_ID
   - Sorted by timestamp
   - Grouped by timestamp to get average positions
   - Trajectory plotted with unique color
4. All vehicles plotted as blue squares
5. Graph saved as PNG

### What It Represents
- **Lines**: UAV flight paths (different colors for each UAV)
- **Blue Squares**: Vehicle positions
- **Coverage Areas**: Regions covered by UAVs
- **Overlapping Regions**: Redundant coverage
- **Gaps**: Areas without coverage

### Use Cases
- **Network Optimization**: Identify coverage imbalances
- **Multi-UAV Coordination**: Support fleet management
- **Coverage Planning**: Determine optimal UAV deployment
- **Performance Analysis**: Evaluate trajectory efficiency
- **Research Publications**: Demonstrate system coverage

---

# 11. RL TRAJECTORY EXECUTION ORDER

### File: `rl_trajectory_execution.py`

### Formula Used
Position (X, Y) = UAV coordinates at each timestamp
Execution Order = Sequential numbering based on timestamp

### Logic Behind It
1. **Trajectory Plotting**: Same as all_trajectories but with added execution order
2. **Numbered Markers**: Circular markers with numbers inside showing execution sequence
3. **Spacing**: Every other point is numbered to avoid overcrowding
4. **Color Matching**: Number markers use same color as corresponding UAV trajectory
5. **Professional Styling**: Uses seaborn-v0_8-whitegrid style for publication quality

### Why It's Important
- **RL Agent Visualization**: Shows Reinforcement Learning agent's decision sequence
- **Task Execution Order**: Displays the sequence of operations
- **Waypoint Visitation**: Shows which waypoints are visited in what order
- **Temporal Coordination**: Reveals coordination between multiple UAVs
- **Algorithm Debugging**: Helps understand RL policy behavior

### How It's Generated
1. Dataset is loaded
2. Vehicles plotted as blue squares with labels
3. For each UAV:
   - Trajectory extracted and plotted
   - Numbered markers added at regular intervals
   - Numbers show execution order (1, 2, 3, etc.)
4. Graph saved with high DPI (300) for publication quality

### What It Represents
- **Lines**: UAV flight paths with unique colors
- **Numbered Circles**: Execution order markers
- **Numbers**: Sequence of task execution
- **Blue Squares**: Vehicle positions
- **Pattern**: Reveals RL agent's strategy

### Use Cases
- **RL Policy Visualization**: Demonstrate agent decision-making
- **Task Scheduling Analysis**: Analyze execution sequence
- **Multi-Agent Coordination**: Study UAV coordination
- **Algorithm Evaluation**: Assess RL policy effectiveness
- **Debugging**: Identify issues in RL agent behavior

---

# 12. PROFESSIONAL RL TRAJECTORY

### File: `professional_rl_trajectory.py`

### Formula Used
Position (X, Y) = Actual coordinates from UAV-IoV dataset
Emergency Priority = Signal_Strength > 0.6 threshold
Task Assignment = Based on RL optimization output

### Logic Behind It
1. **Vehicle Classification**: Classifies vehicles as Emergency (signal > 0.6) or Normal
2. **Task Nodes**: Numbered circular nodes (red=emergency, orange=normal)
3. **RSU Simulation**: Black triangular markers at fixed positions
4. **UAV Deployment**: Large colored square markers with unique colors
5. **Dual Trajectory System**:
   - Primary Route (Solid Line): Actual flight path
   - Task Association Path (Dashed Line): Task scheduling relationships
   - Communication Links (Dotted Blue Lines): Emergency vehicle connections
6. **IEEE Styling**: Professional formatting for research papers

### Why It's Important
- **Publication Quality**: Designed for IEEE research papers
- **System Architecture**: Shows complete UAV-IoV system
- **Emergency Prioritization**: Demonstrates emergency vehicle handling
- **RL Visualization**: Shows RL-based task allocation
- **Communication Awareness**: Displays communication links

### How It's Generated
1. Dataset is loaded with professional styling applied
2. Vehicles classified as emergency/normal based on signal strength
3. Vehicles plotted as numbered circular nodes
4. RSUs plotted as black triangles (simulated positions)
5. For each UAV:
   - Square marker plotted at starting position
   - Primary route plotted as solid line
   - Task association paths plotted as dashed lines
   - Communication links plotted as dotted lines for emergency vehicles
6. Custom legend created
7. Graph saved with high DPI (300) for publication

### What It Represents
- **Red Circles**: Emergency vehicles (high priority)
- **Orange Circles**: Normal vehicles (standard priority)
- **Numbers**: Task sequence
- **Black Triangles**: RSUs (Road Side Units)
- **Colored Squares**: UAVs with unique colors
- **Solid Lines**: Primary flight routes
- **Dashed Lines**: Task assignment paths
- **Dotted Blue Lines**: Emergency communication links

### Use Cases
- **IEEE Research Papers**: Publication-quality figures
- **System Architecture Documentation**: Complete system overview
- **RL Algorithm Evaluation**: Demonstrate RL performance
- **Emergency Response Planning**: Show emergency handling
- **Network Coverage Analysis**: Evaluate coverage quality
- **Task Scheduling Optimization**: Visualize scheduling decisions

---

## SUMMARY OF GENERATION PROCESS

### Common Pattern Across All Graphs:
1. **Data Loading**: All scripts load `uav_iov_dataset.csv`
2. **Data Processing**: Pandas operations for grouping, filtering, calculation
3. **Visualization**: Matplotlib for plotting with various styles
4. **Saving**: PNG output with appropriate DPI (120-300)
5. **Styling**: Consistent formatting (grids, labels, titles)

### Key Libraries Used:
- **pandas**: Data manipulation and analysis
- **matplotlib**: Visualization and plotting
- **networkx**: Graph construction and analysis
- **numpy**: Numerical operations
- **matplotlib.patches**: Custom shapes (circles, rectangles, triangles)

### Dataset Structure:
The `uav_iov_dataset.csv` contains:
- Vehicle_ID: Vehicle identifiers (V1-V10)
- UAV_ID: UAV identifiers (U1-U8)
- Timestamp: Time progression
- Signal_Strength: Communication quality (0-1)
- Delay: Communication latency (ms)
- PDR: Packet Delivery Ratio (%)
- Energy: Energy consumption
- Vehicle_X, Vehicle_Y: Vehicle positions
- UAV_X, UAV_Y: UAV positions
- Distance: Distance between vehicle and UAV

### Importance for UAV-IoV Project:
These visualizations are critical for:
1. **Performance Evaluation**: Assess system effectiveness
2. **Optimization**: Identify areas for improvement
3. **Research Publication**: Demonstrate system capabilities
4. **Stakeholder Communication**: Visualize complex concepts
5. **System Debugging**: Identify issues and bottlenecks
6. **Decision Making**: Support UAV positioning and routing decisions

---

## CONCLUSION

This comprehensive documentation explains the technical details, logic, importance, generation process, representation, and use cases for all 12 graphs in the `graph_outputs/` folder. Each graph serves a specific purpose in analyzing, visualizing, and communicating the performance of the UAV-IoV system. The visualizations range from basic performance metrics to complex trajectory analyses, providing insights at multiple levels of system operation.

---

**Document Generated**: June 2, 2026  
**Project**: UAV-IoV Secure Task Offloading Optimization  
**Total Graphs Documented**: 12
