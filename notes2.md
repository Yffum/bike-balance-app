# 2025-04-17 App Plans


## To Do
- [ ] Get appropriate distribution and generate travel times
- [ ] Create proper output file for godot instead of using print output
- [ ] Add warning for minimum samples not reached when timeout
- [ ] Add way to pause or stop simulation from app
- [ ] New output stat: total number of expected rental/return failures over next hour, to demonstrate the state of the system


### Station
- [ ] Change marker size with zoom
- [ ] Add button

Station States
- Selected/Unselected
  - Use outlined sprite
  - Or use a shader to outline the sprite
- Start/End/Intermediate
  - Green/Red/Gray
  - Half/half for start-end combo
  - try checkered for end

Station Panel
- Station index
  - type in automatically highlights and centers the corresponding station
  - pressing the station marker automatically opens the station panel
- Stats
  - Rental rate
  - Return rate
  - Starting bike count
  - Capacity


Parameters Panel
- Agent Parameters
- Bike Share Parameters

Simulation Panel
- Empty station bias
- Full station bias
- Warmup time
- Mode
  - Single Run
    - Use static seed
  - Batch
    - Parallel batch size # Try the number of cores in your CPU times 2.
    - Confidence level
    - Batch mode
      - Fixed Sample Size
        - Sample size
      - Precision Based
        - Minimum samples # Due to the high variance of samples, decreasing minimum samples may produce innaccurate estimates
        - Relative margin of error
        - Raw margin of error 
        - Maximum total runtime (0.5 min)
        - # Replications will stop once max runtime reached, or margin of error is below bound (relative or raw) for every stat. Lower runtime by increasing margin of error and/or decreasing confidence level.