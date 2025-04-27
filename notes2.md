# 2025-04-17 App Plans


## To Do
- [ ] Get appropriate distribution and generate travel times
- [ ] Create proper output file for godot instead of using print output
- [ ] Add warning for minimum samples not reached when timeout
- [ ] Add way to pause or stop simulation from app
- [ ] New output stat: total number of expected rental/return failures over next hour, to demonstrate the state of the system
- [ ] Add stat_keys arg to tools.estimate_stochastic_stats that filters for only the given keys unless None

!!!!!!!!!!!!!!!!!!!!!!!!!!
- [ ] When results loaded, duplicate map markers and keep separate list of results markers that are shown and hidden when the results tab is revealed/hidden.
- [ ] Results tab has a load button, and automatically loads after simulations

- [x] Write a python script that takes the results and generates a rich text output for godot
- [ ] Add station validation to basic agent (maybe add sim parameter? rental strategy: [selfish, cooperative/fair])

- [ ] Add station agnostic batch setting that randomly sets input start/end stations
  
Results tab
-> show station results, hide station info (or just hide switch buttons)
-> change map markers 


# Search Algo 
- [ ] Agent search node reward value should diminish with time (or maybe node depth?)
- [ ] Search branches that end early should be considered less valuable (right now they arent because its just reward rate)

### Station
- [x] Change marker size with zoom
- [x] Add button

