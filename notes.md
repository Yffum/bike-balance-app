2025-05-02


## Future Work
- Rigorous Automated Testing
  - ensure conditions are met such that simulation and agents behave as expected
- Enhance UI
  - Use separate thread for results processing to prevent stalling on simulation completion
  - Add interface for loading and managing (e.g. removing) previous results
  - Recenter map to station when station selected through station tab
  - Allow selecting station from agent actions list in results
- Sim parameters
  - Rental strategy: [selfish, cooperative/fair]
- Sim results
  - Total walk time
- Refine simulation parameter collection
- Refine simulation accuracy
  - Refine agent
    - Try diminishing reward value with time (or node depth) for smart agent
    - Search branches that end early should be considered less valuable (some function of the time passed or maybe node depth)
  - Simulate other agents
  - Refine trip duration generation: examine data from single rider instead of group to determine appropriate distribution and distribution parameters for generating trip times
- Create app for bike angels
- Expand simulation
  - Use multiple agents