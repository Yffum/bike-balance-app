2025-05-02

# To Do

## Frontend
- [x] Add cancel button for simulation
- [x] Add handling for simulation call fail
- [x] Add setting for python name/path e.g. python3

## Simulation
- [x] Add station validation for basic agent
- [x] Improve station validation for smart agent
- [ ] Add station agnostic batch setting

- [ ] Try diminishing reward value with time (or node depth) for smart agent
- [ ] Search branches that end early should be considered less valuable (some function of the time passed or maybe node depth)

- [x] Implement new end excursion policy

## Refinement
- [ ] Issue where smart agent waits when out of time res://external/saved_results/250505-1622_s1251291833.json

## Finalization
- [ ] Verify default_user_params.json works

## Future Work
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
  - Simulate other agents
  - Refine trip duration generation: examine data from single rider instead of group to determine appropriate distribution and distribution parameters for generating trip times
- Create app for bike angels
- Expand simulation
  - Use multiple agents

# Project Deliverables
- README.md
  - Introduction to app
    - What is the app
    - Who is it for
- Wiki
  - Installation guide
    - Install through build
    - Tested on Python 3.12, but should work with Python 3.10 or later
  - Build guide
  - report.pdf

## Report
- Abstract
- Introduction
- Background
  - Related work
- Problem Statement
- Objectives
- System Architecture
  - Attributions
- Features
- Discussion
- Future Work
- References
- Appendices