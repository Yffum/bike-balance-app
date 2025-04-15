
# ToDo 2025-04-06

- [x] mouse input only over map
  - idea: parent of map checks if mouse is inside and updates bool in child that dictates whether to listen

# 2025-04-10

- [ ] handle exit code 1 after calling sim in sim_controller.gd
	- set spinner to X and write error to log
- [ ] when expanding walk node, check nearest stations as well as nearest stations with negative incentive

# 2025-04-11

- [ ] 250411_1642_s281203567.log Fix issue with basic agent waiting instead of ending trip
- [ ] 250411_1647_s281203567.log Fix issue where smart agent walks instead of ending trip

- [ ] For smart agent, search highest incentives first when biking/walking for first node, instead of nearest
- [ ] Add mode to application for finding average reward instead of batch


- Idea: try predicting and caching all updated incentives, then only calculate new incentive for stations traveled to/from

- Precalculate and cache every possible incentive for each bike count for each station? 125 stations * ~40 bike counts = 5000 incentives, not bad
--> then have agent search best incentives instead of nearest stations


# Notes

- One problem with the twin fail count simulations is sometimes adding a single bike doesn't reduce the fail count very much if it's completely empty (but adding multiple bikes reduces it significantly) so it isn't incentivized for returns when it should be.
  - Analyze incentive data. E.g. min/max/avg incentive for empty/full stations