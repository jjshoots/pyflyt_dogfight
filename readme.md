# PyFlyt Dogfight

This is a single file reinforcement learning environment for training AI agents to perform aerial dogfighting within PyFlyt.
The aircraft itself uses the same number of control surfaces as the default PyFlyt fixedwing aircraft, but is modified to provide much higher maneuverability.

![](./dogfight.gif)

## Environment Rules

- This is a cannons only environment. Meaning there are no missiles. An agent has to point its nose directly at the enemy for it to be considered a hit.
- The gun is only effective within `lethal range`. Outside of this range, the gun deals no damage.
- The gun automatically fires when it can, there is no action for the agent to fire the weapon. This is similar to many [fire control systems](https://en.wikipedia.org/wiki/Fire-control_system) on modern aircraft.
- An agent loses if it:
  a) Hits anything
  b) Flies out of bounds
  c) Loses all its health

## Environment Parameters

- `flight_dome_size`: `float` - size of the dome that the agent must stay within to be within bounds.
- `max_duration_seconds`: `float` - maximum flight time before the environment truncates _when not rendering_.
- `agent_hz`: `int` - how fast the agent operates in the environment. The physics itself steps at 240 Hz, so if `agent_hz` is set at 40, each environment step is 6 physics steps.
- `damage_per_hit`: `float` - how much damage per hit per physics step.
- `spawn_height`: `float` - how high to spawn the agents at the beginning. The orientation is and XY positions are randomized. A fixed height means both agents start with the same amount of [energy](https://en.wikipedia.org/wiki/Energy%E2%80%93maneuverability_theory).
- `lethal_distance`: `float` - how close before the weapons become effective.
- `lethal_angle_radian`: `float` - the width of the cone of fire.
- `lethal_offset`: `float` - how far must bullets hit off the main body for damage to be registered.
- `assisted_flight`: `bool` - whether to use high level commands (RPYT) instead of giving the agent full control over each actuator.
- `render`: `bool` - whether to render the environment. Under rendering, the environment never truncates.
