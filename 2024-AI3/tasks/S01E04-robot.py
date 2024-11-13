prompt="""
[Find Path from A to B in 2D Map]

This move instructs the AI to find and return the path from point A to point B on a 2D map while avoiding obstacles.

Input:
- A=(3,0)
- B=(3,5)
- obsticles=[(0,1), (1,3), (2,1), (2,3), (3,1)]

Objective:
Determine a sequence of moves from point A to point B, specifying each action (RIGHT, DOWN, UP, LEFT), ensuring no entry into obstacles or out-of-bounds areas.

Rules:
- Find a path without entering obstacle coordinates or exiting the map boundaries.
- The x-coordinate must be between 0 and 3.
- The y-coordinate must be between 0 and 5.
- Prioritize moving LEFT or RIGHT before moving UP or DOWN in each step.
- List each move in the format: Current position - Action - Next point.
- Include pathfinding steps detailing possible moves and their validity.
- Conclude with a final XML summary, listing the sequence of actions.
- Actions:
  - UP (-1 to x)
  - DOWN (+1 to x)
  - LEFT (-1 to y)
  - RIGHT (+1 to y)

Pathfind steps:
- Current position: (initial)
  - Attempt move RIGHT - Check for validity: obstacle or out-of-bounds
  - Attempt move DOWN - Check for validity: obstacle or out-of-bounds
  - Attempt move UP - Check for validity: obstacle or out-of-bounds
  - Attempt move LEFT - Check for validity: obstacle or out-of-bounds
  - **Perform valid action**; Update position to the next valid point
- Continue this process until reaching the destination

Final result:
```
<RESULT>
{
 "steps": "sequence of actions to reach B, avoiding obstacles, type string"
}
</RESULT>
```

Example final result
```
<RESULT>
{
 "steps": "DOWN, UP, RIGHT, DOWN,..."
}
</RESULT>
```

Final confirmation and readiness to process the pathfinding request while adhering to defined limitations and output precision.
"""