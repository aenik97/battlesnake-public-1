"""2-Move lookahead simulation engine for the Battlesnake.

Simulates playing a move and evaluates all possible continuations
to avoid dead ends and traps.
"""

import copy
from typing import Dict, List, Optional, Set, Tuple

from features import DIRECTIONS, _in_bounds, _manhattan

Point = Tuple[int, int]


def _clone_state(state: Dict) -> Dict:
    """Deep clone a game state for simulation."""
    return copy.deepcopy(state)


def _get_occupied(snakes: List[Dict]) -> Set[Point]:
    """Get all occupied cells from snakes."""
    occupied = set()
    for snake in snakes:
        for seg in snake["body"]:
            occupied.add((seg["x"], seg["y"]))
    return occupied


def _apply_move_to_state(state: Dict, move: str) -> Dict:
    """Simulate applying a move to a game state.
    
    Returns a new state with the move applied:
    - Head moves in the direction
    - Body shifts (tail removed unless food eaten)
    """
    simulated = _clone_state(state)
    you = simulated["you"]
    board = simulated["board"]
    
    head = (you["head"]["x"], you["head"]["y"])
    dx, dy = DIRECTIONS[move]
    new_head = (head[0] + dx, head[1] + dy)
    
    # Check if food is eaten
    foods = [(f["x"], f["y"]) for f in board["food"]]
    eats_food = new_head in foods
    
    # Update body: add new head, remove tail (unless eating)
    new_body = [{"x": new_head[0], "y": new_head[1]}] + you["body"]
    if not eats_food:
        new_body = new_body[:-1]  # Remove tail
    
    # Update snake state
    you["head"] = {"x": new_head[0], "y": new_head[1]}
    you["body"] = new_body
    you["length"] = len(new_body)
    you["health"] = max(0, you["health"] - 1 + (1 if eats_food else 0))
    
    # Update board snakes
    for snake in board["snakes"]:
        if snake["id"] == you["id"]:
            snake["head"] = you["head"]
            snake["body"] = new_body
            snake["length"] = you["length"]
            snake["health"] = you["health"]
    
    # Remove eaten food
    if eats_food:
        board["food"] = [f for f in board["food"] if (f["x"], f["y"]) != new_head]
    
    return simulated


def _count_open_space(state: Dict, head: Point, limit: int = 100) -> int:
    """Count reachable open space from a position (simplified flood fill)."""
    width = state["board"]["width"]
    height = state["board"]["height"]
    occupied = _get_occupied(state["board"]["snakes"])
    
    visited = {head}
    stack = [head]
    count = 0
    
    while stack and count < limit:
        x, y = stack.pop()
        count += 1
        
        for dx, dy in DIRECTIONS.values():
            neighbor = (x + dx, y + dy)
            if (neighbor not in visited and 
                _in_bounds(neighbor, width, height) and 
                neighbor not in occupied):
                visited.add(neighbor)
                stack.append(neighbor)
    
    return count


def _is_trapped(state: Dict, my_id: str) -> bool:
    """Check if a snake is trapped (no legal moves)."""
    you = state["you"]
    head = (you["head"]["x"], you["head"]["y"])
    width = state["board"]["width"]
    height = state["board"]["height"]
    occupied = _get_occupied(state["board"]["snakes"])
    
    for dx, dy in DIRECTIONS.values():
        neighbor = (head[0] + dx, head[1] + dy)
        if _in_bounds(neighbor, width, height) and neighbor not in occupied:
            return False  # Found at least one escape
    return True


def _get_legal_moves(state: Dict) -> List[str]:
    """Get all legal moves from current state."""
    you = state["you"]
    head = (you["head"]["x"], you["head"]["y"])
    width = state["board"]["width"]
    height = state["board"]["height"]
    occupied = _get_occupied(state["board"]["snakes"])
    
    legal = []
    for move, (dx, dy) in DIRECTIONS.items():
        nxt = (head[0] + dx, head[1] + dy)
        if _in_bounds(nxt, width, height) and nxt not in occupied:
            legal.append(move)
    return legal


def simulate_move(state: Dict, move: str) -> Dict:
    """Simulate one move and return the resulting state."""
    return _apply_move_to_state(state, move)


def evaluate_state(state: Dict, my_id: str) -> float:
    """Evaluate how good a simulated state is.
    
    Higher score = better position.
    """
    you = state["you"]
    head = (you["head"]["x"], you["head"]["y"])
    
    # Check if trapped (very bad)
    if _is_trapped(state, my_id):
        return -1000.0
    
    # Open space (very important)
    space = _count_open_space(state, head, limit=you["length"] + 5)
    space_score = space * 2.0
    
    # Health
    health_score = you["health"] * 1.0
    
    # Food proximity
    foods = [(f["x"], f["y"]) for f in state["board"]["food"]]
    if foods:
        nearest_food = min(_manhattan(head, f) for f in foods)
        food_score = (20 - nearest_food) * 0.5  # Closer is better
    else:
        food_score = 0.0
    
    # Total score
    return space_score + health_score + food_score


def lookahead_2_moves(state: Dict, move: str) -> float:
    """Score a move by simulating 2 turns ahead.
    
    Returns the best possible score achievable after playing `move`
    and then the best continuation.
    """
    # Simulate first move
    state_after_1 = simulate_move(state, move)
    
    # If trapped after move 1, very bad
    if _is_trapped(state_after_1, state["you"]["id"]):
        return -500.0
    
    # Get legal moves after first move
    legal_next = _get_legal_moves(state_after_1)
    
    if not legal_next:
        return -1000.0  # Dead end
    
    # Evaluate best continuation
    best_continuation = -float('inf')
    for move2 in legal_next:
        state_after_2 = simulate_move(state_after_1, move2)
        score = evaluate_state(state_after_2, state["you"]["id"])
        best_continuation = max(best_continuation, score)
    
    # Also consider the state after just 1 move
    score_after_1 = evaluate_state(state_after_1, state["you"]["id"])
    
    # Weight: 40% immediate, 60% best continuation
    return 0.4 * score_after_1 + 0.6 * best_continuation


def get_lookahead_scores(state: Dict) -> Dict[str, float]:
    """Get lookahead scores for all legal moves.
    
    Returns dict mapping move -> lookahead score.
    """
    legal = _get_legal_moves(state)
    scores = {}
    
    for move in legal:
        try:
            scores[move] = lookahead_2_moves(state, move)
        except Exception:
            scores[move] = -100.0  # Fallback on error
    
    return scores
