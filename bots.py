import chess
# We need to import the Crazyhouse variant specifically
from chess.variant import CrazyhouseBoard
import random
import itertools
import math
import time
from collections import defaultdict

# --- MCTS Node Class ---
class MCTSNode:
    """
    A node in the Monte Carlo Tree.
    """
    def __init__(self, board: CrazyhouseBoard, parent=None, move=None):
        self.board = board
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0.0
        self.visits = 0

    def is_fully_expanded(self):
        """Checks if all legal moves from this node have been added as children."""
        return len(self.children) == len(list(self.board.legal_moves))

    def is_terminal(self):
        """Checks if the board state is a terminal state (win, loss, or draw)."""
        return self.board.is_game_over()

    def ucb1(self, exploration_constant=0.7):
        """Calculates the UCB1 value for the node."""
        if self.visits == 0:
            return float('inf')
        return (self.wins / self.visits) + exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)

# --- Policy Networks & Evaluation Functions ---
def random_bot(board: chess.variant.CrazyhouseBoard, current_turn: chess.Color):
    """
    A simple policy network that returns a random move.
    This is used for the rollout phase of MCTS.
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None
    return random.choice(legal_moves)
    
def evaluate_board_material(board_to_evaluate: CrazyhouseBoard):
    """
    A helper function to evaluate a Crazyhouse board state based on material count,
    including pieces in hand. This function now correctly handles terminal game states.
    """
    # Check for terminal states first
    if board_to_evaluate.is_game_over():
        result = board_to_evaluate.result()
        if result == "1-0":
            return float('inf')  # White wins, return a very high score
        elif result == "0-1":
            return float('-inf') # Black wins, return a very low score
        else:
            return 0  # Draw

    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
    }
    
    white_score = 0
    black_score = 0
    
    # Evaluate pieces on the board
    for piece_type in piece_values:
        white_score += len(board_to_evaluate.pieces(piece_type, chess.WHITE)) * piece_values[piece_type]
        black_score += len(board_to_evaluate.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]

    # Evaluate pieces in hand (crazyhouse specific)
    white_pocket = board_to_evaluate.pockets[chess.WHITE]
    black_pocket = board_to_evaluate.pockets[chess.BLACK]

    for piece_type, value in piece_values.items():
        white_score += white_pocket.count(piece_type) * (value + 1)
        black_score += black_pocket.count(piece_type) * (value + 1)

    return white_score - black_score

def random_policy_network(board: CrazyhouseBoard, turn: chess.Color):
    """
    A simple policy network that returns a random move.
    This is used for the rollout phase of MCTS.
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None
    return random.choice(legal_moves)

# Tuning constants
MATE_VALUE = 100000
DEFAULT_TIME_LIMIT = 0.5   # seconds (tune per strength / CPU)
MAX_ITERATIVE_DEPTH = 5    # max search depth for iterative deepening
QUIESCENCE_MAX = 128
# piece values should roughly match evaluate_board_material scale (adjust if needed)
PIECE_VALUES = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330, chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000}

# Transposition flags
EXACT = 0
LOWERBOUND = 1
UPPERBOUND = 2

class SearchTimeout(Exception):
    pass

def _evaluate(board):
    """
    Improved static evaluation:
      - base material from your evaluate_board_material(board)
      - small mobility bonus and center proximity bonus
    Returns: positive => White advantage, negative => Black advantage
    """
    base = evaluate_board_material(board) * 100  # user's function
    # mobility (tiny)
    try:
        mobility = len(list(board.legal_moves))
    except Exception:
        mobility = 0
    mobility_bonus = 3 * mobility

    # center proximity bonus: encourage pieces to the center
    center_bonus = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            file = chess.square_file(sq)
            rank = chess.square_rank(sq)
            # distance from center (3.5,3.5)
            dist = abs(file - 3.5) + abs(rank - 3.5)
            bonus = max(0, int((4 - dist) * 4))  # small integer bonus
            center_bonus += bonus if piece.color == chess.WHITE else -bonus

    return base + mobility_bonus + center_bonus

# Global search data (recreated each top-level call)
def minimax_bot(board: chess.variant.CrazyhouseBoard, current_turn: chess.Color, time_limit: float = DEFAULT_TIME_LIMIT):
    """
    Iterative deepening negamax with alpha-beta, TT, quiescence, move-ordering heuristics.
    `board` is mutated with push/pop during search (we assume the caller provided a copy).
    Returns a chess.Move or None.
    """
    # initialize search containers per-move
    transposition_table = {}         # fen -> (depth, value, flag, best_move_uci)
    history_table = defaultdict(int) # move_uci -> score
    killer_moves = defaultdict(list) # ply -> [move_uci, ...]
    node_count = 0
    start_time = time.perf_counter()
    stop_time = start_time + time_limit
    
    print(evaluate_board_material(board))
    print(board.legal_moves)
    print("Searching:")

    # local helpers capture these closures
    def check_time():
        if time.perf_counter() > stop_time:
            raise SearchTimeout()

    def is_capture_on_board(b, mv):
        # b.is_capture works with Crazyhouse moves too
        try:
            return b.is_capture(mv)
        except Exception:
            return False

    def mvv_lva_score(b, mv):
        """Heuristic score for capture ordering: victim_value*100 - attacker_value"""
        if not is_capture_on_board(b, mv):
            return 0
        victim = b.piece_at(mv.to_square)
        attacker = b.piece_at(mv.from_square)
        victim_val = PIECE_VALUES.get(victim.piece_type, 0) if victim else 0
        attacker_val = PIECE_VALUES.get(attacker.piece_type, 0) if attacker else 0
        return victim_val * 100 - attacker_val

    def generate_ordered_moves(b, ply, tt_best_uci=None):
        """Order moves: TT best -> captures (MVV-LVA) -> promotions -> killers -> history"""
        moves = list(b.legal_moves)
        scored = []
        for mv in moves:
            score = 0
            u = mv.uci()
            # TT best
            if tt_best_uci and u == tt_best_uci:
                score += 10_000_000
            # captures
            if is_capture_on_board(b, mv):
                score += 100_000 + mvv_lva_score(b, mv)
            # promotions
            if getattr(mv, "promotion", None):
                score += 80_000
            # killer
            if u in killer_moves.get(ply, []):
                score += 40_000
            # history heuristic
            score += history_table.get(u, 0)
            scored.append((score, mv))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [mv for _, mv in scored]

    def quiescence(b, alpha, beta, depth=0):
        nonlocal node_count
        check_time()
        node_count += 1

        # Depth limit safeguard
        if depth >= QUIESCENCE_MAX:
            return _evaluate(b) if b.turn == chess.WHITE else -_evaluate(b)

        # Repetition / variant draw check (important in Crazyhouse)
        if b.is_repetition(2) or b.is_variant_draw():
            return 0

        # Static evaluation from perspective of side to move
        eval_raw = _evaluate(b)
        value = eval_raw if b.turn == chess.WHITE else -eval_raw

        if value >= beta:
            return value
        if alpha < value:
            alpha = value

        # Generate tactical moves: captures or promotions only
        moves = [m for m in b.legal_moves if is_capture_on_board(b, m) or getattr(m, "promotion", None)]
        # Order by MVV-LVA
        moves.sort(key=lambda mv: mvv_lva_score(b, mv), reverse=True)

        for mv in moves:
            # Delta pruning: skip obviously losing captures
            victim = b.piece_at(mv.to_square)
            if victim and PIECE_VALUES.get(victim.piece_type, 0) + 200 < alpha - value:
                continue

            check_time()
            b.push(mv)
            score = -quiescence(b, -beta, -alpha, depth+1)
            b.pop()

            if score >= beta:
                return score
            if score > alpha:
                alpha = score
        return alpha

    def negamax(b, depth, alpha, beta, ply):
        """
        Returns score from perspective of the side to move (higher is better for side to move).
        """
        nonlocal node_count
        check_time()
        node_count += 1

        # Terminal checks
        if b.is_game_over():
            if b.is_checkmate():
                # side to move is checkmated => large negative
                return -MATE_VALUE
            else:
                return 0  # draw/stalemate

        fen = b.fen()
        # Transposition lookup
        tt_entry = transposition_table.get(fen)
        if tt_entry and tt_entry[0] >= depth:
            entry_depth, entry_value, entry_flag, entry_best = tt_entry
            if entry_flag == EXACT:
                return entry_value
            if entry_flag == LOWERBOUND:
                alpha = max(alpha, entry_value)
            elif entry_flag == UPPERBOUND:
                beta = min(beta, entry_value)
            if alpha >= beta:
                return entry_value

        if depth == 0:
            return quiescence(b, alpha, beta)

        original_alpha = alpha
        best_value = -float("inf")
        best_move_for_node = None

        # use tt best move for ordering if present
        tt_best_uci = tt_entry[3] if tt_entry else None
        moves = generate_ordered_moves(b, ply, tt_best_uci)

        for mv in moves:
            try:
                check_time()
            except SearchTimeout:
                raise

            capture = is_capture_on_board(b, mv)
            b.push(mv)
            val = -negamax(b, depth - 1, -beta, -alpha, ply + 1)
            b.pop()

            if val > best_value:
                best_value = val
                best_move_for_node = mv

            if val > alpha:
                alpha = val

            # cutoff
            if alpha >= beta:
                # update history/killer for non-captures
                if not capture:
                    u = mv.uci()
                    # killer: keep up to 2 killers
                    km = killer_moves[ply]
                    if u not in km:
                        km.insert(0, u)
                        if len(km) > 2:
                            km.pop()
                    # history
                    history_table[u] += 2 ** depth
                break

        # store in TT
        if best_value <= original_alpha:
            flag = UPPERBOUND
        elif best_value >= beta:
            flag = LOWERBOUND
        else:
            flag = EXACT
        transposition_table[fen] = (depth, best_value, flag, best_move_for_node.uci() if best_move_for_node else None)
        return best_value

    # Root: iterative deepening
    best_move = None
    try:
        for target_depth in range(1, MAX_ITERATIVE_DEPTH + 1):
            # Limit iterative depth by MAX_ITERATIVE_DEPTH and time limit
            # Root search using negamax framework (we do a normal root loop to keep best_move object)
            alpha = -float("inf")
            beta = float("inf")
            root_moves = list(board.legal_moves)
            # order root moves using quick heuristic
            # try to use any TT best move for root fen
            root_tt = transposition_table.get(board.fen())
            tt_best_uci = root_tt[3] if root_tt else None
            root_moves = generate_ordered_moves(board, 0, tt_best_uci)

            moves_evals = {mv: "ND" for mv in root_moves}
            local_best = None
            for mv in root_moves:
                check_time()
                print(board.san(mv), target_depth, end=" ")
                board.push(mv)
                try:
                    score = -negamax(board, target_depth - 1, -beta, -alpha, 1)
                    moves_evals[mv] = score
                except SearchTimeout:
                    board.pop()
                    raise
                board.pop()

                if local_best is None or score > alpha:
                    alpha = score
                    local_best = mv
                # normal alpha-beta pruning at root (rare with full window)
                print(moves_evals[mv])
            if local_best:
                best_move = local_best
            # keep going deeper until timeout or max depth
    except SearchTimeout:
        # timed out — return best_move found so far
        pass
    except Exception as e:
        # safety: any unexpected exception -> fallback to random legal move
        print("Search exception:", e)
        pass

    # final fallback if nothing found
    if best_move is None:
        moves = list(board.legal_moves)
        if moves:
            print("random move")
            return random.choice(moves)
        return None

    return best_move

def minimax_eval(board: chess.variant.CrazyhouseBoard, current_turn: chess.Color, time_limit: float = DEFAULT_TIME_LIMIT):
    """
    Iterative deepening negamax with alpha-beta, TT, quiescence, move-ordering heuristics.
    `board` is mutated with push/pop during search (we assume the caller provided a copy).
    Returns a chess.Move or None.
    """
    # initialize search containers per-move
    transposition_table = {}         # fen -> (depth, value, flag, best_move_uci)
    history_table = defaultdict(int) # move_uci -> score
    killer_moves = defaultdict(list) # ply -> [move_uci, ...]
    node_count = 0
    start_time = time.perf_counter()
    stop_time = start_time + time_limit
    
    print(evaluate_board_material(board))
    print(board.legal_moves)
    print("Searching:")

    # local helpers capture these closures
    def check_time():
        if time.perf_counter() > stop_time:
            raise SearchTimeout()

    def is_capture_on_board(b, mv):
        # b.is_capture works with Crazyhouse moves too
        try:
            return b.is_capture(mv)
        except Exception:
            return False

    def mvv_lva_score(b, mv):
        """Heuristic score for capture ordering: victim_value*100 - attacker_value"""
        if not is_capture_on_board(b, mv):
            return 0
        victim = b.piece_at(mv.to_square)
        attacker = b.piece_at(mv.from_square)
        victim_val = PIECE_VALUES.get(victim.piece_type, 0) if victim else 0
        attacker_val = PIECE_VALUES.get(attacker.piece_type, 0) if attacker else 0
        return victim_val * 100 - attacker_val

    def generate_ordered_moves(b, ply, tt_best_uci=None):
        """Order moves: TT best -> captures (MVV-LVA) -> promotions -> killers -> history"""
        moves = list(b.legal_moves)
        scored = []
        for mv in moves:
            score = 0
            u = mv.uci()
            # TT best
            if tt_best_uci and u == tt_best_uci:
                score += 10_000_000
            # captures
            if is_capture_on_board(b, mv):
                score += 100_000 + mvv_lva_score(b, mv)
            # promotions
            if getattr(mv, "promotion", None):
                score += 80_000
            # killer
            if u in killer_moves.get(ply, []):
                score += 40_000
            # history heuristic
            score += history_table.get(u, 0)
            scored.append((score, mv))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [mv for _, mv in scored]

    def quiescence(b, alpha, beta, depth=0):
        nonlocal node_count
        check_time()
        node_count += 1

        # Depth limit safeguard
        if depth >= QUIESCENCE_MAX:
            return _evaluate(b) if b.turn == chess.WHITE else -_evaluate(b)

        # Repetition / variant draw check (important in Crazyhouse)
        if b.is_repetition(2) or b.is_variant_draw():
            return 0

        # Static evaluation from perspective of side to move
        eval_raw = _evaluate(b)
        value = eval_raw if b.turn == chess.WHITE else -eval_raw

        if value >= beta:
            return value
        if alpha < value:
            alpha = value

        # Generate tactical moves: captures or promotions only
        moves = [m for m in b.legal_moves if is_capture_on_board(b, m) or getattr(m, "promotion", None)]
        # Order by MVV-LVA
        moves.sort(key=lambda mv: mvv_lva_score(b, mv), reverse=True)

        for mv in moves:
            # Delta pruning: skip obviously losing captures
            victim = b.piece_at(mv.to_square)
            if victim and PIECE_VALUES.get(victim.piece_type, 0) + 200 < alpha - value:
                continue

            check_time()
            b.push(mv)
            score = -quiescence(b, -beta, -alpha, depth+1)
            b.pop()

            if score >= beta:
                return score
            if score > alpha:
                alpha = score
        return alpha

    def negamax(b, depth, alpha, beta, ply):
        """
        Returns score from perspective of the side to move (higher is better for side to move).
        """
        nonlocal node_count
        check_time()
        node_count += 1

        # Terminal checks
        if b.is_game_over():
            if b.is_checkmate():
                # side to move is checkmated => large negative
                return -MATE_VALUE
            else:
                return 0  # draw/stalemate

        fen = b.fen()
        # Transposition lookup
        tt_entry = transposition_table.get(fen)
        if tt_entry and tt_entry[0] >= depth:
            entry_depth, entry_value, entry_flag, entry_best = tt_entry
            if entry_flag == EXACT:
                return entry_value
            if entry_flag == LOWERBOUND:
                alpha = max(alpha, entry_value)
            elif entry_flag == UPPERBOUND:
                beta = min(beta, entry_value)
            if alpha >= beta:
                return entry_value

        if depth == 0:
            return quiescence(b, alpha, beta)

        original_alpha = alpha
        best_value = -float("inf")
        best_move_for_node = None

        # use tt best move for ordering if present
        tt_best_uci = tt_entry[3] if tt_entry else None
        moves = generate_ordered_moves(b, ply, tt_best_uci)

        for mv in moves:
            try:
                check_time()
            except SearchTimeout:
                raise

            capture = is_capture_on_board(b, mv)
            b.push(mv)
            val = -negamax(b, depth - 1, -beta, -alpha, ply + 1)
            b.pop()

            if val > best_value:
                best_value = val
                best_move_for_node = mv

            if val > alpha:
                alpha = val

            # cutoff
            if alpha >= beta:
                # update history/killer for non-captures
                if not capture:
                    u = mv.uci()
                    # killer: keep up to 2 killers
                    km = killer_moves[ply]
                    if u not in km:
                        km.insert(0, u)
                        if len(km) > 2:
                            km.pop()
                    # history
                    history_table[u] += 2 ** depth
                break

        # store in TT
        if best_value <= original_alpha:
            flag = UPPERBOUND
        elif best_value >= beta:
            flag = LOWERBOUND
        else:
            flag = EXACT
        transposition_table[fen] = (depth, best_value, flag, best_move_for_node.uci() if best_move_for_node else None)
        return best_value

    # Root: iterative deepening
    best_move = None
    try:
        for target_depth in range(1, MAX_ITERATIVE_DEPTH + 1):
            # Limit iterative depth by MAX_ITERATIVE_DEPTH and time limit
            # Root search using negamax framework (we do a normal root loop to keep best_move object)
            alpha = -float("inf")
            beta = float("inf")
            root_moves = list(board.legal_moves)
            # order root moves using quick heuristic
            # try to use any TT best move for root fen
            root_tt = transposition_table.get(board.fen())
            tt_best_uci = root_tt[3] if root_tt else None
            root_moves = generate_ordered_moves(board, 0, tt_best_uci)

            moves_evals = {mv: "ND" for mv in root_moves}
            local_best = None
            for mv in root_moves:
                check_time()
                print(board.san(mv), target_depth, end=" ")
                board.push(mv)
                try:
                    score = -negamax(board, target_depth - 1, -beta, -alpha, 1)
                    moves_evals[mv] = score
                except SearchTimeout:
                    board.pop()
                    raise
                board.pop()

                if local_best is None or score > alpha:
                    alpha = score
                    local_best = mv
                # normal alpha-beta pruning at root (rare with full window)
                print(moves_evals[mv])
            if local_best:
                best_move = local_best
            # keep going deeper until timeout or max depth
    except SearchTimeout:
        # timed out — return best_move found so far
        pass
    except Exception as e:
        # safety: any unexpected exception -> fallback to random legal move
        print("Search exception:", e)
        pass

    # final fallback if nothing found
    if best_move is None:
        moves = list(board.legal_moves)
        if moves:
            print("random move")
            return moves_evals[random.choice(moves)]
        return None

    return moves_evals[best_move]


def mcts_bot(current_board, current_turn, iterations=50, policy_network=random_policy_network):
    root = MCTSNode(current_board)
    
    for _ in range(iterations):
        node = root
        
        # Selection & Expansion as before...
        while not node.is_terminal() and node.is_fully_expanded():
            node = max(node.children, key=lambda c: c.ucb1())
        if not node.is_terminal():
            unexplored_moves = [move for move in node.board.legal_moves if move not in [child.move for child in node.children]]
            if unexplored_moves:
                move = random.choice(unexplored_moves)
                new_board = node.board.copy()
                new_board.push(move)
                new_child = MCTSNode(new_board, parent=node, move=move)
                node.children.append(new_child)
                node = new_child
        
        # Simulation (Rollout)
        simulation_board = node.board.copy()
        move_count = 0
        while not simulation_board.is_game_over() and move_count < 15:
            move = policy_network(simulation_board, simulation_board.turn)
            if not move or move not in simulation_board.legal_moves:
                break
            simulation_board.push(move)
            move_count += 1
        
        # Determine outcome or evaluate if 15-move limit reached
        outcome = simulation_board.result()
        if outcome == '*':  # game not finished after 15 moves
            # Evaluate material to estimate who is ahead
            material_score = evaluate_board_material(simulation_board) if current_turn == chess.WHITE else -evaluate_board_material(simulation_board)
            # Normalize to score between 0 and 1 (assuming max material roughly 39)
            score = 0.5 + (material_score / 39) * 0.5
            # Clamp score between 0 and 1
            score = max(0.0, min(1.0, score))
        else:
            if outcome == "1-0":  # White wins
                score = 1 if current_turn == chess.WHITE else 0
            elif outcome == "0-1":  # Black wins
                score = 1 if current_turn == chess.BLACK else 0
            else:  # Draw or no result
                score = 0.5
        
        print(outcome, score)
        # Backpropagation
        while node is not None:
            node.visits += 1
            node.wins += score
            node = node.parent
            if node and node.board.turn != current_turn:
                score = 1 - score

    # Return best move as before
    if not root.children:
        return minimax_bot(current_board, current_turn, 0.1)
    best_child = max(root.children, key=lambda c: c.visits)
    return best_child.move

def random_mcts_bot(current_board, current_turn):
    """An MCTS bot using the random policy network for rollouts."""
    return mcts_bot(current_board, current_turn, 20, random_policy_network)

def material_policy_network(current_board, current_turn):
    legal_moves = list(current_board.legal_moves)
    move_evals = {mv: 0 for mv in legal_moves}
    for mv in move_evals.keys():
        current_board.push(mv)
        move_evals[mv] = evaluate_board_material(current_board) if current_turn == chess.WHITE else -evaluate_board_material(current_board)
        current_board.pop()
    best_move = max(move_evals, key=move_evals.get)
    return best_move
    
def material_mcts_bot(current_board, current_turn):
    """An MCTS bot using the material policy network for rollouts."""
    return mcts_bot(current_board, current_turn, 160, material_policy_network)

def minimax_mcts_bot(current_board, current_turn):
    """An MCTS bot using the minimax bot as a policy network for rollouts."""
    return mcts_bot(current_board, current_turn, 160, lambda x, y: minimax_bot(x,y,0.1))

# --- Bot Initialization ---
# This list defines all the bots that will participate in the tournament.
# The format is a tuple: (Bot Name, Bot Function, Initial ELO)
additional_bots = [
    ("Random Bot", random_bot, 1500),
    ("MCTS Material Bot", material_mcts_bot, 1600),
    ("Minimax Bot", minimax_bot, 1700),
    ("Minimax MCTS Bot", minimax_mcts_bot, 1800),
]

# --- ELO Calculation (simplified) ---

def calculate_elo_change(elo1, elo2, outcome, k_factor=32):
    """
    Calculates the ELO change based on the outcome of a game.
    Outcome: 1 for win, 0.5 for draw, 0 for loss.
    """
    expected_score = 1 / (1 + 10**((elo2 - elo1) / 400))
    return k_factor * (outcome - expected_score)

# --- Round Robin Logic ---

def run_game(player1_bot, player2_bot):
    """
    Simulates a single Crazyhouse chess game between two bots.
    Returns: 1 for player1 win, 0 for player2 win, 0.5 for draw.
    """
    # We now create an instance of the CrazyhouseBoard class.
    board = CrazyhouseBoard()
    
    # Track the outcome of the game
    game_outcome = None

    # We use is_game_over(), which is the correct method for checking game end
    # for all variants of chess.
    while not board.is_game_over():
        # Player 1 (White) turn
        move = player1_bot(board, chess.WHITE)
        if move and move in board.legal_moves:
            board.push(move)
        else:
            # Player 1 made an illegal move or resigned, Player 2 wins
            game_outcome = 0
            break

        if board.is_game_over():
            break

        # Player 2 (Black) turn
        move = player2_bot(board, chess.BLACK)
        if move and move in board.legal_moves:
            board.push(move)
        else:
            # Player 2 made an illegal move or resigned, Player 1 wins
            game_outcome = 1
            break

    # If the loop finished naturally, determine the outcome from the board state
    if game_outcome is None:
        result = board.result()
        if result == "1-0":
            game_outcome = 1
        elif result == "0-1":
            game_outcome = 0
        else: # "1/2-1/2" for a draw
            game_outcome = 0.5
    
    return game_outcome

def run_round_robin(bots_with_elo):
    """
    Runs a single round robin tournament where each bot plays every other bot twice
    (once as white, once as black).
    Returns a dictionary of updated ELO ratings.
    """
    num_bots = len(bots_with_elo)
    bot_elos = {name: elo for name, _, elo in bots_with_elo}
    
    # Use itertools to generate all unique pairs of bots
    bot_pairs = list(itertools.combinations(range(num_bots), 2))
    
    for i, j in bot_pairs:
        bot1_name, bot1_func, _ = bots_with_elo[i]
        bot2_name, bot2_func, _ = bots_with_elo[j]

        # Game 1: bot1 (white) vs bot2 (black)
        print(f"Playing game: {bot1_name} (W) vs {bot2_name} (B)")
        outcome1 = run_game(bot1_func, bot2_func)
        
        # Update ELOs based on game 1
        elo1_change = calculate_elo_change(bot_elos[bot1_name], bot_elos[bot2_name], outcome1)
        elo2_change = calculate_elo_change(bot_elos[bot2_name], bot_elos[bot1_name], 1 - outcome1)
        
        bot_elos[bot1_name] += elo1_change
        bot_elos[bot2_name] += elo2_change

        # Game 2: bot2 (white) vs bot1 (black)
        print(f"Playing game: {bot2_name} (W) vs {bot1_name} (B)")
        outcome2 = run_game(bot2_func, bot1_func)
        
        # Update ELOs based on game 2
        elo2_change_2 = calculate_elo_change(bot_elos[bot2_name], bot_elos[bot1_name], outcome2)
        elo1_change_2 = calculate_elo_change(bot_elos[bot1_name], bot_elos[bot2_name], 1 - outcome2)
        
        bot_elos[bot2_name] += elo2_change_2
        bot_elos[bot1_name] += elo1_change_2

    return bot_elos

def find_simple_positions(player1_bot, player2_bot):
    """
    Simulates a single Crazyhouse chess game between two bots.
    Returns: 1 for player1 win, 0 for player2 win, 0.5 for draw.
    """
    # We now create an instance of the CrazyhouseBoard class.
    board = CrazyhouseBoard()
    
    # Track the outcome of the game
    game_outcome = None

    # We use is_game_over(), which is the correct method for checking game end
    # for all variants of chess.
    simple_positions = []
    while not board.is_game_over():
        # Player 1 (White) turn
        branching_factor = len(board.legal_moves)
        if branching_factor <= 15:
            simple_positions.append(board)
            
        move = player1_bot(board, chess.WHITE)
        if move and move in board.legal_moves:
            board.push(move)
        else:
            # Player 1 made an illegal move or resigned, Player 2 wins
            game_outcome = 0
            break

        if board.is_game_over():
            break

        # Player 2 (Black) turn
        branching_factor = len(board.legal_moves)
        if branching_factor <= 15:
            simple_positions.append(board)
        move = player2_bot(board, chess.BLACK)
        if move and move in board.legal_moves:
            board.push(move)
        else:
            # Player 2 made an illegal move or resigned, Player 1 wins
            game_outcome = 1
            break
        
    return simple_positions
# --- Main Program Loop ---

def main():
    """
    The main loop that runs the tournament over multiple rounds.
    """
    total_rounds = 5  # Set the total number of rounds here
    cur_round = 0
    
    # Initialize ELOs from the initial bot list
    bots_with_current_elo = [(name, func, elo) for name, func, elo in additional_bots]

    simple_positions = find_simple_positions(random_bot, random_bot)
    for b in simple_positions:
        # pred = Eval_model(b, b.turn)
        eval = minimax_eval(b, b.turn, 0.1)
        # loss = abs(eval - pred)
        # backpropagate to train model
        
        
        
        

if __name__ == "__main__":
    main()
