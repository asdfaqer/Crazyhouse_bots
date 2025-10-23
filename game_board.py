import pygame
import chess.variant
import os
import random
import threading
import time

# --- Configuration ---
BOARD_WIDTH = 800
PANEL_WIDTH = 200
WIDTH, HEIGHT = BOARD_WIDTH + PANEL_WIDTH, 800
BOARD_SIZE = 8
SQUARE_SIZE = BOARD_WIDTH // BOARD_SIZE
POCKET_PIECE_SIZE = 50
FPS = 60

# --- Colors ---
WHITE_SQUARE = (238, 238, 210)
BLACK_SQUARE = (118, 150, 86)
HIGHLIGHT_COLOR = (255, 255, 0)
WHITE_TEXT = (255, 255, 255)
BLACK_TEXT = (0, 0, 0)
BACKGROUND_COLOR = (40, 40, 40)
COUNT_TEXT_COLOR = (200, 200, 200)
DROP_HIGHLIGHT_COLOR = (255, 0, 0)
PROMOTION_BG = (100, 100, 100)
PROMOTION_OUTLINE = (255, 255, 255)
BUTTON_COLOR = (60, 60, 60)
BUTTON_HOVER_COLOR = (80, 80, 80)
TITLE_COLOR = (255, 255, 255)
GAME_OVER_TEXT_COLOR = (255, 255, 255)
DROPDOWN_BG_COLOR = (70, 70, 70)
DROPDOWN_TEXT_COLOR = (255, 255, 255)
DROPDOWN_HOVER_COLOR = (90, 90, 90)
ELO_COLOR = (150, 150, 150)
ARROW_COLOR = (255, 0, 255)

# --- Game States ---
START_SCREEN = 0
PLAYING = 1
GAME_OVER = 2
BOT_SELECTION = 3

# --- Pygame Setup ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Crazyhouse")
clock = pygame.time.Clock()

# --- Fonts (Initialized Globally) ---
title_font_lg = pygame.font.Font(None, 100)
title_font_sm = pygame.font.Font(None, 70)
button_font_lg = pygame.font.Font(None, 50)
button_font_sm = pygame.font.Font(None, 40)
label_font = pygame.font.Font(None, 30)
elo_font = pygame.font.Font(None, 25)
font_board = pygame.font.Font(None, SQUARE_SIZE)
font_pocket = pygame.font.Font(None, POCKET_PIECE_SIZE)

# --- Piece Images ---
def load_piece_images():
    images = {}
    pieces = ['p', 'n', 'b', 'r', 'q', 'k']
    colors = ['w', 'b']
    for color in colors:
        for piece in pieces:
            symbol = piece.upper() if color == 'w' else piece
            filename = f"{color}{piece}.png"
            path = os.path.join("pieces", filename)
            text_color = WHITE_TEXT if color == 'w' else BLACK_TEXT
            try:
                image_board = pygame.image.load(path)
                images[symbol] = pygame.transform.scale(image_board, (SQUARE_SIZE, SQUARE_SIZE))
                images[symbol + '_pocket'] = pygame.transform.scale(image_board, (POCKET_PIECE_SIZE, POCKET_PIECE_SIZE))
            except (pygame.error, FileNotFoundError) as e:
                print(f"Error loading image '{path}': {e}")
                images[symbol] = font_board.render(symbol, True, text_color)
                images[symbol + '_pocket'] = font_pocket.render(symbol, True, text_color)
    return images

piece_images = load_piece_images()

# --- Game State Variables ---
game_state = START_SCREEN
board = chess.variant.CrazyhouseBoard()
board_lock = threading.Lock()  # lock to protect fen reads and pushes
selected_square = None
valid_moves = []
pocket_selected = None
promoting_move = None
promotion_rects = []
promotion_options = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
is_white_human = False
is_black_human = False
game_over_message = ""
white_bot_function = None
black_bot_function = None
is_board_flipped = False

# --- Arrow drawing state ---
arrows = []  # list of tuples ((file, rank), (file, rank)) in board coords 0..7
arrow_drawing = False
arrow_start_pixel = None
arrow_current_pixel = None

# --- Bot Functions and List ---
def random_bot(current_board, current_turn):
    legal_moves = list(current_board.legal_moves)
    if legal_moves:
        time.sleep(0.2)  # small delay to make thinking visible
        return random.choice(legal_moves)
    return None

# Attempt to import additional bots from a 'bots.py' file.
try:
    from bots import additional_bots
    bot_list = [
        ("Human", None, None),
    ] + additional_bots
except ImportError:
    print("Warning: Could not import 'bots.py'. Using default bots.")
    bot_list = [
        ("Human", None, None),
        ("Random Bot", random_bot, 800)
    ]

# --- UI elements for bot selection screen ---
dropdown_buttons = []
selected_white_bot_index = 0
selected_black_bot_index = 0
is_white_dropdown_open = False
is_black_dropdown_open = False

# --- Bot worker class (runs in a separate thread) ---
class BotWorker(threading.Thread):
    def __init__(self, bot_fn, board_snapshot_fen, color, result_container, result_lock):
        super().__init__(daemon=True)
        self.bot_fn = bot_fn
        self.board_fen = board_snapshot_fen
        self.color = color
        self.result_container = result_container  # dict to put result into
        self.result_lock = result_lock

    def run(self):
        try:
            # Construct board copy from fen and pass to bot
            # use keyword 'fen=' for safety
            temp_board = chess.variant.CrazyhouseBoard(fen=self.board_fen)
            move = self.bot_fn(temp_board, self.color)
            # allow bot to return a Move or UCI string
            with self.result_lock:
                self.result_container['move'] = move
                self.result_container['fen_at_start'] = self.board_fen
        except Exception as e:
            print(f"Bot thread error: {e}")
            with self.result_lock:
                self.result_container['move'] = None
                self.result_container['fen_at_start'] = self.board_fen

# --- Drawing Functions ---
def draw_start_screen(mouse_pos):
    screen.fill(BACKGROUND_COLOR)
    title_text = title_font_lg.render("Crazyhouse Chess", True, TITLE_COLOR)
    title_rect = title_text.get_rect(center=(WIDTH // 2, HEIGHT // 4))
    screen.blit(title_text, title_rect)
    button_width, button_height = 300, 70
    create_game_rect = pygame.Rect(0, 0, button_width, button_height)
    create_game_rect.center = (WIDTH // 2, HEIGHT // 2)
    buttons = [("Create Game", create_game_rect)]
    for text, rect in buttons:
        color = BUTTON_HOVER_COLOR if rect.collidepoint(mouse_pos) else BUTTON_COLOR
        pygame.draw.rect(screen, color, rect, border_radius=10)
        text_surface = button_font_lg.render(text, True, WHITE_TEXT)
        text_rect = text_surface.get_rect(center=rect.center)
        screen.blit(text_surface, text_rect)
    return buttons

def draw_bot_selection_screen(mouse_pos):
    global dropdown_buttons
    screen.fill(BACKGROUND_COLOR)
    title_text = title_font_sm.render("Select Players", True, TITLE_COLOR)
    title_rect = title_text.get_rect(center=(WIDTH // 2, HEIGHT // 6))
    screen.blit(title_text, title_rect)
    dropdown_buttons = []
    dropdown_width, dropdown_height = 300, 50
    spacing_x = 40
    y_pos = HEIGHT // 2 - 50
    white_dropdown_x = WIDTH // 2 - dropdown_width - spacing_x // 2
    white_label_text = label_font.render("White Player", True, WHITE_TEXT)
    screen.blit(white_label_text, (white_dropdown_x, y_pos - 30))
    white_dropdown_rect = pygame.Rect(white_dropdown_x, y_pos, dropdown_width, dropdown_height)
    color_white = DROPDOWN_HOVER_COLOR if white_dropdown_rect.collidepoint(mouse_pos) else DROPDOWN_BG_COLOR
    pygame.draw.rect(screen, color_white, white_dropdown_rect, border_radius=10)
    white_bot_name, _, white_elo = bot_list[selected_white_bot_index]
    white_text = button_font_sm.render(white_bot_name, True, DROPDOWN_TEXT_COLOR)
    white_text_rect = white_text.get_rect(center=white_dropdown_rect.center)
    screen.blit(white_text, white_text_rect)
    if white_elo:
        elo_text = elo_font.render(f"({white_elo})", True, ELO_COLOR)
        elo_text_rect = elo_text.get_rect(left=white_text_rect.right + 5, centery=white_text_rect.centery)
        screen.blit(elo_text, elo_text_rect)
    dropdown_buttons.append(("white", white_dropdown_rect))
    if is_white_dropdown_open:
        for i, (bot_name, _, elo) in enumerate(bot_list):
            option_rect = pygame.Rect(white_dropdown_x, y_pos + (i + 1) * dropdown_height, dropdown_width, dropdown_height)
            color = DROPDOWN_HOVER_COLOR if option_rect.collidepoint(mouse_pos) else DROPDOWN_BG_COLOR
            pygame.draw.rect(screen, color, option_rect, border_radius=5)
            option_text = button_font_sm.render(bot_name, True, DROPDOWN_TEXT_COLOR)
            option_text_rect = option_text.get_rect(center=option_rect.center)
            screen.blit(option_text, option_text_rect)
            if elo:
                elo_text = elo_font.render(f"({elo})", True, ELO_COLOR)
                elo_text_rect = elo_text.get_rect(left=option_text_rect.right + 5, centery=option_text_rect.centery)
                screen.blit(elo_text, elo_text_rect)
            dropdown_buttons.append(("white_option", option_rect, i))
    black_dropdown_x = WIDTH // 2 + spacing_x // 2
    black_label_text = label_font.render("Black Player", True, WHITE_TEXT)
    screen.blit(black_label_text, (black_dropdown_x, y_pos - 30))
    black_dropdown_rect = pygame.Rect(black_dropdown_x, y_pos, dropdown_width, dropdown_height)
    color_black = DROPDOWN_HOVER_COLOR if black_dropdown_rect.collidepoint(mouse_pos) else DROPDOWN_BG_COLOR
    pygame.draw.rect(screen, color_black, black_dropdown_rect, border_radius=10)
    black_bot_name, _, black_elo = bot_list[selected_black_bot_index]
    black_text = button_font_sm.render(black_bot_name, True, DROPDOWN_TEXT_COLOR)
    black_text_rect = black_text.get_rect(center=black_dropdown_rect.center)
    screen.blit(black_text, black_text_rect)
    if black_elo:
        elo_text = elo_font.render(f"({black_elo})", True, ELO_COLOR)
        elo_text_rect = elo_text.get_rect(left=black_text_rect.right + 5, centery=black_text_rect.centery)
        screen.blit(elo_text, elo_text_rect)
    dropdown_buttons.append(("black", black_dropdown_rect))
    if is_black_dropdown_open:
        for i, (bot_name, _, elo) in enumerate(bot_list):
            option_rect = pygame.Rect(black_dropdown_x, y_pos + (i + 1) * dropdown_height, dropdown_width, dropdown_height)
            color = DROPDOWN_HOVER_COLOR if option_rect.collidepoint(mouse_pos) else DROPDOWN_BG_COLOR
            pygame.draw.rect(screen, color, option_rect, border_radius=5)
            option_text = button_font_sm.render(bot_name, True, DROPDOWN_TEXT_COLOR)
            option_text_rect = option_text.get_rect(center=option_rect.center)
            screen.blit(option_text, option_text_rect)
            if elo:
                elo_text = elo_font.render(f"({elo})", True, ELO_COLOR)
                elo_text_rect = elo_text.get_rect(left=option_text_rect.right + 5, centery=option_text_rect.centery)
                screen.blit(elo_text, elo_text_rect)
            dropdown_buttons.append(("black_option", option_rect, i))
    confirm_button_rect = pygame.Rect(0, 0, 200, 60)
    confirm_button_rect.center = (WIDTH // 2, HEIGHT - 100)
    confirm_color = BUTTON_HOVER_COLOR if confirm_button_rect.collidepoint(mouse_pos) else BUTTON_COLOR
    pygame.draw.rect(screen, confirm_color, confirm_button_rect, border_radius=10)
    confirm_text = button_font_sm.render("Confirm", True, WHITE_TEXT)
    confirm_text_rect = confirm_text.get_rect(center=confirm_button_rect.center)
    screen.blit(confirm_text, confirm_text_rect)
    dropdown_buttons.append(("confirm", confirm_button_rect))

def draw_game_over_screen(mouse_pos):
    screen.fill(BACKGROUND_COLOR)
    message_font = pygame.font.Font(None, 70)
    button_font = pygame.font.Font(None, 50)
    message_text = message_font.render(game_over_message, True, GAME_OVER_TEXT_COLOR)
    message_rect = message_text.get_rect(center=(WIDTH // 2, HEIGHT // 4))
    screen.blit(message_text, message_rect)
    button_width, button_height = 250, 60
    spacing = 20
    rematch_rect = pygame.Rect(0, 0, button_width, button_height)
    rematch_rect.center = (WIDTH // 2, HEIGHT // 2 - button_height - spacing)
    new_game_rect = pygame.Rect(0, 0, button_width, button_height)
    new_game_rect.center = (WIDTH // 2, HEIGHT // 2)
    quit_rect = pygame.Rect(0, 0, button_width, button_height)
    quit_rect.center = (WIDTH // 2, HEIGHT // 2 + button_height + spacing)
    rematch_color = BUTTON_HOVER_COLOR if rematch_rect.collidepoint(mouse_pos) else BUTTON_COLOR
    pygame.draw.rect(screen, rematch_color, rematch_rect, border_radius=10)
    rematch_text = button_font.render("Rematch", True, WHITE_TEXT)
    rematch_text_rect = rematch_text.get_rect(center=rematch_rect.center)
    screen.blit(rematch_text, rematch_text_rect)
    new_game_color = BUTTON_HOVER_COLOR if new_game_rect.collidepoint(mouse_pos) else BUTTON_COLOR
    pygame.draw.rect(screen, new_game_color, new_game_rect, border_radius=10)
    new_game_text = button_font.render("New Game", True, WHITE_TEXT)
    new_game_text_rect = new_game_text.get_rect(center=new_game_rect.center)
    screen.blit(new_game_text, new_game_text_rect)
    quit_color = BUTTON_HOVER_COLOR if quit_rect.collidepoint(mouse_pos) else BUTTON_COLOR
    pygame.draw.rect(screen, quit_color, quit_rect, border_radius=10)
    quit_text = button_font.render("Quit", True, WHITE_TEXT)
    quit_text_rect = quit_text.get_rect(center=quit_rect.center)
    screen.blit(quit_text, quit_text_rect)
    return rematch_rect, new_game_rect, quit_rect

def draw_board():
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            color = WHITE_SQUARE if (r + c) % 2 == 0 else BLACK_SQUARE
            draw_r = r if not is_board_flipped else 7 - r
            pygame.draw.rect(screen, color, pygame.Rect(c * SQUARE_SIZE, draw_r * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

def draw_pieces():
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            piece = board.piece_at(chess.square(c, 7 - r))
            if piece and (r, c) != selected_square:
                symbol = piece.symbol()
                draw_r = r if not is_board_flipped else 7 - r
                screen.blit(piece_images[symbol], pygame.Rect(c * SQUARE_SIZE, draw_r * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

def draw_pocket():
    pocket_x_start = BOARD_WIDTH + 10
    pocket_spacing = 10
    y_pos = 20
    for piece_type in reversed(chess.PIECE_TYPES):
        count = board.pockets[chess.BLACK].count(piece_type)
        if count > 0:
            piece_symbol = chess.Piece(piece_type, chess.BLACK).symbol()
            piece_image = piece_images[piece_symbol + '_pocket']
            screen.blit(piece_image, (pocket_x_start, y_pos))
            count_text = label_font.render(f"x{count}", True, COUNT_TEXT_COLOR)
            text_x = pocket_x_start + POCKET_PIECE_SIZE + pocket_spacing
            text_y = y_pos + (POCKET_PIECE_SIZE - count_text.get_height()) // 2
            screen.blit(count_text, (text_x, text_y))
            y_pos += POCKET_PIECE_SIZE + pocket_spacing
    y_pos = HEIGHT - 20
    for piece_type in reversed(chess.PIECE_TYPES):
        count = board.pockets[chess.WHITE].count(piece_type)
        if count > 0:
            y_pos -= (POCKET_PIECE_SIZE + pocket_spacing)
            piece_symbol = chess.Piece(piece_type, chess.WHITE).symbol()
            piece_image = piece_images[piece_symbol + '_pocket']
            screen.blit(piece_image, (pocket_x_start, y_pos))
            count_text = label_font.render(f"x{count}", True, COUNT_TEXT_COLOR)
            text_x = pocket_x_start + POCKET_PIECE_SIZE + pocket_spacing
            text_y = y_pos + (POCKET_PIECE_SIZE - count_text.get_height()) // 2
            screen.blit(count_text, (text_x, text_y))

def draw_highlights():
    if selected_square:
        r, c = selected_square
        draw_r = r if not is_board_flipped else 7 - r
        pygame.draw.rect(screen, HIGHLIGHT_COLOR, pygame.Rect(c * SQUARE_SIZE, draw_r * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 5)
    if pocket_selected:
        legal_drops = board.generate_legal_drops()
        drop_squares = [move.to_square for move in legal_drops if move.drop == pocket_selected]
        for square in drop_squares:
            r, c = 7 - chess.square_rank(square), chess.square_file(square)
            draw_r = r if not is_board_flipped else 7 - r
            pygame.draw.circle(screen, DROP_HIGHLIGHT_COLOR, (c * SQUARE_SIZE + SQUARE_SIZE // 2, draw_r * SQUARE_SIZE + SQUARE_SIZE // 2), 10)
    else:
        for move in valid_moves:
            end_square = move.to_square
            r, c = 7 - chess.square_rank(end_square), chess.square_file(end_square)
            draw_r = r if not is_board_flipped else 7 - r
            pygame.draw.circle(screen, HIGHLIGHT_COLOR, (c * SQUARE_SIZE + SQUARE_SIZE // 2, draw_r * SQUARE_SIZE + SQUARE_SIZE // 2), 10)

def draw_promotion_options():
    global promotion_rects
    end_square = promoting_move.to_square
    c, r = chess.square_file(end_square), 7 - chess.square_rank(end_square)
    draw_r = r if not is_board_flipped else 7 - r
    menu_height = SQUARE_SIZE * 4
    menu_width = SQUARE_SIZE
    if is_board_flipped:
        if board.turn == chess.BLACK:
            y_start = draw_r * SQUARE_SIZE
        else:
            y_start = draw_r * SQUARE_SIZE - menu_height
    else:
        if board.turn == chess.WHITE:
            y_start = draw_r * SQUARE_SIZE
        else:
            y_start = draw_r * SQUARE_SIZE - menu_height
    menu_rect = pygame.Rect(c * SQUARE_SIZE, y_start, menu_width, menu_height)
    pygame.draw.rect(screen, PROMOTION_BG, menu_rect)
    pygame.draw.rect(screen, PROMOTION_OUTLINE, menu_rect, 2)
    promotion_rects = []
    piece_color = board.turn
    for i, piece_type in enumerate(promotion_options):
        symbol = chess.Piece(piece_type, piece_color).symbol()
        piece_image = piece_images[symbol]
        y_pos = y_start + i * SQUARE_SIZE
        rect = pygame.Rect(c * SQUARE_SIZE, y_pos, SQUARE_SIZE, SQUARE_SIZE)
        screen.blit(piece_image, rect)
        promotion_rects.append((rect, piece_type))

def board_coords_to_pixel_center(file, rank):
    """
    Convert board coords (file 0..7, rank 0..7 where rank 0 is White's first rank)
    to screen pixel center, accounting for is_board_flipped.
    """
    # adjust file/rank for flipped view
    screen_file = file if not is_board_flipped else 7 - file
    screen_row = 7 - rank if not is_board_flipped else rank
    x = screen_file * SQUARE_SIZE + SQUARE_SIZE // 2
    y = screen_row * SQUARE_SIZE + SQUARE_SIZE // 2
    return x, y

def draw_arrows():
    # draw stored arrows
    for start_sq, end_sq in arrows:
        sx, sy = board_coords_to_pixel_center(start_sq[0], start_sq[1])
        ex, ey = board_coords_to_pixel_center(end_sq[0], end_sq[1])
        pygame.draw.line(screen, ARROW_COLOR, (sx, sy), (ex, ey), 4)
        # small arrowhead
        draw_arrowhead((sx, sy), (ex, ey), ARROW_COLOR)

    # draw current dragging arrow (screen pixels)
    if arrow_drawing and arrow_start_pixel and arrow_current_pixel:
        pygame.draw.line(screen, ARROW_COLOR, arrow_start_pixel, arrow_current_pixel, 3)
        draw_arrowhead(arrow_start_pixel, arrow_current_pixel, ARROW_COLOR)

def draw_arrowhead(start, end, color):
    # simple arrowhead drawing
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    dist = max((dx*dx + dy*dy) ** 0.5, 1)
    ux, uy = dx / dist, dy / dist
    # two side points
    size = 12
    left = (end[0] - ux * size - uy * (size/2), end[1] - uy * size + ux * (size/2))
    right = (end[0] - ux * size + uy * (size/2), end[1] - uy * size - ux * (size/2))
    pygame.draw.polygon(screen, color, [end, left, right])

def draw_game_screen():
    screen.fill(BACKGROUND_COLOR)
    draw_board()
    draw_highlights()
    draw_pieces()
    draw_pocket()
    draw_arrows()
    if promoting_move:
        draw_promotion_options()
    pygame.display.flip()


# Shared state for bot threads
bot_result_lock = threading.Lock()
bot_result = {
    'move': None,
    'fen_at_start': None
}
bot_thread = None

last_bot_fen = None  # global variable to track last spawned bot board

def spawn_bot_if_needed():
    global bot_thread, bot_result, last_bot_fen
    with board_lock:
        cur_turn = board.turn
        current_fen = board.fen()
    # If human turn, no bot spawn
    if (cur_turn == chess.WHITE and is_white_human) or (cur_turn == chess.BLACK and is_black_human):
        last_bot_fen = None  # reset so bot can spawn next bot turn
        return
    bot_fn = white_bot_function if cur_turn == chess.WHITE else black_bot_function
    if not bot_fn:
        last_bot_fen = None
        return
    # If a thread is already running, don't start another
    if bot_thread and bot_thread.is_alive():
        return
    # Check if we've already spawned bot for this board state
    if last_bot_fen == current_fen:
        return  # Already spawned for this turn

    # reset bot_result
    with bot_result_lock:
        bot_result['move'] = None
        bot_result['fen_at_start'] = None

    # spawn thread
    with board_lock:
        fen_snapshot = board.fen()
    worker = BotWorker(bot_fn, fen_snapshot, cur_turn, bot_result, bot_result_lock)
    bot_thread = worker
    last_bot_fen = fen_snapshot
    worker.start()


def try_apply_bot_result():
    """
    If a bot thread produced a move, apply it only if the board fen still matches the fen
    the bot started from (so we don't apply stale moves).
    Returns True if applied.
    """
    global bot_result, bot_thread
    with bot_result_lock:
        move = bot_result.get('move')
        fen_at_start = bot_result.get('fen_at_start')
    if move is None:
        return False

    # accept either a Move object or UCI string
    mv_obj = None
    try:
        if isinstance(move, str):
            mv_obj = chess.Move.from_uci(move)
        else:
            mv_obj = move
    except Exception:
        mv_obj = None

    # double-check board hasn't changed and the move is legal
    with board_lock:
        current_fen = board.fen()
        if fen_at_start == current_fen and mv_obj is not None and mv_obj in board.legal_moves:
            board.push(mv_obj)
            # clear container and thread ref
            with bot_result_lock:
                bot_result['move'] = None
                bot_result['fen_at_start'] = None
            # mark thread done so spawn_bot_if_needed can make another if needed
            bot_thread = None
            return True
        else:
            # either board changed or move illegal now; discard
            with bot_result_lock:
                bot_result['move'] = None
                bot_result['fen_at_start'] = None
            bot_thread = None
            return False

def square_to_screen_coords(square, flipped=False):
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    if flipped:
        file = 7 - file
        rank = 7 - rank
    return file * SQUARE_SIZE, (7 - rank) * SQUARE_SIZE

# --- Game Loop ---
running = True
while running:
    mouse_pos = pygame.mouse.get_pos()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # --- mouse down ---
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Left click clears all arrows
            if event.button == 1:
                arrows.clear()

            # Right-click starts arrow drawing
            if event.button == 3 and game_state == PLAYING:
                arrow_drawing = True
                arrow_start_pixel = event.pos
                arrow_current_pixel = event.pos

            if game_state == START_SCREEN:
                for text, rect in buttons:
                    if rect.collidepoint(mouse_pos):
                        if text == "Create Game":
                            game_state = BOT_SELECTION
                        break

            elif game_state == BOT_SELECTION:
                for type, rect, *args in dropdown_buttons:
                    if rect.collidepoint(mouse_pos):
                        if type == "black":
                            is_black_dropdown_open = not is_black_dropdown_open
                            is_white_dropdown_open = False
                        elif type == "white":
                            is_white_dropdown_open = not is_white_dropdown_open
                            is_black_dropdown_open = False
                        elif type == "black_option" and is_black_dropdown_open:
                            selected_black_bot_index = args[0]
                            is_black_dropdown_open = False
                        elif type == "white_option" and is_white_dropdown_open:
                            selected_white_bot_index = args[0]
                            is_white_dropdown_open = False
                        elif type == "confirm":
                            game_state = PLAYING
                            with board_lock:
                                board.reset()
                            white_selection = bot_list[selected_white_bot_index]
                            black_selection = bot_list[selected_black_bot_index]
                            is_white_human = (white_selection[0] == "Human")
                            is_black_human = (black_selection[0] == "Human")
                            white_bot_function = white_selection[1]
                            black_bot_function = black_selection[1]
                            if is_black_human and not is_white_human:
                                is_board_flipped = True
                            else:
                                is_board_flipped = False
                            is_white_dropdown_open = False
                            is_black_dropdown_open = False
                        else:
                            is_white_dropdown_open = False
                            is_black_dropdown_open = False

            elif game_state == GAME_OVER:
                rematch_rect, new_game_rect, quit_rect = draw_game_over_screen(mouse_pos)
                if rematch_rect.collidepoint(mouse_pos):
                    game_state = PLAYING
                    with board_lock:
                        board.reset()
                    game_over_message = ""
                    selected_square = None
                    valid_moves = []
                    pocket_selected = None
                    promoting_move = None
                elif new_game_rect.collidepoint(mouse_pos):
                    game_state = BOT_SELECTION
                    with board_lock:
                        board.reset()
                    game_over_message = ""
                    selected_square = None
                    valid_moves = []
                    pocket_selected = None
                    promoting_move = None
                elif quit_rect.collidepoint(mouse_pos):
                    running = False

            elif game_state == PLAYING:
                mouse_x, mouse_y = event.pos
                c = mouse_x // SQUARE_SIZE
                r = mouse_y // SQUARE_SIZE
                board_r = 7 - r if not is_board_flipped else r
                clicked_square = chess.square(c, board_r)

                # Handle promotion choice if a promotion is pending
                if promoting_move:
                    promoted = False
                    for rect, piece_type in promotion_rects:
                        if rect.collidepoint(mouse_x, mouse_y):
                            final_move = chess.Move(promoting_move.from_square, promoting_move.to_square, promotion=piece_type)
                            with board_lock:
                                board.push(final_move)
                            promoting_move = None
                            selected_square = None
                            valid_moves = []
                            promoted = True
                            draw_game_screen()
                            break
                    if not promoted:
                        promoting_move = None
                        selected_square = None
                        valid_moves = []

                elif selected_square or pocket_selected:
                    if selected_square:
                        start_square = chess.square(selected_square[1], 7 - selected_square[0])
                        piece = board.piece_at(start_square)
                        is_pawn_promotion = (piece and piece.piece_type == chess.PAWN and
                                             chess.square_rank(clicked_square) == (7 if board.turn == chess.WHITE else 0))
                        if is_pawn_promotion:
                            any_legal_promotion = False
                            for p_type in promotion_options:
                                promo_move = chess.Move(start_square, clicked_square, promotion=p_type)
                                if promo_move in board.legal_moves:
                                    any_legal_promotion = True
                                    break
                            if any_legal_promotion:
                                promoting_move = chess.Move(start_square, clicked_square)
                            else:
                                selected_square = None
                                valid_moves = []
                        else:
                            move = chess.Move(start_square, clicked_square)
                            if move in board.legal_moves:
                                with board_lock:
                                    board.push(move)
                                selected_square = None
                                valid_moves = []
                                draw_game_screen()
                            else:
                                selected_square = None
                                valid_moves = []
                    elif pocket_selected:
                        drop_move = chess.Move(clicked_square, clicked_square, drop=pocket_selected)
                        if drop_move in board.legal_moves:
                            with board_lock:
                                board.push(drop_move)
                            pocket_selected = None
                            draw_game_screen()
                        else:
                            pocket_selected = None
                else:
                    pocket_x_start = BOARD_WIDTH + 10
                    pocket_spacing = 10
                    current_turn_human = (board.turn == chess.WHITE and is_white_human) or (board.turn == chess.BLACK and is_black_human)
                    if not current_turn_human:
                        pass
                    else:
                        y_pos = 20
                        if board.turn == chess.BLACK:
                            for piece_type in reversed(chess.PIECE_TYPES):
                                count = board.pockets[chess.BLACK].count(piece_type)
                                if count > 0:
                                    rect = pygame.Rect(pocket_x_start, y_pos, POCKET_PIECE_SIZE, POCKET_PIECE_SIZE)
                                    if rect.collidepoint(mouse_x, mouse_y):
                                        pocket_selected = piece_type
                                        break
                                    y_pos += POCKET_PIECE_SIZE + pocket_spacing
                        if board.turn == chess.WHITE:
                            y_pos = HEIGHT - 20
                            for piece_type in reversed(chess.PIECE_TYPES):
                                count = board.pockets[chess.WHITE].count(piece_type)
                                if count > 0:
                                    y_pos -= (POCKET_PIECE_SIZE + pocket_spacing)
                                    rect = pygame.Rect(pocket_x_start, y_pos, POCKET_PIECE_SIZE, POCKET_PIECE_SIZE)
                                    if rect.collidepoint(mouse_x, mouse_y):
                                        pocket_selected = piece_type
                                        break
                        if not pocket_selected:
                            if c < BOARD_SIZE:
                                piece = board.piece_at(clicked_square)
                                if piece and piece.color == board.turn:
                                    selected_square = (r if not is_board_flipped else 7 - r, c)
                                    valid_moves = [move for move in board.legal_moves if move.from_square == clicked_square]

        # --- mouse up ---
        if event.type == pygame.MOUSEBUTTONUP:
            # Right click release -> finalize arrow
            if event.button == 3 and game_state == PLAYING and arrow_drawing:
                # finalize arrow in board coords
                sx, sy = arrow_start_pixel
                ex, ey = event.pos

                def pixel_to_sq(px, py):
                    if 0 <= px < BOARD_WIDTH and 0 <= py < BOARD_WIDTH:
                        file = px // SQUARE_SIZE
                        rank_r = py // SQUARE_SIZE
                        rank = 7 - rank_r if not is_board_flipped else rank_r
                        return (file, rank)
                    return None

                start_sq = pixel_to_sq(sx, sy)
                end_sq = pixel_to_sq(ex, ey)
                if start_sq and end_sq:
                    arrows.append((start_sq, end_sq))

                # reset dragging state
                arrow_drawing = False
                arrow_start_pixel = None
                arrow_current_pixel = None

        # --- mouse motion ---
        if event.type == pygame.MOUSEMOTION:
            if arrow_drawing:
                arrow_current_pixel = event.pos

    # --- Main game logic based on game state ---
    if game_state == START_SCREEN:
        buttons = draw_start_screen(mouse_pos)

    elif game_state == BOT_SELECTION:
        draw_bot_selection_screen(mouse_pos)

    elif game_state == PLAYING:
        # spawn bot thread if needed
        spawn_bot_if_needed()
        # try apply bot result if ready
        applied = try_apply_bot_result()
        if applied:
            draw_game_screen()  # show the board immediately after bot move

        # check game end
        with board_lock:
            is_checkmate = board.is_checkmate()
            is_stalemate = board.is_stalemate()
            current_turn = board.turn

        if is_checkmate:
            game_over_message = f"Checkmate! {'White' if current_turn == chess.BLACK else 'Black'} wins."
            game_state = GAME_OVER
        elif is_stalemate:
            game_over_message = "Stalemate!"
            game_state = GAME_OVER

        draw_game_screen()

    elif game_state == GAME_OVER:
        draw_game_over_screen(mouse_pos)

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
