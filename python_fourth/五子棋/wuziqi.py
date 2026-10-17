# gomoku.py
import sys
import math
import time
from collections import defaultdict

SIZE = 15  # 棋盘大小
WIN_LEN = 5

EMPTY = 0
HUMAN = 1  # X
AI = -1    # O

# 搜索深度（越大AI越强但越慢）
MAX_DEPTH = 3

# 在已有棋子周围多远范围内考虑候选落子（减少分支）
SEARCH_RANGE = 2

DIRECTIONS = [(1,0),(0,1),(1,1),(1,-1)]

def init_board():
    return [[EMPTY for _ in range(SIZE)] for _ in range(SIZE)]

def print_board(board):
    header = "   " + " ".join(f"{i:2}" for i in range(SIZE))
    print(header)
    for r in range(SIZE):
        row = board[r]
        line = f"{r:2} " + " ".join('.' if c==EMPTY else ('X' if c==HUMAN else 'O') for c in row)
        print(line)

def in_board(r,c):
    return 0<=r<SIZE and 0<=c<SIZE

def is_win(board, r, c, player):
    # 判断 player 在 r,c 落子后是否获胜
    for dr,dc in DIRECTIONS:
        cnt = 1
        # 正方向
        i=1
        while True:
            nr, nc = r + dr*i, c + dc*i
            if not in_board(nr,nc) or board[nr][nc] != player:
                break
            cnt += 1
            i += 1
        # 反方向
        i=1
        while True:
            nr, nc = r - dr*i, c - dc*i
            if not in_board(nr,nc) or board[nr][nc] != player:
                break
            cnt += 1
            i += 1
        if cnt >= WIN_LEN:
            return True
    return False

def get_neighbors(board):
    # 返回所有非空格子的邻域（用于生成候选落点）
    neigh = set()
    for r in range(SIZE):
        for c in range(SIZE):
            if board[r][c] != EMPTY:
                for dr in range(-SEARCH_RANGE, SEARCH_RANGE+1):
                    for dc in range(-SEARCH_RANGE, SEARCH_RANGE+1):
                        nr, nc = r+dr, c+dc
                        if in_board(nr,nc) and board[nr][nc]==EMPTY:
                            neigh.add((nr,nc))
    return neigh

# 评估函数：对每一条线计算模式分
PATTERN_SCORES = {
    # 常用启发式评分（对AI视角，player为AI时为正分）
    (5, 0): 100000,   # 五连
    (4, 0): 10000,    # 连四，无敌
    (4, 1): 5000,     # 活四（一个空口）
    (3, 0): 1000,     # 三连
    (3, 1): 500,
    (2, 0): 100,
    (2, 1): 50,
    (1, 0): 10
}

def line_score(count, open_ends):
    # 简单映射
    key = (count, open_ends)
    if count >= 5:
        return PATTERN_SCORES[(5,0)]
    if key in PATTERN_SCORES:
        return PATTERN_SCORES[key]
    # fallback
    return count * (10 if open_ends>0 else 1)

def evaluate_board(board):
    # 对整盘进行启发式评分：AI 为正，人类为负
    score = 0
    for r in range(SIZE):
        for c in range(SIZE):
            if board[r][c] == EMPTY:
                continue
            player = board[r][c]
            for dr,dc in DIRECTIONS:
                # 只从线的一端开始统计避免重复太多（简单做法：只在前一格不是该player时统计）
                pr, pc = r - dr, c - dc
                if in_board(pr, pc) and board[pr][pc] == player:
                    continue
                # 从(r,c)向方向(dr,dc)扫描连续同色
                cnt = 0
                i = 0
                while True:
                    nr, nc = r + dr*i, c + dc*i
                    if not in_board(nr,nc) or board[nr][nc] != player:
                        break
                    cnt += 1
                    i += 1
                # 计算开放端数量
                open_ends = 0
                # 前端
                nr, nc = r - dr, c - dc
                if in_board(nr,nc) and board[nr][nc] == EMPTY:
                    open_ends += 1
                # 后端
                nr, nc = r + dr*cnt, c + dc*cnt
                if in_board(nr,nc) and board[nr][nc] == EMPTY:
                    open_ends += 1
                s = line_score(cnt, open_ends)
                if player == AI:
                    score += s
                else:
                    score -= s
    return score

def generate_moves(board):
    # 生成候选落点：若无子则放在中间
    neigh = get_neighbors(board)
    if not neigh:
        center = SIZE//2
        return [(center, center)]
    # 可排序：优先临近高分位置（简单用评估对单点假设落子）
    moves = list(neigh)
    # 小优化：按单步启发式分排序，取前 N（也可以不取）
    scored = []
    for (r,c) in moves:
        board[r][c] = AI
        s_ai = evaluate_board(board)
        board[r][c] = HUMAN
        s_hu = evaluate_board(board)
        board[r][c] = EMPTY
        # 我们希望 AI 落子时得分高，且能压制人类，所以用 s_ai - s_hu 的差作为评分
        scored.append(((r,c), s_ai - s_hu))
    scored.sort(key=lambda x: x[1], reverse=True)
    # 截断候选数，减少分支；你可调大或调小
    CANDIDATE_LIMIT = 40
    chosen = [mv for mv, sc in scored[:CANDIDATE_LIMIT]]
    return chosen

def minimax(board, depth, alpha, beta, maximizing, last_move):
    # last_move 用于快速胜负判断（r,c）或 None
    if last_move is not None:
        r,c = last_move
        if is_win(board, r, c, AI):
            return (None, 1_000_000 if maximizing==False else -1_000_000)
        if is_win(board, r, c, HUMAN):
            return (None, -1_000_000 if maximizing==False else 1_000_000)
    if depth == 0:
        return (None, evaluate_board(board))

    moves = generate_moves(board)
    if not moves:
        return (None, evaluate_board(board))

    best_move = None
    if maximizing:
        max_eval = -math.inf
        for mv in moves:
            r,c = mv
            board[r][c] = AI
            _, eval_score = minimax(board, depth-1, alpha, beta, False, (r,c))
            board[r][c] = EMPTY
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = mv
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return (best_move, max_eval)
    else:
        min_eval = math.inf
        for mv in moves:
            r,c = mv
            board[r][c] = HUMAN
            _, eval_score = minimax(board, depth-1, alpha, beta, True, (r,c))
            board[r][c] = EMPTY
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = mv
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return (best_move, min_eval)

def ai_move(board):
    start = time.time()
    mv, score = minimax(board, MAX_DEPTH, -math.inf, math.inf, True, None)
    end = time.time()
    # 如果没有得到任何候选（不太可能），从所有空格中随机或第一个
    if mv is None:
        for r in range(SIZE):
            for c in range(SIZE):
                if board[r][c] == EMPTY:
                    return r,c
    print(f"AI 计算时间: {end-start:.2f}s, 估值: {score}")
    return mv

def human_move(board):
    while True:
        s = input("请输入你的落子（格式：row col，例如 7 7），或输入 q 退出: ").strip()
        if s.lower() in ('q','quit','exit'):
            print("退出游戏")
            sys.exit(0)
        parts = s.split()
        if len(parts) != 2:
            print("输入格式错误，请输入两个数字，用空格分开。")
            continue
        try:
            r = int(parts[0])
            c = int(parts[1])
        except:
            print("请输入整数坐标。")
            continue
        if not in_board(r,c):
            print(f"坐标越界，请输入 0..{SIZE-1}。")
            continue
        if board[r][c] != EMPTY:
            print("该位置已有棋子，请重新选择。")
            continue
        return r,c

def main():
    print("五子棋（Gomoku） - 人机对弈")
    print("你为 X（黑），AI 为 O（白）。先手由你开始。")
    board = init_board()
    print_board(board)
    turn = HUMAN  # HUMAN 先
    last_move = None
    while True:
        if turn == HUMAN:
            r,c = human_move(board)
            board[r][c] = HUMAN
            last_move = (r,c)
            print_board(board)
            if is_win(board, r, c, HUMAN):
                print("恭喜，你赢了！")
                break
            turn = AI
        else:
            print("AI 正在思考...")
            r,c = ai_move(board)
            board[r][c] = AI
            last_move = (r,c)
            print(f"AI 落子: {r} {c}")
            print_board(board)
            if is_win(board, r, c, AI):
                print("AI 胜利。")
                break
            turn = HUMAN
        # 平局判断（无空位）
        if all(board[r][c] != EMPTY for r in range(SIZE) for c in range(SIZE)):
            print("棋盘已满，平局。")
            break

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n游戏中断，拜~")
