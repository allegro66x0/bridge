import sys
import os
import random
import time
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from func_timeout import func_timeout, FunctionTimedOut
import numpy as np

# --- Import existing AI ---
# Add L6 directory to sys.path to import webcam_gomoku_ai
current_dir = os.path.dirname(os.path.abspath(__file__))
l6_dir = os.path.join(current_dir, '../L6')
sys.path.append(l6_dir)

try:
    import webcam_gomoku_ai as ai
    print("✅ AI Module Loaded Successfully")
except ImportError as e:
    print(f"❌ Failed to import AI module: {e}")
    sys.exit(1)

app = FastAPI()

# --- Data Models ---
class BoardRequest(BaseModel):
    board: List[List[int]] # 2D array: 0=Empty, 1=AI, 2=Player
    depth: Optional[int] = 5

class MoveResponse(BaseModel):
    x: int
    y: int
    score: int
    processing_time: float
    method: str # "ai", "random_fallback"

# --- Constants ---
BOARD_SIZE = 13
TIMEOUT_SECONDS = 3.0

# --- Helper Functions ---
def get_valid_moves(board_np):
    moves = []
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            if board_np[y][x] == 0:
                moves.append((x, y))
    return moves

def run_ai_logic(board_np, depth):
    # This function wraps the AI call to be used with func_timeout
    # webcam_gomoku_ai.find_best_move_parallel returns (score, (x, y))
    # Note: AI code uses x=col, y=row convention in return
    score, move = ai.find_best_move_parallel(board_np, depth)
    return score, move

# --- API Endpoints ---

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Gomoku AI Backend Ready"}

@app.post("/api/think", response_model=MoveResponse)
def think(req: BoardRequest):
    start_time = time.time()
    
    # Validation
    if len(req.board) != BOARD_SIZE or len(req.board[0]) != BOARD_SIZE:
        raise HTTPException(status_code=400, detail=f"Board must be {BOARD_SIZE}x{BOARD_SIZE}")

    board_np = np.array(req.board, dtype=np.int32)
    
    # Fallback preparation
    valid_moves = get_valid_moves(board_np)
    if not valid_moves:
         raise HTTPException(status_code=400, detail="Board is full")

    # Select random fallback move immediately (just in case)
    fallback_move = random.choice(valid_moves)
    
    try:
        # Execute AI with timeout
        # Using func_timeout to enforce hard limit
        score, move = func_timeout(
            TIMEOUT_SECONDS, 
            run_ai_logic, 
            args=(board_np, req.depth)
        )
        
        # AI Success
        return MoveResponse(
            x=move[0],
            y=move[1],
            score=score,
            processing_time=time.time() - start_time,
            method="ai"
        )
        
    except FunctionTimedOut:
        print(f"⚠️ AI Timeout ({TIMEOUT_SECONDS}s). Using fallback.")
        return MoveResponse(
            x=fallback_move[0],
            y=fallback_move[1],
            score=0,
            processing_time=time.time() - start_time,
            method="random_fallback_timeout"
        )
        
    except Exception as e:
        print(f"⚠️ AI Error: {e}. Using fallback.")
        return MoveResponse(
            x=fallback_move[0],
            y=fallback_move[1],
            score=0,
            processing_time=time.time() - start_time,
            method="random_fallback_error"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
