// Firebase Configuration
const firebaseConfig = {
    apiKey: "AIzaSyCQUsgsn5YJAfwHuBXp6rLbWZLervv4Gis",
    authDomain: "mirs-vote.firebaseapp.com",
    databaseURL: "https://mirs-vote-default-rtdb.firebaseio.com",
    projectId: "mirs-vote",
    storageBucket: "mirs-vote.firebasestorage.app",
    messagingSenderId: "942268147154",
    appId: "1:942268147154:web:fb3087f430ba58f199940b"
};

// Initialize Firebase (Compat Mode)
firebase.initializeApp(firebaseConfig);
const db = firebase.database();

let currentSessionId = "demo_session"; // Default session
let userId = "Guest_" + Math.floor(Math.random() * 1000);
let selectedCell = null;

// --- GUEST VIEW ---
// --- GUEST VIEW ---
function initGuest() {
    const board = document.getElementById("guest-board");
    if (!board) return;

    // Create 13x13 Grid
    for (let y = 0; y < 13; y++) {
        for (let x = 0; x < 13; x++) {
            const cell = document.createElement("div");
            cell.className = "guest-cell";
            cell.dataset.x = x;
            cell.dataset.y = y;

            cell.addEventListener("click", () => {
                // Select Cell
                if (selectedCell) selectedCell.classList.remove("selected");
                selectedCell = cell;
                cell.classList.add("selected");

                // Enable Vote Button
                document.getElementById("vote-btn").disabled = false;
            });
            board.appendChild(cell);
        }
    }

    // Vote Button
    const btn = document.getElementById("vote-btn");
    // Initially Disabled
    btn.disabled = true;

    btn.addEventListener("click", () => {
        if (!selectedCell) return; // Should be disabled, but safety check

        const x = parseInt(selectedCell.dataset.x);
        const y = parseInt(selectedCell.dataset.y);

        // Send Vote
        db.ref(`sessions/${currentSessionId}/votes`).push({
            user: userId,
            x: x,
            y: y,
            timestamp: Date.now()
        });

        // Feedback
        const msg = document.getElementById("status-msg");
        msg.innerText = `投票しました！ (${x}, ${y})`;
        msg.style.color = "#4caf50";

        // Reset selection and Disable Button
        selectedCell.classList.remove("selected");
        selectedCell = null;
        btn.disabled = true;
    });

    document.getElementById("user-id-display").innerText = "ID: " + userId;
}

// --- HOST VIEW ---
function initHost() {
    const board = document.getElementById("board-area");
    if (!board) return;

    // Create 13x13 Grid
    for (let y = 0; y < 13; y++) {
        for (let x = 0; x < 13; x++) {
            const cell = document.createElement("div");
            cell.className = "cell";
            cell.id = `cell-${x}-${y}`;
            board.appendChild(cell);
        }
    }

    // Listen to Board State
    db.ref(`sessions/${currentSessionId}/board`).on("value", (snapshot) => {
        const grid = snapshot.val(); // 1D array or 2D? Assuming 2D or handled by index
        // Clear previous stones
        document.querySelectorAll(".stone").forEach(el => el.remove());

        if (grid) {
            // Grid is flat list or 2D? Let's assume Backend sends 2D list or we flattened it.
            // Backend sends numpy array as list of lists usually.
            for (let y = 0; y < 13; y++) {
                for (let x = 0; x < 13; x++) {
                    const val = grid[y][x];
                    if (val === 1 || val === 2) {
                        const cell = document.getElementById(`cell-${x}-${y}`);
                        const stone = document.createElement("div");
                        stone.className = `stone ${val === 1 ? 'black' : 'white'}`;
                        cell.appendChild(stone);
                    }
                }
            }
        }
    });

    // Listen to Votes (Real-time Animations)
    const votesRef = db.ref(`sessions/${currentSessionId}/votes`);

    // Store local counts for relative scaling
    const voteCounts = {};
    let maxVoteCount = 1;

    votesRef.on("child_added", (snapshot) => {
        const data = snapshot.val();
        if (!data) return;

        // Update counts
        const key = `${data.x},${data.y}`;
        voteCounts[key] = (voteCounts[key] || 0) + 1;
        if (voteCounts[key] > maxVoteCount) {
            maxVoteCount = voteCounts[key];
        }

        // 1. Update/Create Circle
        const cell = document.getElementById(`cell-${data.x}-${data.y}`);
        if (cell) {
            let circle = cell.querySelector(".vote-circle");
            if (!circle) {
                circle = document.createElement("div");
                circle.className = "vote-circle";
                // Start small
                circle.style.width = "20%";
                circle.style.height = "20%";
                cell.appendChild(circle);

                // Trigger reflow
                void circle.offsetWidth;
            }

            // --- SCALING & COLOR LOGIC ---
            // Cap growth at some count (e.g., 20)
            const count = voteCounts[key];
            const CAP = 20;

            // Size: 30% base + up to 60% more (max 90%)
            // Grows linearly until CAP
            let size = 30 + (Math.min(count, CAP) / CAP * 60);

            // Color Intensity:
            // Starts Light Red (rgba(255, 100, 100, 0.5))
            // Turns Dark Red/Blackish (rgba(100, 0, 0, 0.9))
            // Continues changing even after size CAP

            // Scale factor for color (0 to 1) based on count 
            // e.g. up to 50 votes for full darkness
            const colorScale = Math.min(count, 50) / 50;

            const r = 255 - (colorScale * 155); // 255 -> 100
            const g = 100 - (colorScale * 100); // 100 -> 0
            const b = 100 - (colorScale * 100); // 100 -> 0
            const a = 0.5 + (colorScale * 0.4); // 0.5 -> 0.9

            circle.style.width = size + "%";
            circle.style.height = size + "%";
            circle.style.backgroundColor = `rgba(${r}, ${g}, ${b}, ${a})`;

            // Pop Animation
            circle.classList.remove("vote-pop");
            void circle.offsetWidth;
            circle.classList.add("vote-pop");
        }

        // 2. Kill Log Animation
        addKillLog(data.user, data.x, data.y);
    });

    // Clean up circles on reset (detected via board clear? or separate listener)
    // Actually board listener clears stones. Votes cleared when DB clears?
    // Listen for removal of votes
    votesRef.on("child_removed", () => {
        document.querySelectorAll(".vote-circle").forEach(el => el.remove());
        // reset counts
        for (let k in voteCounts) delete voteCounts[k];
        maxVoteCount = 1;
    });
}

function addKillLog(user, x, y) {
    const container = document.getElementById("kill-log-container");
    const div = document.createElement("div");
    div.className = "log-entry";
    div.innerHTML = `<span class="user">${user}</span> ▶ <span class="coord">(${x}, ${y})</span>`;

    // Add to top or bottom? APEX style usually top?
    // CSS uses column-reverse so appending adds to top visibly if we want stack up.
    container.appendChild(div);

    // Auto remove logic handled by CSS animation fadeOut (4s)
    setTimeout(() => {
        div.remove();
    }, 4500);
}

// Admin Functions
async function closeVoting() {
    // 1. Get Votes
    const snapshot = await db.ref(`sessions/${currentSessionId}/votes`).once("value");
    const votes = snapshot.val();
    if (!votes) {
        alert("投票がありません");
        return;
    }

    // 2. Tally
    const tally = {};
    let max = -1;
    let bestMove = null;

    Object.values(votes).forEach(v => {
        const key = `${v.x},${v.y}`;
        tally[key] = (tally[key] || 0) + 1;
        if (tally[key] > max) {
            max = tally[key];
            bestMove = { x: v.x, y: v.y };
        }
    });

    // 3. Update Board with Player Move (White=2 for player?) 
    // Wait, AI is usually Black(1)? Let's assume AI=1, Guest=2.
    // Or Gomoku rule: Black starts. AI is usually First?
    // If Guest are deciding the NEXT move.
    // Let's assume current state is stored and we add to it.

    // Get current board first
    const boardSnap = await db.ref(`sessions/${currentSessionId}/board`).once("value");
    let grid = boardSnap.val();
    if (!grid) {
        // Initialize 13x13 zeros
        grid = Array(13).fill().map(() => Array(13).fill(0));
    }

    // Apply Guest Move (Player = 2)
    grid[bestMove.y][bestMove.x] = 2; // White
    await db.ref(`sessions/${currentSessionId}/board`).set(grid);

    // 4. Call AI
    try {
        const response = await fetch("http://localhost:8000/api/think", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ board: grid })
        });
        const aiMove = await response.json();

        // Apply AI Move (Black = 1 or AI ID)
        // Adjust ID based on backend logic (Backend sends x,y. We decide ID)
        // usually AI is 1.
        grid[aiMove.y][aiMove.x] = 1;

        await db.ref(`sessions/${currentSessionId}/board`).set(grid);

        // Clear votes for next round
        await db.ref(`sessions/${currentSessionId}/votes`).remove();

    } catch (e) {
        console.error(e);
        alert("AI Error: " + e.message);
    }
}

function resetGame() {
    if (!confirm("リセットしますか？")) return;
    const emptyGrid = Array(13).fill().map(() => Array(13).fill(0));
    db.ref(`sessions/${currentSessionId}/board`).set(emptyGrid);
    db.ref(`sessions/${currentSessionId}/votes`).remove();
}

// Router
if (window.location.pathname.includes("host.html")) {
    initHost();
} else {
    initGuest();
}
