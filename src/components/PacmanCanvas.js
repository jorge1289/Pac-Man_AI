class PacmanCanvasDisplay {
  constructor(canvasContext, gridSize = 30.0) {
    this.ctx = canvasContext;
    this.gridSize = gridSize;
    this.turn = 0;
    this.agentCounter = 0;
  }

  initialize(state, isBlue = false) {
    // Match the initialization from original PacmanGraphics
    this.draw(state);
    this.turn = 0;
    this.agentCounter = 0;
  }

  update(state) {
    const numAgents = state.agentStates.length;
    this.agentCounter = (this.agentCounter + 1) % numAgents;
    
    if (this.agentCounter === 0) {
      this.turn += 1;
      this.draw(state);
    }
    
    if (state._win || state._lose) {
      this.draw(state);
    }
  }

  draw(state) {
    // Clear canvas
    this.ctx.fillStyle = '#000000'; // BACKGROUND_COLOR from original
    this.ctx.fillRect(0, 0, this.ctx.canvas.width, this.ctx.canvas.height);
    
    // Draw walls
    this.drawWalls(state.layout.walls);
    
    // Draw food
    this.drawFood(state.food);
    
    // Draw agents (Pacman and ghosts)
    state.agentStates.forEach((agent, index) => {
      if (agent.isPacman) {
        this.drawPacman(agent);
      } else {
        this.drawGhost(agent, index);
      }
    });

    // Draw score
    this.drawScore(state.score);
  }
} 