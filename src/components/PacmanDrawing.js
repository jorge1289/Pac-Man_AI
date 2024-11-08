/**
 * Utility functions for drawing Pacman on the canvas
 * Handles the visual representation of Pacman including:
 * - Position calculation
 * - Direction-based rotation
 * - Mouth animation
 * - Color and size properties
 */

export const drawPacman = (ctx, agent, gridSize) => {
  const { x, y } = agent.getPosition();
  const direction = agent.getDirection();
  
  // Calculate angle based on direction
  const angles = {
    'West': Math.PI,
    'East': 0,
    'North': -Math.PI/2,
    'South': Math.PI/2
  };
  
  ctx.beginPath();
  ctx.arc(
    x * gridSize,
    y * gridSize,
    gridSize/2,
    angles[direction] + Math.PI/4,
    angles[direction] - Math.PI/4,
    true
  );
  ctx.fillStyle = '#FFFF00';
  ctx.fill();
}; 