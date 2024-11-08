/**
 * Main Pacman game component that handles:
 * - Game initialization and state management
 * - Integration with Pyodide for Python game logic
 * - Canvas setup and rendering
 * - React component lifecycle
 */

import React, { useRef, useEffect, useState } from 'react';

const PacmanGame = () => {
  const canvasRef = useRef(null);
  const [gameState, setGameState] = useState(null);
  
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    const initGame = async () => {
      const pyodide = await loadPyodide();
      await pyodide.loadPackage(['numpy']);
      
      // Load core game logic
      const gameLogic = await fetch('/pacman/game.py');
      const gameCode = await gameLogic.text();
      
      // Initialize game state
      pyodide.runPython(gameCode);
      const initialState = pyodide.runPython(`
        game = ClassicGameRules().newGame(layout, agents[0], agents[1:], display)
        game.state
      `);
      
      setGameState(initialState);
      renderGame(ctx, initialState);
    };
    
    initGame();
  }, []);
  
  return (
    <div className="pacman-game">
      <canvas 
        ref={canvasRef}
        width={DEFAULT_GRID_SIZE * 20}
        height={DEFAULT_GRID_SIZE * 20 + INFO_PANE_HEIGHT}
      />
    </div>
  );
}; 