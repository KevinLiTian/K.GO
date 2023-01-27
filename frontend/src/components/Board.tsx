import { useState } from 'react';

import { createBoard, playerColor } from '../utils/utils';
import BoardGrid from './BoardGrid';
import Stone from './Stone';
import StoneShadow from './StoneShadow';

const Board = () => {
  const [board, setBoard] = useState(createBoard());
  const [hoveredCell, setHoveredCell] = useState<{
    row: number;
    col: number;
  } | null>(null);
  const [player, setPlayer] = useState(1);

  function handleMouseOver(row: number, col: number) {
    setHoveredCell({ row, col });
  }

  function handleMouseOut() {
    setHoveredCell(null);
  }

  function handleMouseClick(row: number, col: number) {
    const newBoard = JSON.parse(JSON.stringify(board));
    newBoard[row][col] = player;
    setBoard(newBoard);
    setPlayer(player === 1 ? -1 : 1);
  }

  return (
    <table className="border-[10px] border-[#533939] bg-[#e8b060]">
      <tbody>
        {board.map((row, rowIndex) => (
          <tr key={rowIndex}>
            {row.map((__, colIndex) => (
              <td className="h-[30px] w-[30px] p-0" key={colIndex}>
                <div
                  onMouseOver={() => handleMouseOver(rowIndex, colIndex)}
                  onMouseOut={handleMouseOut}
                  onClick={() => handleMouseClick(rowIndex, colIndex)}
                >
                  <svg width="30" height="30px">
                    <g>
                      <rect x="0" y="0" width="30" height="30" fill="#e8b060" />
                    </g>
                    <BoardGrid row={rowIndex} col={colIndex} />
                    {board[rowIndex][colIndex] === 0 &&
                      rowIndex === hoveredCell?.row &&
                      colIndex === hoveredCell.col && (
                        <Stone color={playerColor(player)} opacity={0.5} />
                      )}
                    {board[rowIndex][colIndex] !== 0 && (
                      <>
                        <StoneShadow />
                        <Stone
                          color={playerColor(board[rowIndex][colIndex])}
                          opacity={1}
                        />
                      </>
                    )}
                  </svg>
                </div>
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
};

export default Board;
