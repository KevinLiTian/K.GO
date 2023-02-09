import { useState } from 'react';

import {
  copyBoard,
  createBoard,
  getBoardHash,
  getGroups,
  playerColor,
  removeDeadGroups,
} from '../utils/utils';

import BoardGrid from './BoardComponents/BoardGrid';
import Stone from './BoardComponents/Stone';
import StoneShadow from './BoardComponents/StoneShadow';
import { woodTexture1 } from '../utils/styles';

const Board = () => {
  const [board, setBoard] = useState(createBoard());
  const [hoveredCell, setHoveredCell] = useState<{
    row: number;
    col: number;
  } | null>(null);
  const [player, setPlayer] = useState(1);
  const [previousBoards, setPreviousBoards] = useState<number[]>([]);

  function handleMouseOver(row: number, col: number) {
    setHoveredCell({ row, col });
  }

  function handleMouseOut() {
    setHoveredCell(null);
  }

  function handleMouseClick(row: number, col: number) {
    // Cell occupied
    if (board[row][col] !== 0) return;

    // Deep copy new board for processing
    const newBoard = copyBoard(board);
    newBoard[row][col] = player;

    // KO Rule (打劫)
    const hash = getBoardHash(newBoard);
    if (previousBoards.includes(hash)) {
      return;
    } else {
      const newPreviousBoards = [...previousBoards, hash];
      if (newPreviousBoards.length > 3) newPreviousBoards.shift();
      setPreviousBoards(newPreviousBoards);
    }

    // Get current stone group and all dead stone groups
    const [groups, curStoneGroup, deadGroups] = getGroups(newBoard, row, col);

    // Cannot make stone's own group die if not making other groups die
    if (deadGroups.length === 1 && deadGroups.includes(curStoneGroup)) {
      return;
    }

    // Remove dead groups
    removeDeadGroups(newBoard, groups, curStoneGroup, deadGroups);

    // Update board and switch player
    setBoard(newBoard);
    setPlayer(player === 1 ? 2 : 1);
  }

  return (
    <table className="border-[10px] border-[#533939] scale-[120%]">
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
                    {woodTexture1}
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
