import { useEffect, useRef, useState } from 'react';
import API from '../api';

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
import StoneHover from './BoardComponents/StoneHover';

const Board = () => {
  const [board, setBoard] = useState(() => createBoard());
  const [hoveredCell, setHoveredCell] = useState<{
    row: number;
    col: number;
  } | null>(null);
  const [player, setPlayer] = useState(1);
  const [previousBoards, setPreviousBoards] = useState<number[]>([]);

  // Game ID
  const id = useRef(null);

  useEffect(() => {
    API.post('/setup').then((res) => (id.current = res.data.id));
  }, []);

  function handleMouseOver(row: number, col: number) {
    setHoveredCell({ row, col });
  }

  function handleMouseOut() {
    setHoveredCell(null);
  }

  async function handleMouseClick(row: number, col: number) {
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

    // AI make move
    API.post('/greedypolicy', { id: id.current, move: [row, col] }).then(
      (res) => {
        const move = res.data.move;
        const cpBoard = copyBoard(newBoard);
        cpBoard[move[0]][move[1]] = 2;
        setBoard(cpBoard);
      }
    );
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
                    <g>
                      <rect x="0" y="0" width="30" height="30" fill="#cd9d6f" />
                    </g>
                    <BoardGrid row={rowIndex} col={colIndex} />
                    {board[rowIndex][colIndex] === 0 &&
                      rowIndex === hoveredCell?.row &&
                      colIndex === hoveredCell.col && (
                        <StoneHover color={playerColor(player)} />
                      )}
                    {board[rowIndex][colIndex] !== 0 && (
                      <>
                        <StoneShadow />
                        <Stone color={playerColor(board[rowIndex][colIndex])} />
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
