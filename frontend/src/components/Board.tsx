import { useEffect, useState } from 'react';
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
import { boardProps } from '../utils/Interfaces';

const Board = ({ initial, settings }: boardProps) => {
  const [board, setBoard] = useState(initial ? initial : () => createBoard());
  const [hoveredCell, setHoveredCell] = useState<{
    row: number;
    col: number;
  } | null>(null);
  const [clientPlayer, setClientPlayer] = useState(
    settings ? (settings.turn == '0' ? 1 : 2) : 1
  );
  const [AIPlayer, setAIPlayer] = useState(
    settings ? (settings.turn == '0' ? 2 : 1) : 2
  );
  const [currentPlayer, setCurrentPlayer] = useState(1);
  const [previousBoards, setPreviousBoards] = useState<number[]>([]);

  useEffect(() => {
    if (settings && settings.mode == '1' && settings.turn == '1') {
      API.post(settings.api!, { id: settings.id }).then((res) => {
        const move = res.data.move;
        const cpBoard = copyBoard(board);
        cpBoard[move[0]][move[1]] = 1;
        setBoard(cpBoard);
        setCurrentPlayer((prev) => (prev == 1 ? 2 : 1));
      });
    }
  }, []);

  function handleMouseOver(row: number, col: number) {
    setHoveredCell({ row, col });
  }

  function handleMouseOut() {
    setHoveredCell(null);
  }

  async function handleMouseClick(row: number, col: number) {
    if (settings && settings.mode == '1' && currentPlayer != clientPlayer)
      return;

    // Cell occupied
    if (board[row][col] !== 0) return;

    // Deep copy new board for processing
    const newBoard = copyBoard(board);
    newBoard[row][col] = currentPlayer;

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
    setCurrentPlayer((prev) => (prev == 1 ? 2 : 1));

    if (settings && settings.mode == '1') {
      // AI make move
      API.post(settings?.api!, { id: settings?.id, move: [row, col] }).then(
        (res) => {
          const move = res.data.move;
          const cpBoard = copyBoard(newBoard);
          cpBoard[move[0]][move[1]] = AIPlayer;
          setBoard(cpBoard);
          setCurrentPlayer((prev) => (prev == 1 ? 2 : 1));
        }
      );
    }
  }

  return (
    <table className="border-[5px] border-[#533939]">
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
                        <StoneHover color={playerColor(currentPlayer)} />
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
