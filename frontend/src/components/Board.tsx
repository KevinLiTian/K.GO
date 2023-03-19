import { useEffect, useState } from 'react';

import {
  BLACK,
  WHITE,
  EMPTY,
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
  const [currentPlayer, setCurrentPlayer] = useState(BLACK);
  const [previousBoards, setPreviousBoards] = useState<number[]>([]);
  const [socket, setSocket] = useState<any>(null);

  const clientPlayer = settings
    ? settings.turn == '0'
      ? BLACK
      : WHITE
    : BLACK;
  const AIPlayer = settings ? (settings.turn == '0' ? WHITE : BLACK) : WHITE;

  useEffect(() => {
    if (settings && settings.mode == '1') {
      // create WebSocket connection
      const ws = new WebSocket(`ws://${settings?.api}`);

      // set up event listeners for WebSocket events
      ws.onopen = () => {
        ws.send(AIPlayer == BLACK ? 'b' : 'w');
      };

      ws.onmessage = (event) => {
        setBoard(JSON.parse(event.data));
        setCurrentPlayer((prev) => (prev == BLACK ? WHITE : BLACK));
      };

      ws.onclose = () => console.log('WebSocket connection closed.');

      // save WebSocket instance in state
      setSocket(ws);

      // clean up function to close WebSocket connection on component unmount
      return () => {
        ws.close();
      };
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
    if (board[row][col] !== EMPTY) return;

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
    setCurrentPlayer((prev) => (prev == BLACK ? WHITE : BLACK));

    if (settings && settings.mode == '1') {
      socket.send(JSON.stringify([row, col]));
    }
  }

  return (
    <table className="border-[5px] border-[#533939]">
      <tbody>
        {board.map((row, rowIndex) => (
          <tr key={rowIndex}>
            {row.map((__, colIndex) => (
              <td className="h-[36px] w-[36px] p-0" key={colIndex}>
                <div
                  onMouseOver={() => handleMouseOver(rowIndex, colIndex)}
                  onMouseOut={handleMouseOut}
                  onClick={() => handleMouseClick(rowIndex, colIndex)}
                >
                  <svg width="36" height="36">
                    <g>
                      <rect x="0" y="0" width="36" height="36" fill="#cd9d6f" />
                    </g>
                    <BoardGrid row={rowIndex} col={colIndex} />
                    {board[rowIndex][colIndex] === EMPTY &&
                      rowIndex === hoveredCell?.row &&
                      colIndex === hoveredCell.col && (
                        <StoneHover color={playerColor(currentPlayer)} />
                      )}
                    {board[rowIndex][colIndex] !== EMPTY && (
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
