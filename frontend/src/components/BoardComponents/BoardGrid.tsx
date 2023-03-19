import BoardCorner from './BoardCorner';
import BoardEdge from './BoardEdge';
import BoardGeneral from './BoardGeneral';
import BoardStar from './BoardStar';
import { BoardGridProps } from '../../utils/Interfaces';
import { isStar } from '../../utils/utils';

const BoardGrid = ({ col, row }: BoardGridProps) => {
  return (
    <>
      {/* 4 Corners */}
      {col === 0 && row === 0 && <BoardCorner style="rotate(0, 18, 18)" />}
      {col === 0 && row === 18 && <BoardCorner style="rotate(270, 18, 18)" />}
      {col === 18 && row === 0 && <BoardCorner style="rotate(90, 18, 18)" />}
      {col === 18 && row === 18 && <BoardCorner style="rotate(180, 18, 18)" />}

      {/* 4 Edges */}
      {col === 0 && row !== 0 && row !== 18 && (
        <BoardEdge style="rotate(270, 18, 18)" />
      )}
      {col === 18 && row !== 0 && row !== 18 && (
        <BoardEdge style="rotate(90, 18, 18)" />
      )}
      {row === 0 && col !== 0 && col !== 18 && (
        <BoardEdge style="rotate(0, 18, 18)" />
      )}
      {row === 18 && col !== 0 && col !== 18 && (
        <BoardEdge style="rotate(180, 18, 18)" />
      )}

      {/* General Grid */}
      {row !== 0 &&
        row !== 18 &&
        col !== 0 &&
        col !== 18 &&
        (isStar(row, col) ? <BoardStar /> : <BoardGeneral />)}
    </>
  );
};

export default BoardGrid;
