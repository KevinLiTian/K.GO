import { BoardCornerProps } from '../utils/Interfaces';

const BoardCorner = ({ style }: BoardCornerProps) => {
  return (
    <g transform={style}>
      <rect x="14.0" y="15.0" width="2.0" height="15.0" fill="#533939" />
      <rect x="14.0" y="14.0" width="16.0" height="2.0" fill="#533939" />
    </g>
  );
};

export default BoardCorner;
