import { BoardCornerProps } from '../../utils/Interfaces';

const BoardCorner = ({ style }: BoardCornerProps) => {
  return (
    <g transform={style}>
      <rect x="16.5" y="16.5" width="2.0" height="19.5" fill="#533939" />
      <rect x="16.5" y="16.5" width="19.5" height="2.0" fill="#533939" />
    </g>
  );
};

export default BoardCorner;
