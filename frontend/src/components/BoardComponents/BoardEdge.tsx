import { BoardEdgeProps } from '../../utils/Interfaces';

const BoardEdge = ({ style }: BoardEdgeProps) => {
  return (
    <g transform={style}>
      <rect x="14.5" y="15.0" width="1.0" height="15.0" fill="#533939" />
      <rect x="0.0" y="14.0" width="30.0" height="2.0" fill="#533939" />
    </g>
  );
};

export default BoardEdge;