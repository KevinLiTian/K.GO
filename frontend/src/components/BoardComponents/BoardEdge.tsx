import { BoardEdgeProps } from '../../utils/Interfaces';

const BoardEdge = ({ style }: BoardEdgeProps) => {
  return (
    <g transform={style}>
      <rect x="17.5" y="16.5" width="1.0" height="19.5" fill="#533939" />
      <rect x="0.0" y="16.5" width="36.0" height="2.0" fill="#533939" />
    </g>
  );
};

export default BoardEdge;
