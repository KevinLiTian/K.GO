import { StoneProps } from '../../utils/Interfaces';

const StoneHover = ({ color }: StoneProps) => {
  return (
    <g>
      <circle cx="15" cy="15" r="12" fill={color} opacity={0.5} />
    </g>
  );
};

export default StoneHover;
